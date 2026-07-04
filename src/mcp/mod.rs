//! MCP server for claude-chron session-history search.

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::LazyLock;

use mcp_server::router::{CapabilitiesBuilder, Router};
use mcp_spec::content::Content;
use mcp_spec::handler::{PromptError, ResourceError, ToolError};
use mcp_spec::prompt::Prompt;
use mcp_spec::protocol::ServerCapabilities;
use mcp_spec::resource::Resource;
use mcp_spec::tool::Tool;
use serde_json::Value;

use crate::config::Config;
use crate::embeddings::{EmbeddingProvider, OnnxEmbeddingProvider};
use crate::search::hybrid_search;
use crate::storage::{Storage, init_db};
use crate::transcript::parse_transcript;

static TOOLS: LazyLock<Vec<Tool>> = LazyLock::new(|| {
    [
        serde_json::json!({
            "name": "search",
            "description": "Search the user's past Claude Code conversations using hybrid semantic + keyword search. Use this when the user asks about past decisions, prior debugging sessions, previous code patterns, or anything that might have been discussed before. Returns ranked chunks with session_id and line_start — pass these to `expand` for more context.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Search query" },
                    "limit": { "type": "integer", "description": "Max results (default 10)", "default": 10 },
                    "project": { "type": "string", "description": "Filter by project path" }
                },
                "required": ["query"]
            }
        }),
        serde_json::json!({
            "name": "read_session",
            "description": "Read a session transcript. Returns parsed conversation turns with secrets redacted.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "session_id": { "type": "string", "description": "Session UUID" }
                },
                "required": ["session_id"]
            }
        }),
        serde_json::json!({
            "name": "expand",
            "description": "Get surrounding conversation turns for a chunk.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "session_id": { "type": "string", "description": "Session UUID from search result" },
                    "line_start": { "type": "integer", "description": "Chunk line_start from search result" },
                    "context": { "type": "integer", "description": "Number of turns before/after (default 3)", "default": 3 }
                },
                "required": ["session_id", "line_start"]
            }
        }),
        serde_json::json!({
            "name": "list_sessions",
            "description": "List indexed sessions.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project": { "type": "string", "description": "Filter by project path" },
                    "limit": { "type": "integer", "description": "Max results (default 20)", "default": 20 }
                }
            }
        }),
    ]
    .into_iter()
    .map(|value| serde_json::from_value(value).expect("built-in tool definition must be valid"))
    .collect()
});

#[derive(Clone)]
pub struct ChronRouter {
    db_path: PathBuf,
    busy_timeout_ms: u64,
    wal: bool,
    dimension: usize,
    embedding_provider: String,
    embedding_model: String,
    intra_threads: usize,
    /// Lazily-initialized, shared across the long-lived `serve` process so the
    /// ~558 MB ONNX model is loaded once instead of on every search.
    provider_cache: std::sync::Arc<std::sync::Mutex<Option<OnnxEmbeddingProvider>>>,
}

impl ChronRouter {
    pub fn new(config: Config) -> Self {
        Self {
            db_path: config.db.path,
            busy_timeout_ms: config.db.busy_timeout_ms,
            wal: config.db.wal,
            dimension: config.embedding.dimension,
            embedding_provider: config.embedding.provider,
            embedding_model: config.embedding.model,
            intra_threads: config.embedding.intra_threads,
            provider_cache: std::sync::Arc::new(std::sync::Mutex::new(None)),
        }
    }

    /// Embeds a single query, initializing and caching the provider on first
    /// use. Searches serialize on the embedding step, which is fast; the win is
    /// not reloading the model from disk every call.
    fn embed_query(&self, text: &str) -> Result<Vec<f32>, ToolError> {
        let mut guard = self
            .provider_cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if guard.is_none() {
            *guard = Some(self.open_provider()?);
        }
        let provider = guard.as_ref().expect("provider initialized above");
        provider
            .embed_texts(&[text])
            .map_err(|e| ToolError::ExecutionError(format!("embed query: {e}")))?
            .into_iter()
            .next()
            .ok_or_else(|| ToolError::ExecutionError("no embedding returned".into()))
    }

    fn open_storage(&self) -> Result<Storage, ToolError> {
        init_db(
            &self.db_path,
            self.busy_timeout_ms,
            self.wal,
            self.dimension,
        )
        .map_err(|e| ToolError::ExecutionError(format!("db open: {e}")))
    }

    fn open_provider(&self) -> Result<OnnxEmbeddingProvider, ToolError> {
        crate::embeddings::create_provider(
            &self.embedding_provider,
            &self.embedding_model,
            self.dimension,
            self.intra_threads,
        )
        .map_err(|e| ToolError::ExecutionError(format!("embedding: {e}")))
    }
}

impl Router for ChronRouter {
    fn name(&self) -> String {
        "claude-chron".to_string()
    }

    fn instructions(&self) -> String {
        "Search and recall information from the user's past Claude Code sessions. \
         Use `search` when the user's question could benefit from historical context — \
         past decisions, debugging approaches, code patterns, configuration choices, \
         or anything previously discussed. Especially trigger on questions like \
         'what did I decide about X', 'why did we do Y', 'how did I solve Z before', \
         'have I seen this before', or when the user references prior work without \
         providing full context. Typical flow: `search` for relevant chunks (top 3-5), \
         then `expand` to see surrounding conversation, or `read_session` for the full \
         transcript. Use `list_sessions` to browse by project or recency. \
         Skip when the question is purely about current code state (use Read/Grep), \
         ephemeral (today's task only), or the user explicitly asks to ignore history."
            .to_string()
    }

    fn capabilities(&self) -> ServerCapabilities {
        CapabilitiesBuilder::new().with_tools(false).build()
    }

    fn list_tools(&self) -> Vec<Tool> {
        TOOLS.clone()
    }

    fn call_tool(
        &self,
        tool_name: &str,
        arguments: Value,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<Content>, ToolError>> + Send + 'static>> {
        let this = self.clone();
        let tool_name = tool_name.to_string();
        Box::pin(async move {
            match tool_name.as_str() {
                "search" => this.handle_search(arguments),
                "expand" => this.handle_expand(arguments),
                "read_session" => this.handle_read_session(arguments),
                "list_sessions" => this.handle_list_sessions(arguments),
                _ => Err(ToolError::NotFound(format!("unknown tool: {tool_name}"))),
            }
        })
    }

    fn list_resources(&self) -> Vec<Resource> {
        vec![]
    }

    fn read_resource(
        &self,
        _uri: &str,
    ) -> Pin<Box<dyn Future<Output = Result<String, ResourceError>> + Send + 'static>> {
        Box::pin(async { Err(ResourceError::NotFound("no resources".into())) })
    }

    fn list_prompts(&self) -> Vec<Prompt> {
        vec![]
    }

    fn get_prompt(
        &self,
        _prompt_name: &str,
    ) -> Pin<Box<dyn Future<Output = Result<String, PromptError>> + Send + 'static>> {
        Box::pin(async { Err(PromptError::NotFound("no prompts".into())) })
    }
}

impl ChronRouter {
    fn handle_search(&self, arguments: Value) -> Result<Vec<Content>, ToolError> {
        let query = arguments
            .get("query")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidParameters("missing 'query'".into()))?;
        let limit = arguments
            .get("limit")
            .and_then(Value::as_u64)
            .unwrap_or(10)
            .min(100) as usize;
        let project = arguments.get("project").and_then(Value::as_str);

        let storage = self.open_storage()?;
        let query_embedding = self.embed_query(query)?;

        let results = hybrid_search(&storage, query, &query_embedding, limit, project)
            .map_err(|e| ToolError::ExecutionError(format!("search: {e}")))?;

        let json = serde_json::to_string_pretty(&results)
            .map_err(|e| ToolError::ExecutionError(format!("serialize: {e}")))?;
        Ok(vec![Content::text(json)])
    }

    /// Loads a session's turns, preferring the stored (already-redacted) copy
    /// so recall works even if the original transcript file was pruned. Falls
    /// back to parsing the transcript only when nothing is stored yet.
    fn load_turns(
        &self,
        storage: &Storage,
        session_id: &str,
    ) -> Result<Vec<crate::transcript::Turn>, ToolError> {
        let session = storage
            .get_session(session_id)
            .map_err(|e| ToolError::ExecutionError(format!("db: {e}")))?
            .ok_or_else(|| ToolError::ExecutionError(format!("session not found: {session_id}")))?;

        let stored = storage
            .get_turns(session_id)
            .map_err(|e| ToolError::ExecutionError(format!("db: {e}")))?;
        if !stored.is_empty() {
            // Stored turns are already redacted; redact again is idempotent and
            // protects against any row that bypassed the redacting insert path.
            return Ok(stored.into_iter().map(redact_turn).collect());
        }

        let transcript_path = PathBuf::from(&session.transcript_path);
        if !transcript_path.exists() {
            return Err(ToolError::ExecutionError(format!(
                "no stored turns and transcript file missing: {}",
                session.transcript_path.display()
            )));
        }
        let turns = parse_transcript(&transcript_path)
            .map_err(|e| ToolError::ExecutionError(format!("parse: {e}")))?;
        Ok(turns.into_iter().map(redact_turn).collect())
    }

    fn handle_read_session(&self, arguments: Value) -> Result<Vec<Content>, ToolError> {
        let session_id = arguments
            .get("session_id")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidParameters("missing 'session_id'".into()))?;

        let storage = self.open_storage()?;
        let turns = self.load_turns(&storage, session_id)?;

        let json = serde_json::to_string_pretty(&turns)
            .map_err(|e| ToolError::ExecutionError(format!("serialize: {e}")))?;
        Ok(vec![Content::text(json)])
    }

    fn handle_expand(&self, arguments: Value) -> Result<Vec<Content>, ToolError> {
        let session_id = arguments
            .get("session_id")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidParameters("missing 'session_id'".into()))?;
        let target_line = arguments
            .get("line_start")
            .and_then(Value::as_u64)
            .ok_or_else(|| ToolError::InvalidParameters("missing 'line_start'".into()))?
            as usize;
        let context = arguments
            .get("context")
            .and_then(Value::as_u64)
            .unwrap_or(3)
            .min(10) as usize;

        let storage = self.open_storage()?;
        let turns = self.load_turns(&storage, session_id)?;

        // Return the window of turns around the one nearest target_line.
        let target_idx = turns
            .iter()
            .position(|turn| turn.line_start >= target_line)
            .unwrap_or(turns.len().saturating_sub(1));
        let start = target_idx.saturating_sub(context);
        let end = (target_idx + context + 1).min(turns.len());
        let window: Vec<_> = turns[start..end].iter().cloned().map(redact_turn).collect();

        let json = serde_json::to_string_pretty(&window)
            .map_err(|e| ToolError::ExecutionError(format!("serialize: {e}")))?;
        Ok(vec![Content::text(json)])
    }

    fn handle_list_sessions(&self, arguments: Value) -> Result<Vec<Content>, ToolError> {
        let project = arguments.get("project").and_then(Value::as_str);
        let limit = arguments
            .get("limit")
            .and_then(Value::as_u64)
            .unwrap_or(20)
            .min(100) as usize;

        let storage = self.open_storage()?;
        let sessions = storage
            .list_sessions(project, limit)
            .map_err(|e| ToolError::ExecutionError(format!("db: {e}")))?;

        let json = serde_json::to_string_pretty(&sessions)
            .map_err(|e| ToolError::ExecutionError(format!("serialize: {e}")))?;
        Ok(vec![Content::text(json)])
    }
}

/// Redacts every text field of a turn — user, assistant, and tool summaries —
/// before it is returned over MCP. Tool summaries carry Bash commands, file
/// paths, and tool-result first lines, which routinely contain secrets.
fn redact_turn(mut turn: crate::transcript::Turn) -> crate::transcript::Turn {
    use crate::transcript::redact_or_placeholder;
    turn.user = redact_or_placeholder(&turn.user);
    turn.assistant = redact_or_placeholder(&turn.assistant);
    turn.tool_summary = turn
        .tool_summary
        .iter()
        .map(|summary| redact_or_placeholder(summary))
        .collect();
    turn
}
