use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use serde_json::Value;

use claude_chron::config::load_config;
use claude_chron::embeddings::{EmbeddingProvider, create_provider};
use claude_chron::indexer::index_all;
use claude_chron::mcp::ChronRouter;
use claude_chron::search::hybrid_search;
use claude_chron::storage::init_db;
use claude_chron::transcript::{chunk_turns, parse_transcript, scan_sessions};

#[derive(Debug, Parser)]
#[command(name = "cchron")]
#[command(version)]
#[command(about = "Local RAG search for Claude Code session history")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Index all Claude Code sessions into the local database.
    Index {
        /// Re-index sessions even if sha256 and mtime already exist.
        #[arg(long)]
        force: bool,
    },
    /// Search indexed session chunks.
    Search {
        /// Search query.
        query: String,
        /// Maximum number of results to return.
        #[arg(long, default_value_t = 10)]
        limit: usize,
        /// Restrict results to an exact decoded project path.
        #[arg(long)]
        project: Option<String>,
    },
    /// Show database status.
    Status,
    /// Run the MCP server.
    Serve,
    /// Print the merged configuration as JSON.
    Config {
        /// Project directory containing .claude-chron.toml.
        #[arg(long)]
        project_dir: Option<PathBuf>,
    },
    /// Scan the Claude Code projects directory for transcripts.
    Scan {
        /// Override the scanner root.
        #[arg(long)]
        root: Option<PathBuf>,
    },
    /// Stop hook: reads hook JSON from stdin and indexes the affected session.
    HookStop,
    /// SessionEnd hook: finalizes session indexing.
    HookSessionEnd,
    /// Migrate project path in the database (e.g. after renaming a project directory).
    Migrate {
        /// Old project path.
        #[arg(long)]
        from: String,
        /// New project path.
        #[arg(long)]
        to: String,
    },
    /// Set up claude-chron: register MCP server and run initial index.
    Setup,
    /// Print hook configuration for Claude Code settings.
    InstallHooks,
    /// Parse and chunk a transcript, printing chunks as JSON.
    Parse {
        /// Transcript JSONL file to parse.
        path: PathBuf,
        /// Session id to use in generated chunk identifiers.
        #[arg(long)]
        session_id: Option<String>,
        /// Project path to attach to generated chunks.
        #[arg(long)]
        project_path: Option<PathBuf>,
    },
    /// Re-apply redaction to every stored row, in place and idempotently.
    RedactDb,
}

fn run_hook(force: bool) {
    let input: Value = serde_json::from_reader(std::io::stdin())
        .unwrap_or_else(|_| Value::Object(serde_json::Map::new()));
    let transcript_path = input
        .get("transcript_path")
        .and_then(Value::as_str)
        .unwrap_or("");
    let path = PathBuf::from(transcript_path);
    if transcript_path.is_empty() || !path.exists() {
        return;
    }
    if let Ok(file) = std::fs::File::open(&path) {
        use std::io::{BufRead, BufReader};
        if BufReader::new(file).lines().take(3).count() < 3 {
            return;
        }
    }
    let config = load_config(None).unwrap_or_default();
    let Ok(mut storage) = init_db(
        &config.db.path,
        config.db.busy_timeout_ms,
        config.db.wal,
        config.embedding.dimension,
    ) else {
        return;
    };

    // Build session metadata with the same logic as a full scan (real file
    // mtime + cwd-derived project path) so incremental skipping and
    // project-scoped search stay consistent.
    if let (Ok(provider), Ok(Some(session))) = (
        create_provider(
            &config.embedding.provider,
            &config.embedding.model,
            config.embedding.dimension,
        ),
        claude_chron::transcript::session_file_from_path(&path),
    ) {
        let _ =
            claude_chron::indexer::index_session(&mut storage, &provider, &session, &config, force);
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli
        .command
        .unwrap_or(Commands::Config { project_dir: None })
    {
        Commands::Config { project_dir } => {
            let config = load_config(project_dir.as_deref()).context("failed to load config")?;
            println!("{}", serde_json::to_string_pretty(&config)?);
        }
        Commands::Index { force } => {
            let config = load_config(None).context("failed to load config")?;
            let mut storage = init_db(
                &config.db.path,
                config.db.busy_timeout_ms,
                config.db.wal,
                config.embedding.dimension,
            )
            .context("failed to initialize database")?;
            let provider = create_provider(
                &config.embedding.provider,
                &config.embedding.model,
                config.embedding.dimension,
            )
            .context("failed to initialize embedding provider")?;
            let stats = index_all(&mut storage, &provider, &config, force)
                .context("failed to index sessions")?;
            println!("{}", serde_json::to_string_pretty(&stats)?);
        }
        Commands::Search {
            query,
            limit,
            project,
        } => {
            let config = load_config(None).context("failed to load config")?;
            let storage = init_db(
                &config.db.path,
                config.db.busy_timeout_ms,
                config.db.wal,
                config.embedding.dimension,
            )
            .context("failed to initialize database")?;
            let provider = create_provider(
                &config.embedding.provider,
                &config.embedding.model,
                config.embedding.dimension,
            )
            .context("failed to initialize embedding provider")?;
            let query_refs = [query.as_str()];
            let query_embedding = provider
                .embed_texts(&query_refs)
                .context("failed to embed query")?
                .into_iter()
                .next()
                .context("embedding provider returned no query vector")?;
            let results = hybrid_search(
                &storage,
                &query,
                &query_embedding,
                limit,
                project.as_deref(),
            )
            .context("search failed")?;
            println!("{}", serde_json::to_string_pretty(&results)?);
        }
        Commands::Status => {
            let config = load_config(None).context("failed to load config")?;
            let storage = init_db(
                &config.db.path,
                config.db.busy_timeout_ms,
                config.db.wal,
                config.embedding.dimension,
            )
            .context("failed to initialize database")?;
            println!("{}", serde_json::to_string_pretty(&storage.stats()?)?);
        }
        Commands::Migrate { from, to } => {
            let config = load_config(None).context("failed to load config")?;
            let mut storage = init_db(
                &config.db.path,
                config.db.busy_timeout_ms,
                config.db.wal,
                config.embedding.dimension,
            )
            .context("failed to initialize database")?;
            let stats = storage
                .migrate_project(&from, &to)
                .context("failed to migrate project")?;
            println!("{}", serde_json::to_string_pretty(&stats)?);
        }
        Commands::Serve => {
            let config = load_config(None).context("failed to load config")?;
            let router = ChronRouter::new(config);
            let service = mcp_server::router::RouterService(router);
            let server = mcp_server::Server::new(service);
            let transport = mcp_server::ByteTransport::new(tokio::io::stdin(), tokio::io::stdout());
            server.run(transport).await.context("MCP server error")?;
        }
        Commands::Scan { root } => {
            let config = load_config(None).context("failed to load config")?;
            let root = root.unwrap_or(config.scanner.root);
            let sessions = scan_sessions(&root).context("failed to scan sessions")?;
            println!("{}", serde_json::to_string_pretty(&sessions)?);
        }
        Commands::Parse {
            path,
            session_id,
            project_path,
        } => {
            let config = load_config(None).context("failed to load config")?;
            let turns = parse_transcript(&path).context("failed to parse transcript")?;
            let session_id = session_id.unwrap_or_else(|| {
                path.file_stem()
                    .and_then(|stem| stem.to_str())
                    .unwrap_or("session")
                    .to_string()
            });
            let project_path = project_path
                .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
            let chunks = chunk_turns(
                &turns,
                &session_id,
                &project_path,
                &config.embedding.model,
                config.chunking.max_chars,
            )
            .context("failed to chunk transcript")?;
            println!("{}", serde_json::to_string_pretty(&chunks)?);
        }
        Commands::RedactDb => {
            let config = load_config(None).context("failed to load config")?;
            let mut storage = init_db(
                &config.db.path,
                config.db.busy_timeout_ms,
                config.db.wal,
                config.embedding.dimension,
            )
            .context("failed to initialize database")?;
            let (turns_redacted, chunks_redacted) = storage
                .reredact_stored()
                .context("failed to re-redact stored rows")?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "turns_redacted": turns_redacted,
                    "chunks_redacted": chunks_redacted
                }))?
            );
        }
        Commands::Setup => {
            let cchron_path =
                std::env::current_exe().context("failed to get current executable path")?;

            eprintln!("Registering MCP server...");
            let mcp_status = std::process::Command::new("claude")
                .args([
                    "mcp",
                    "add",
                    "claude-chron",
                    "--",
                    &cchron_path.to_string_lossy(),
                    "serve",
                ])
                .status();
            match mcp_status {
                Ok(s) if s.success() => eprintln!("MCP server registered."),
                Ok(s) => eprintln!(
                    "MCP registration exited with {s}. You can register manually:\n  claude mcp add claude-chron -- \"{}\" serve",
                    cchron_path.display()
                ),
                Err(e) => eprintln!(
                    "Could not run `claude`: {e}. Register manually:\n  claude mcp add claude-chron -- \"{}\" serve",
                    cchron_path.display()
                ),
            }

            eprintln!("\nRunning initial index (this may take a while)...");
            let config = load_config(None).unwrap_or_default();
            if let Ok(mut storage) = init_db(
                &config.db.path,
                config.db.busy_timeout_ms,
                config.db.wal,
                config.embedding.dimension,
            ) {
                if let Ok(provider) = create_provider(
                    &config.embedding.provider,
                    &config.embedding.model,
                    config.embedding.dimension,
                ) {
                    match index_all(&mut storage, &provider, &config, false) {
                        Ok(stats) => {
                            eprintln!(
                                "Indexed {} sessions ({} skipped, {} failed).",
                                stats.indexed, stats.skipped, stats.failed
                            );
                            for failure in &stats.failures {
                                eprintln!("  failed: {} ({})", failure.session_id, failure.error);
                            }
                        }
                        Err(e) => eprintln!("Index error: {e}. Run `cchron index` to retry."),
                    }
                } else {
                    eprintln!("Embedding provider failed. Run `cchron index` to retry.");
                }
            }

            eprintln!("\nDone. Search with: cchron search \"your query\"");
            eprintln!("\nTo enable automatic indexing of new sessions, run:");
            eprintln!("  cchron install-hooks");
            eprintln!("and add the printed JSON to ~/.claude/settings.json.");
        }
        Commands::HookStop => {
            run_hook(false);
        }
        Commands::HookSessionEnd => {
            run_hook(true);
        }
        Commands::InstallHooks => {
            let cchron_path =
                std::env::current_exe().context("failed to get current executable path")?;
            // Mirror the packaged plugin's hook schema (event → group →
            // `hooks` array) so manual and plugin installs behave identically.
            let hooks_json = serde_json::json!({
                "hooks": {
                    "Stop": [{
                        "hooks": [{
                            "type": "command",
                            "command": format!("\"{}\" hook-stop", cchron_path.display()),
                            "async": true,
                            "timeout": 120,
                        }]
                    }],
                    "SessionEnd": [{
                        "hooks": [{
                            "type": "command",
                            "command": format!("\"{}\" hook-session-end", cchron_path.display()),
                            "async": true,
                            "timeout": 30,
                        }]
                    }]
                }
            });
            println!("Add the following to ~/.claude/settings.json:\n");
            println!("{}", serde_json::to_string_pretty(&hooks_json)?);
        }
    }
    Ok(())
}
