//! Conversation-based chunking for parsed transcript turns.

use std::path::Path;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::util::{char_count, strip_case_insensitive_tag, take_chars};

use super::parser::Turn;
use super::redactor::{RedactionError, redact_or_placeholder, redact_text};

/// A searchable transcript chunk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chunk {
    /// Stable identifier, formatted as
    /// `session_id:line_start:turn_number:part_number:content_hash`.
    pub chunk_uid: String,
    /// Source Claude Code session identifier.
    pub session_id: String,
    /// Decoded project path.
    pub project_path: String,
    /// Short title derived from the user request.
    pub title: Option<String>,
    /// Redacted full chunk text.
    pub text: String,
    /// Cleaned text used for embedding.
    pub text_for_embedding: String,
    /// One-based source start line.
    pub line_start: Option<usize>,
    /// One-based source end line.
    pub line_end: Option<usize>,
    /// Approximate token count based on whitespace.
    pub token_count: usize,
    /// Source creation timestamp.
    pub created_at: Option<String>,
    /// Embedding model used for this chunk.
    pub embedding_model: String,
    /// First 16 hex characters of the SHA-256 embedding text hash.
    pub content_hash: String,
}

/// Errors that can occur while chunking transcripts.
#[derive(Debug, thiserror::Error)]
pub enum ChunkError {
    /// `max_chars` must be greater than zero.
    #[error("max_chars must be greater than zero")]
    InvalidMaxChars,
    /// Redaction failed.
    #[error(transparent)]
    Redaction(#[from] RedactionError),
}

/// Converts parsed turns into conversation chunks.
///
/// # Errors
/// Returns [`ChunkError`] when `max_chars` is zero or redaction fails.
pub fn chunk_turns(
    turns: &[Turn],
    session_id: &str,
    project_path: &Path,
    embedding_model: &str,
    max_chars: usize,
) -> Result<Vec<Chunk>, ChunkError> {
    if max_chars == 0 {
        return Err(ChunkError::InvalidMaxChars);
    }

    let mut chunks = Vec::new();
    let project_path_text = project_path.to_string_lossy().to_string();
    for (turn_index, turn) in turns.iter().enumerate() {
        let text = redact_text(&render_turn(turn), None)?;
        for (part_index, part) in split_text(&text, max_chars).into_iter().enumerate() {
            let text_for_embedding = clean_for_embedding(&part);
            let content_hash = content_hash(&text_for_embedding);
            chunks.push(Chunk {
                chunk_uid: format!(
                    "{session_id}:{}:{}:{}:{content_hash}",
                    turn.line_start,
                    turn_index + 1,
                    part_index + 1
                ),
                session_id: session_id.to_string(),
                project_path: project_path_text.clone(),
                title: Some(redact_or_placeholder(&title_from_turn(turn))),
                text: part,
                text_for_embedding: text_for_embedding.clone(),
                line_start: nonzero_line(turn.line_start),
                line_end: nonzero_line(turn.line_end),
                token_count: estimate_token_count(&text_for_embedding),
                created_at: normalize_timestamp(turn.ts.as_deref()),
                embedding_model: embedding_model.to_string(),
                content_hash,
            });
        }
    }

    Ok(chunks)
}

/// Cleans text for embedding by removing noisy markup and truncating logs.
pub fn clean_for_embedding(text: &str) -> String {
    let cleaned = strip_case_insensitive_tag(text, "<!--", "-->");
    let cleaned = strip_case_insensitive_tag(&cleaned, "<system-reminder>", "</system-reminder>");
    let cleaned = clean_fenced_blocks(&cleaned);
    let cleaned = drop_long_log_runs(&cleaned);
    let cleaned = cleaned
        .lines()
        .map(|line| truncate_long_line(line, 500))
        .collect::<Vec<_>>()
        .join("\n");
    collapse_blank_lines(&cleaned).trim().to_string()
}

fn render_turn(turn: &Turn) -> String {
    let mut parts = vec![format!("User: {}", turn.user)];
    if !turn.assistant.is_empty() {
        parts.push(format!("Assistant: {}", turn.assistant));
    }
    if !turn.tool_summary.is_empty() {
        parts.push(format!("Tools:\n- {}", turn.tool_summary.join("\n- ")));
    }
    parts.join("\n\n").trim().to_string()
}

fn split_text(text: &str, max_chars: usize) -> Vec<String> {
    if char_count(text) <= max_chars {
        return vec![text.to_string()];
    }

    let mut parts = Vec::new();
    let mut current = String::new();
    for paragraph in paragraphs(text) {
        let candidate = if current.is_empty() {
            paragraph.clone()
        } else {
            format!("{current}\n\n{paragraph}")
        };
        if char_count(&candidate) <= max_chars {
            current = candidate;
            continue;
        }
        if !current.trim().is_empty() {
            parts.extend(hard_split(current.trim(), max_chars));
        }
        current = paragraph;
    }

    if !current.trim().is_empty() {
        parts.extend(hard_split(current.trim(), max_chars));
    }

    parts
        .into_iter()
        .map(|part| part.trim().to_string())
        .filter(|part| !part.is_empty())
        .collect()
}

fn paragraphs(text: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            if !current.is_empty() {
                parts.push(current.join("\n"));
                current.clear();
            }
        } else {
            current.push(line.to_string());
        }
    }
    if !current.is_empty() {
        parts.push(current.join("\n"));
    }
    parts
}

fn hard_split(text: &str, max_chars: usize) -> Vec<String> {
    if char_count(text) <= max_chars {
        return vec![text.to_string()];
    }

    let mut parts = Vec::new();
    let mut remaining = text.trim().to_string();
    while char_count(&remaining) > max_chars {
        let split_at = preferred_split_byte_index(&remaining, max_chars);
        let part = remaining[..split_at].trim().to_string();
        if !part.is_empty() {
            parts.push(part);
        }
        remaining = remaining[split_at..].trim().to_string();
    }
    if !remaining.is_empty() {
        parts.push(remaining);
    }
    parts
}

fn preferred_split_byte_index(text: &str, max_chars: usize) -> usize {
    let max_byte = byte_index_after_chars(text, max_chars);
    let half_byte = byte_index_after_chars(text, max_chars / 2);
    if let Some(index) = text[..max_byte].rfind('\n')
        && index >= half_byte
    {
        return index;
    }
    if let Some(index) = text[..max_byte].rfind(' ')
        && index >= half_byte
    {
        return index;
    }
    max_byte
}

fn byte_index_after_chars(text: &str, limit: usize) -> usize {
    if limit == 0 {
        return 0;
    }
    text.char_indices()
        .nth(limit)
        .map_or_else(|| text.len(), |(index, _)| index)
}

fn clean_fenced_blocks(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut remaining = text;
    while let Some(start) = remaining.find("```") {
        output.push_str(&remaining[..start]);
        let after_start = start + 3;
        let Some(end_relative) = remaining[after_start..].find("```") else {
            output.push_str(&remaining[start..]);
            return output;
        };
        let end = after_start + end_relative + 3;
        let block = &remaining[start..end];
        let body = &remaining[after_start..after_start + end_relative];
        if char_count(body) > 1200 || body.lines().count() > 40 {
            output.push_str("```[long output omitted]```");
        } else {
            output.push_str(block);
        }
        remaining = &remaining[end..];
    }
    output.push_str(remaining);
    output
}

fn drop_long_log_runs(text: &str) -> String {
    let lines = text.lines().collect::<Vec<_>>();
    if lines.len() <= 120 {
        return text.to_string();
    }
    let mut kept = lines[..80].to_vec();
    kept.push("[long output omitted]");
    kept.extend_from_slice(&lines[lines.len() - 20..]);
    kept.join("\n")
}

fn truncate_long_line(line: &str, limit: usize) -> String {
    if char_count(line) <= limit {
        return line.trim_end().to_string();
    }
    format!("{} [truncated]", take_chars(line, limit).trim_end())
}

fn collapse_blank_lines(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut blank_count = 0usize;
    for line in text.lines() {
        if line.trim().is_empty() {
            blank_count += 1;
            if blank_count <= 2 {
                output.push('\n');
            }
        } else {
            blank_count = 0;
            if !output.is_empty() && !output.ends_with('\n') {
                output.push('\n');
            }
            output.push_str(line);
            output.push('\n');
        }
    }
    output
}

fn title_from_turn(turn: &Turn) -> String {
    let first_line = turn.user.lines().next().unwrap_or("Conversation");
    if first_line.is_empty() {
        return "Conversation".to_string();
    }
    if char_count(first_line) <= 80 {
        return first_line.to_string();
    }
    format!("{}...", take_chars(first_line, 79).trim_end())
}

fn estimate_token_count(text: &str) -> usize {
    if text.is_empty() {
        0
    } else {
        text.split_whitespace().count().max(1)
    }
}

fn content_hash(text: &str) -> String {
    let digest = Sha256::digest(text.as_bytes());
    hex::encode(&digest[..8])
}

fn normalize_timestamp(timestamp: Option<&str>) -> Option<String> {
    timestamp.map(|timestamp| {
        DateTime::parse_from_rfc3339(timestamp).map_or_else(
            |_| timestamp.to_string(),
            |datetime| datetime.with_timezone(&Utc).to_rfc3339(),
        )
    })
}

fn nonzero_line(line: usize) -> Option<usize> {
    if line == 0 { None } else { Some(line) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_turn() -> Turn {
        Turn {
            user: "How do I run this?".to_string(),
            assistant: "Use cargo test.".to_string(),
            tool_summary: vec!["Bash: cargo test".to_string()],
            line_start: 3,
            line_end: 4,
            ts: Some("2026-05-13T01:00:00Z".to_string()),
        }
    }

    #[test]
    fn chunk_turns_should_create_conversation_chunk() {
        let turn = fixture_turn();

        let chunks = chunk_turns(
            &[turn],
            "session-1",
            Path::new("/tmp/project"),
            "model-name",
            800,
        )
        .expect("chunking should work");

        assert_eq!(chunks[0].title, Some("How do I run this?".to_string()));
    }

    #[test]
    fn chunk_turns_should_split_long_chunks() {
        let turn = Turn {
            user: "a ".repeat(200),
            assistant: "b ".repeat(200),
            tool_summary: Vec::new(),
            line_start: 1,
            line_end: 2,
            ts: None,
        };

        let chunks = chunk_turns(
            &[turn],
            "session-1",
            Path::new("/tmp/project"),
            "model",
            120,
        )
        .expect("chunking should work");

        assert!(chunks.len() > 1);
    }

    #[test]
    fn clean_for_embedding_should_remove_comments_and_long_blocks() {
        let block_body = (0..45)
            .map(|index| format!("line {index}"))
            .collect::<Vec<_>>()
            .join("\n");
        let text = format!("keep <!-- hidden -->\n```{block_body}```");

        let cleaned = clean_for_embedding(&text);

        assert_eq!(cleaned, "keep\n```[long output omitted]```");
    }

    #[test]
    fn clean_for_embedding_should_truncate_long_lines() {
        let text = "x".repeat(510);

        let cleaned = clean_for_embedding(&text);

        assert!(cleaned.ends_with("[truncated]"));
    }
}
