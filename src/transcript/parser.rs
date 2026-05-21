//! Parser for Claude Code JSONL transcripts.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::util::{
    char_count, find_ascii_case_insensitive, strip_case_insensitive_tag, take_chars,
};

const SKIP_TYPES: &[&str] = &["progress", "system", "permission-mode", "last-prompt"];

/// A single user request plus assistant response from a transcript.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Turn {
    /// Cleaned user request text.
    pub user: String,
    /// Cleaned assistant response text.
    pub assistant: String,
    /// Concise summaries of tool calls and tool results associated with the turn.
    pub tool_summary: Vec<String>,
    /// One-based JSONL line where the turn started.
    pub line_start: usize,
    /// One-based JSONL line where the turn ended.
    pub line_end: usize,
    /// Transcript timestamp, if present.
    pub ts: Option<String>,
}

/// Errors that can occur while parsing transcripts.
#[derive(Debug, thiserror::Error)]
pub enum ParserError {
    /// Transcript file could not be opened or read.
    #[error("failed to read transcript")]
    Io(#[from] std::io::Error),
    /// A JSONL line could not be parsed.
    #[error("failed to parse JSON at line {line}")]
    Json {
        /// One-based line number.
        line: usize,
        /// Source JSON error.
        source: serde_json::Error,
    },
}

#[derive(Debug)]
struct TurnBuilder {
    user: String,
    assistant_parts: Vec<String>,
    tool_summary: Vec<String>,
    line_start: usize,
    line_end: usize,
    ts: Option<String>,
}

impl TurnBuilder {
    fn build(self) -> Turn {
        Turn {
            user: self.user,
            assistant: self
                .assistant_parts
                .into_iter()
                .filter(|part| !part.is_empty())
                .collect::<Vec<_>>()
                .join("\n\n")
                .trim()
                .to_string(),
            tool_summary: self.tool_summary,
            line_start: self.line_start,
            line_end: self.line_end,
            ts: self.ts,
        }
    }
}

/// Parses a Claude Code JSONL transcript file.
///
/// # Errors
/// Returns [`ParserError`] when the file cannot be read or contains invalid JSON.
pub fn parse_transcript(path: &Path) -> Result<Vec<Turn>, ParserError> {
    let file = File::open(path)?;
    parse_jsonl_reader(BufReader::new(file))
}

/// Parses Claude Code JSONL text.
///
/// # Errors
/// Returns [`ParserError`] when any non-empty line contains invalid JSON.
pub fn parse_jsonl_lines(lines: &str) -> Result<Vec<Turn>, ParserError> {
    parse_jsonl_reader(BufReader::new(lines.as_bytes()))
}

/// Parses Claude Code JSONL from a buffered reader.
///
/// # Errors
/// Returns [`ParserError`] when reading fails or any non-empty line contains invalid JSON.
pub fn parse_jsonl_reader<R>(reader: R) -> Result<Vec<Turn>, ParserError>
where
    R: BufRead,
{
    let mut turns = Vec::new();
    let mut current: Option<TurnBuilder> = None;

    for (index, raw_line) in reader.lines().enumerate() {
        let line_number = index + 1;
        let raw_line = raw_line?;
        if raw_line.trim().is_empty() {
            continue;
        }

        let record: Value =
            serde_json::from_str(&raw_line).map_err(|source| ParserError::Json {
                line: line_number,
                source,
            })?;
        let event_type = value_str(record.get("type"));
        if should_skip_event(event_type) || is_hook_injection(&record) {
            continue;
        }

        let message = message(&record);
        let content = message.and_then(|message| message.get("content"));
        let role = message
            .and_then(|message| message.get("role"))
            .and_then(Value::as_str);

        if event_type == "user" || role == Some("user") {
            let (user_text, tool_results) = parse_user_content(content);
            if !tool_results.is_empty()
                && let Some(builder) = current.as_mut()
            {
                builder.tool_summary.extend(tool_results);
                builder.line_end = line_number;
            }

            if user_text.is_empty() {
                continue;
            }

            if let Some(builder) = current.take() {
                turns.push(builder.build());
            }

            current = Some(TurnBuilder {
                user: user_text,
                assistant_parts: Vec::new(),
                tool_summary: Vec::new(),
                line_start: line_number,
                line_end: line_number,
                ts: timestamp(&record),
            });
            continue;
        }

        if event_type == "assistant" || role == Some("assistant") {
            let Some(builder) = current.as_mut() else {
                continue;
            };
            let (assistant_text, tool_calls) = parse_assistant_content(content);
            if !assistant_text.is_empty() {
                builder.assistant_parts.push(assistant_text);
            }
            builder.tool_summary.extend(tool_calls);
            builder.line_end = line_number;
        }
    }

    if let Some(builder) = current {
        turns.push(builder.build());
    }

    Ok(turns)
}

fn should_skip_event(event_type: &str) -> bool {
    SKIP_TYPES.contains(&event_type)
}

fn message(record: &Value) -> Option<&serde_json::Map<String, Value>> {
    record.get("message").and_then(Value::as_object)
}

fn timestamp(record: &Value) -> Option<String> {
    record
        .get("timestamp")
        .or_else(|| record.get("created_at"))
        .and_then(value_to_plain_string)
}

fn parse_user_content(content: Option<&Value>) -> (String, Vec<String>) {
    match content {
        Some(Value::String(text)) => (clean_user_text(text), Vec::new()),
        Some(Value::Array(blocks)) => {
            let mut text_parts = Vec::new();
            let mut tool_results = Vec::new();
            for block in blocks {
                let Some(block) = block.as_object() else {
                    continue;
                };
                match value_str(block.get("type")) {
                    "tool_result" => tool_results.push(summarize_tool_result(block)),
                    "text" => {
                        if let Some(text) = block.get("text").and_then(value_to_plain_string) {
                            text_parts.push(text);
                        }
                    }
                    _ => {
                        if let Some(text) = block.get("content").and_then(Value::as_str) {
                            text_parts.push(text.to_string());
                        }
                    }
                }
            }
            (clean_user_text(&text_parts.join("\n")), tool_results)
        }
        _ => (String::new(), Vec::new()),
    }
}

fn parse_assistant_content(content: Option<&Value>) -> (String, Vec<String>) {
    match content {
        Some(Value::String(text)) => (clean_text(text), Vec::new()),
        Some(Value::Array(blocks)) => {
            let mut text_parts = Vec::new();
            let mut tool_calls = Vec::new();
            for block in blocks {
                let Some(block) = block.as_object() else {
                    continue;
                };
                match value_str(block.get("type")) {
                    "text" => {
                        if let Some(text) = block.get("text").and_then(value_to_plain_string) {
                            text_parts.push(text);
                        }
                    }
                    "tool_use" => tool_calls.push(summarize_tool_use(block)),
                    _ => {}
                }
            }
            (clean_text(&text_parts.join("\n")), tool_calls)
        }
        _ => (String::new(), Vec::new()),
    }
}

fn clean_user_text(text: &str) -> String {
    clean_text(&strip_case_insensitive_tag(
        text,
        "<system-reminder>",
        "</system-reminder>",
    ))
}

fn clean_text(text: &str) -> String {
    text.replace("\r\n", "\n")
        .split('\n')
        .map(collapse_horizontal_whitespace)
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

fn summarize_tool_use(block: &serde_json::Map<String, Value>) -> String {
    let name = block
        .get("name")
        .and_then(value_to_plain_string)
        .unwrap_or_else(|| "tool".to_string());
    let Some(input) = block.get("input").and_then(Value::as_object) else {
        return name;
    };

    match name.as_str() {
        "Bash" => input
            .get("command")
            .and_then(value_to_plain_string)
            .map(|command| truncate(&command, 160))
            .filter(|command| !command.is_empty())
            .map_or_else(|| "Bash".to_string(), |command| format!("Bash: {command}")),
        "Read" | "Edit" | "MultiEdit" | "Write" => input
            .get("file_path")
            .or_else(|| input.get("path"))
            .and_then(value_to_plain_string)
            .map_or_else(|| name.clone(), |path| format!("{name}: {path}")),
        "Grep" | "Glob" => {
            let details = ["pattern", "path"]
                .into_iter()
                .filter_map(|key| input.get(key).and_then(value_to_plain_string))
                .filter(|value| !value.is_empty())
                .collect::<Vec<_>>()
                .join(", ");
            if details.is_empty() {
                name
            } else {
                format!("{name}: {details}")
            }
        }
        "TodoWrite" => input.get("todos").and_then(Value::as_array).map_or_else(
            || name.clone(),
            |todos| format!("TodoWrite: {} todos", todos.len()),
        ),
        _ => {
            let pairs = input
                .iter()
                .filter(|(key, _)| !matches!(key.as_str(), "content" | "old_string" | "new_string"))
                .take(3)
                .map(|(key, value)| {
                    let value = value_to_plain_string(value).unwrap_or_else(|| value.to_string());
                    format!("{key}={}", truncate(&value, 80))
                })
                .collect::<Vec<_>>();
            if pairs.is_empty() {
                name
            } else {
                format!("{name}: {}", pairs.join(", "))
            }
        }
    }
}

fn summarize_tool_result(block: &serde_json::Map<String, Value>) -> String {
    let content = block.get("content").map_or_else(String::new, |content| {
        if let Some(items) = content.as_array() {
            items
                .iter()
                .filter_map(|item| item.as_object())
                .filter_map(|item| item.get("text"))
                .filter_map(value_to_plain_string)
                .collect::<Vec<_>>()
                .join(" ")
        } else {
            value_to_plain_string(content).unwrap_or_else(|| content.to_string())
        }
    });

    let text = clean_text(&content);
    let prefix = if block
        .get("is_error")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        "Tool result error"
    } else {
        "Tool result"
    };

    text.lines().next().map_or_else(
        || prefix.to_string(),
        |line| format!("{prefix}: {}", truncate(line, 120)),
    )
}

/// Detects hook-injection events, which must not be indexed as real turns.
/// Three independent signals are checked: a structured `hook_event_name`/`hook`
/// field, a `type` value in the `hook*` family, and — for events carrying
/// neither — a hook marker embedded in the message content.
fn is_hook_injection(record: &Value) -> bool {
    let hook_field_present = record
        .get("hook_event_name")
        .or_else(|| record.get("hook"))
        .is_some_and(|value| !value.is_null());
    if hook_field_present {
        return true;
    }

    let event_type = value_str(record.get("type"));
    if event_type.starts_with("hook") {
        return true;
    }

    let Some(content) = message(record)
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str)
    else {
        return false;
    };
    find_ascii_case_insensitive(content, "<hook").is_some()
        || find_ascii_case_insensitive(content, "hook injection").is_some()
}

fn value_str(value: Option<&Value>) -> &str {
    value.and_then(Value::as_str).unwrap_or("")
}

fn value_to_plain_string(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.to_string()),
        Value::Number(number) => Some(number.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}

fn collapse_horizontal_whitespace(line: &str) -> String {
    let mut output = String::with_capacity(line.len());
    let mut previous_was_space = false;
    for ch in line.chars() {
        if ch == ' ' || ch == '\t' {
            if !previous_was_space {
                output.push(' ');
            }
            previous_was_space = true;
        } else {
            output.push(ch);
            previous_was_space = false;
        }
    }
    output.trim_end().to_string()
}

fn truncate(text: &str, limit: usize) -> String {
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if char_count(&text) <= limit {
        return text;
    }
    let mut truncated = take_chars(&text, limit.saturating_sub(1));
    truncated = truncated.trim_end().to_string();
    truncated.push_str("...");
    truncated
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_jsonl_lines_should_group_user_and_assistant() {
        let jsonl = r#"
{"type":"user","timestamp":"2026-05-13T01:00:00Z","message":{"role":"user","content":"Hello <system-reminder>hide</system-reminder> world"}}
{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Hi there"},{"type":"tool_use","name":"Bash","input":{"command":"echo hello"}}]}}
"#;

        let turns = parse_jsonl_lines(jsonl).expect("transcript should parse");

        assert_eq!(
            turns,
            vec![Turn {
                user: "Hello world".to_string(),
                assistant: "Hi there".to_string(),
                tool_summary: vec!["Bash: echo hello".to_string()],
                line_start: 2,
                line_end: 3,
                ts: Some("2026-05-13T01:00:00Z".to_string()),
            }]
        );
    }

    #[test]
    fn parse_jsonl_lines_should_skip_hook_injections() {
        let jsonl = r#"
{"type":"user","message":{"role":"user","content":"keep"}}
{"type":"assistant","message":{"role":"assistant","content":"answer"}}
{"type":"hook_event","message":{"role":"user","content":"ignore"}}
"#;

        let turns = parse_jsonl_lines(jsonl).expect("transcript should parse");

        assert_eq!(turns.len(), 1);
    }

    #[test]
    fn parse_jsonl_lines_should_attach_tool_results_to_current_turn() {
        let jsonl = r#"
{"type":"user","message":{"role":"user","content":"run it"}}
{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","name":"Read","input":{"file_path":"/tmp/a.txt"}}]}}
{"type":"user","message":{"role":"user","content":[{"type":"tool_result","content":"first line\nsecond line"}]}}
"#;

        let turns = parse_jsonl_lines(jsonl).expect("transcript should parse");

        assert_eq!(
            turns[0].tool_summary,
            vec![
                "Read: /tmp/a.txt".to_string(),
                "Tool result: first line".to_string()
            ]
        );
    }

    #[test]
    fn parse_jsonl_lines_should_report_invalid_json_line() {
        let err = parse_jsonl_lines("{\"type\":\"user\"}\nnot-json\n")
            .expect_err("invalid JSON should fail");

        assert!(matches!(err, ParserError::Json { line: 2, .. }));
    }
}
