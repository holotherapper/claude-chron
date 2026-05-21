//! Scanner for Claude Code transcript files.

use std::fs;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A discovered Claude Code session transcript file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionFile {
    /// Session identifier, derived from the JSONL filename stem.
    pub session_id: String,
    /// Decoded project path from the parent directory.
    pub project_path: String,
    /// Transcript JSONL path.
    pub transcript_path: PathBuf,
    /// File modification time.
    pub mtime: DateTime<Utc>,
    /// File size in bytes.
    pub size: u64,
}

/// Errors that can occur while scanning for transcripts.
#[derive(Debug, thiserror::Error)]
pub enum ScannerError {
    /// A directory entry could not be read.
    #[error("failed to read directory entry under {path}")]
    ReadDir {
        /// Directory being scanned.
        path: PathBuf,
        /// Source I/O error.
        source: std::io::Error,
    },
    /// Metadata could not be read for a transcript file.
    #[error("failed to read metadata for {path}")]
    Metadata {
        /// File path.
        path: PathBuf,
        /// Source I/O error.
        source: std::io::Error,
    },
    /// Modified time could not be read for a transcript file.
    #[error("failed to read modified time for {path}")]
    Modified {
        /// File path.
        path: PathBuf,
        /// Source I/O error.
        source: std::io::Error,
    },
}

/// Scans a Claude Code projects root for JSONL transcript files.
///
/// Missing roots return an empty list.
///
/// # Errors
/// Returns [`ScannerError`] when directory traversal or file metadata reads fail.
pub fn scan_sessions(root: &Path) -> Result<Vec<SessionFile>, ScannerError> {
    let root = crate::util::expand_path(root);
    if !root.exists() {
        return Ok(Vec::new());
    }

    let mut sessions = Vec::new();
    scan_dir(&root, &mut sessions)?;
    sessions.sort_by(|left, right| {
        right
            .mtime
            .cmp(&left.mtime)
            .then_with(|| right.transcript_path.cmp(&left.transcript_path))
    });
    Ok(sessions)
}

fn scan_dir(dir: &Path, sessions: &mut Vec<SessionFile>) -> Result<(), ScannerError> {
    let entries = fs::read_dir(dir).map_err(|source| ScannerError::ReadDir {
        path: dir.to_path_buf(),
        source,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| ScannerError::ReadDir {
            path: dir.to_path_buf(),
            source,
        })?;
        let file_type = entry.file_type().map_err(|source| ScannerError::Metadata {
            path: entry.path(),
            source,
        })?;
        // Do not follow symlinks: a symlinked directory could form a cycle and
        // recurse indefinitely.
        if file_type.is_symlink() {
            continue;
        }
        let path = entry.path();
        if file_type.is_dir() {
            scan_dir(&path, sessions)?;
            continue;
        }
        if let Some(session) = session_file_from_path(&path)? {
            sessions.push(session);
        }
    }
    Ok(())
}

/// Builds the canonical [`SessionFile`] for a single transcript path.
///
/// This is the single source of truth for session metadata so the Stop hook
/// and a full scan agree, keeping incremental skipping and project-scoped
/// search consistent. The project path is taken from the transcript's `cwd`
/// field, falling back to decoding the parent directory name. Returns
/// `Ok(None)` for non-transcript or non-file paths.
///
/// # Errors
/// Returns [`ScannerError`] when file metadata cannot be read.
pub fn session_file_from_path(path: &Path) -> Result<Option<SessionFile>, ScannerError> {
    if path.extension().and_then(|extension| extension.to_str()) != Some("jsonl") {
        return Ok(None);
    }
    let metadata = fs::metadata(path).map_err(|source| ScannerError::Metadata {
        path: path.to_path_buf(),
        source,
    })?;
    if !metadata.is_file() {
        return Ok(None);
    }
    let modified = metadata
        .modified()
        .map_err(|source| ScannerError::Modified {
            path: path.to_path_buf(),
            source,
        })?;
    let session_id = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or_default()
        .to_string();
    let project_path = read_cwd_from_jsonl(path).unwrap_or_else(|| {
        path.parent()
            .and_then(Path::file_name)
            .and_then(|name| name.to_str())
            .map_or_else(String::new, decode_project_path)
    });
    Ok(Some(SessionFile {
        session_id,
        project_path,
        transcript_path: path.to_path_buf(),
        mtime: DateTime::<Utc>::from(modified),
        size: metadata.len(),
    }))
}

/// Reads the `cwd` field from the first matching line in a JSONL transcript.
fn read_cwd_from_jsonl(path: &Path) -> Option<String> {
    use std::io::{BufRead, BufReader};

    let file = fs::File::open(path).ok()?;
    let reader = BufReader::new(file);
    for line in reader.lines().take(20) {
        let Ok(line) = line else {
            continue;
        };
        if let Ok(obj) = serde_json::from_str::<serde_json::Value>(&line)
            && let Some(cwd) = obj.get("cwd").and_then(|v| v.as_str())
        {
            return Some(cwd.to_string());
        }
    }
    None
}

fn decode_project_path(directory_name: &str) -> String {
    directory_name.strip_prefix('-').map_or_else(
        || directory_name.to_string(),
        |rest| format!("/{}", rest.replace('-', "/")),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_project_path_should_restore_leading_slash() {
        let decoded = decode_project_path("-Users-alice-Projects-demo");

        assert_eq!(decoded, "/Users/alice/Projects/demo");
    }

    #[test]
    fn scan_sessions_should_find_jsonl_files() {
        let temp =
            std::env::temp_dir().join(format!("claude_chron_scanner_test_{}", std::process::id()));
        let project_dir = temp.join("-tmp-project");
        fs::create_dir_all(&project_dir).expect("project dir should be created");
        fs::write(project_dir.join("session-1.jsonl"), "{}\n").expect("jsonl should be written");
        fs::write(project_dir.join("ignore.txt"), "").expect("txt should be written");

        let sessions = scan_sessions(&temp).expect("scan should work");

        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].session_id, "session-1");
        assert_eq!(sessions[0].project_path, "/tmp/project");
        fs::remove_dir_all(temp).expect("temp dir should be removed");
    }
}
