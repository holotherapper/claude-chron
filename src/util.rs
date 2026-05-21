//! Shared utilities.

use std::path::{Path, PathBuf};

/// Expands a leading `~` or `~/` to the user's home directory.
pub fn expand_path(path: &Path) -> PathBuf {
    let text = path.to_string_lossy();
    if text == "~" {
        return dirs::home_dir().unwrap_or_else(|| path.to_path_buf());
    }
    if let Some(rest) = text.strip_prefix("~/")
        && let Some(home) = dirs::home_dir()
    {
        return home.join(rest);
    }
    path.to_path_buf()
}

/// Removes all occurrences of a case-insensitive tag pair from text.
pub fn strip_case_insensitive_tag(text: &str, open_tag: &str, close_tag: &str) -> String {
    let mut output = text.to_string();
    while let Some(start) = find_ascii_case_insensitive(&output, open_tag) {
        let after_open = start + open_tag.len();
        let Some(relative_end) = find_ascii_case_insensitive(&output[after_open..], close_tag)
        else {
            break;
        };
        let end = after_open + relative_end + close_tag.len();
        output.replace_range(start..end, "");
    }
    output
}

/// Finds the byte index of the first ASCII-case-insensitive match of `needle`.
pub fn find_ascii_case_insensitive(haystack: &str, needle: &str) -> Option<usize> {
    let needle_lower = needle.to_ascii_lowercase();
    let needle_bytes = needle_lower.as_bytes();
    haystack
        .as_bytes()
        .windows(needle_bytes.len())
        .position(|window| window.eq_ignore_ascii_case(needle_bytes))
}

/// Returns the first `limit` characters of `text` as a new string.
pub fn take_chars(text: &str, limit: usize) -> String {
    text.chars().take(limit).collect()
}

/// Counts the Unicode scalar values in `text`.
pub fn char_count(text: &str) -> usize {
    text.chars().count()
}
