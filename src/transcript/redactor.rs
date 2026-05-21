//! Secret redaction and path normalization.

use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use regex::{Captures, Regex};

/// Version of the redaction ruleset. Bump this whenever the patterns change so
/// `cchron index` re-indexes sessions that were stored under the old rules
/// instead of skipping them as up to date.
pub const REDACTION_VERSION: u32 = 1;

/// Errors that can occur while redacting text.
#[derive(Debug, thiserror::Error)]
pub enum RedactionError {
    /// One of the static redaction patterns failed to compile.
    #[error("failed to compile redaction regex: {0}")]
    Regex(String),
}

struct RedactionRegexes {
    token_patterns: Vec<Regex>,
    env_assignment: Regex,
    secret_assignment: Regex,
}

static HOME_DIR: LazyLock<Option<PathBuf>> = LazyLock::new(dirs::home_dir);

static REGEXES: LazyLock<Result<RedactionRegexes, regex::Error>> = LazyLock::new(|| {
    Ok(RedactionRegexes {
        token_patterns: vec![
            // PEM private key blocks (multi-line; spans BEGIN..END).
            Regex::new(
                r"(?s)-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
            )?,
            // sk-... covers OpenAI and Anthropic (sk-ant-...) keys.
            Regex::new(r"\bsk-[A-Za-z0-9_-]{20,}\b")?,
            Regex::new(r"\bgh[pousr]_[A-Za-z0-9_]{30,}\b")?,
            Regex::new(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")?,
            Regex::new(r"\bglpat-[A-Za-z0-9_-]{20,}\b")?,
            Regex::new(r"\bhf_[A-Za-z0-9]{20,}\b")?,
            Regex::new(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b")?,
            Regex::new(r"\bA(?:KIA|SIA)[0-9A-Z]{16}\b")?,
            Regex::new(r"\bAIza[0-9A-Za-z_-]{35}\b")?,
            Regex::new(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")?,
            // Authorization: Bearer <opaque token> (also redacts the scheme).
            Regex::new(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}")?,
        ],
        env_assignment: Regex::new(
            r#"(?P<prefix>\b(?:export\s+)?[A-Z][A-Z0-9_]{1,}=)(?P<value>"[^"]*"|'[^']*'|[^\s'"`;$]+)"#,
        )?,
        // Optional snake/kebab prefixes (e.g. `client_`, `aws_secret_`) may
        // precede the sensitive core, so compound keys like `client_secret`,
        // `aws_secret_access_key`, and `private_key` are covered.
        secret_assignment: Regex::new(
            r#"(?i)(?P<prefix>\b(?:[a-z0-9]+[_-])*(?:secret[_-]?access[_-]?key|access[_-]?key[_-]?id|access[_-]?key|secret[_-]?key|private[_-]?key|client[_-]?secret|api[_-]?key|access[_-]?token|auth[_-]?token|refresh[_-]?token|token|secret|password|passwd|pwd)\s*[:=]\s*)(?P<value>"[^"]*"|'[^']*'|[^\s'"`]+)"#,
        )?,
    })
});

/// Redacts known secret tokens and secret-looking assignments.
///
/// # Errors
/// Returns [`RedactionError`] if the static regex set cannot be compiled.
pub fn redact_text(text: &str, home: Option<&Path>) -> Result<String, RedactionError> {
    let regexes = regexes()?;
    let mut redacted = normalize_home_paths(text, home);
    redacted = regexes
        .env_assignment
        .replace_all(&redacted, redact_assignment)
        .into_owned();
    redacted = regexes
        .secret_assignment
        .replace_all(&redacted, redact_assignment)
        .into_owned();
    for pattern in &regexes.token_patterns {
        redacted = pattern.replace_all(&redacted, "<redacted>").into_owned();
    }
    Ok(redacted)
}

/// Redacts `text`, returning a `<redacted>` placeholder if the static regex
/// set fails to compile. Fails closed: a redaction failure never yields raw,
/// possibly secret text.
pub fn redact_or_placeholder(text: &str) -> String {
    redact_text(text, None).unwrap_or_else(|_| "<redacted>".to_string())
}

/// Replaces the user's home directory prefix with `~`.
pub fn normalize_home_paths(text: &str, home: Option<&Path>) -> String {
    let home_path = home
        .map(Path::to_path_buf)
        .or_else(|| HOME_DIR.clone())
        .unwrap_or_default();
    let home_text = home_path.to_string_lossy();
    if home_text.is_empty() || home_text == "/" {
        return text.to_string();
    }
    text.replace(home_text.as_ref(), "~")
}

fn regexes() -> Result<&'static RedactionRegexes, RedactionError> {
    match &*REGEXES {
        Ok(regexes) => Ok(regexes),
        Err(err) => Err(RedactionError::Regex(err.to_string())),
    }
}

fn redact_assignment(captures: &Captures<'_>) -> String {
    let prefix = captures
        .name("prefix")
        .map_or("", |capture| capture.as_str());
    let quote = captures
        .name("value")
        .and_then(|capture| capture.as_str().chars().next())
        .filter(|ch| *ch == '"' || *ch == '\'')
        .map_or("", |ch| if ch == '"' { "\"" } else { "'" });
    format!("{prefix}{quote}<redacted>{quote}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redact_text_should_replace_api_tokens() {
        let text = "openai sk-abcdefghijklmnopqrstuvwxyz123456";

        let redacted = redact_text(text, None).expect("redaction should work");

        assert_eq!(redacted, "openai <redacted>");
    }

    #[test]
    fn redact_text_should_replace_secret_assignments() {
        let text = "export API_KEY=\"abc123\" token: ghp_abcdefghijklmnopqrstuvwxyzABCDE";

        let redacted = redact_text(text, None).expect("redaction should work");

        assert_eq!(redacted, "export API_KEY=\"<redacted>\" token: <redacted>");
    }

    #[test]
    fn normalize_home_paths_should_replace_home_prefix() {
        let home = Path::new("/Users/alex");

        let normalized = normalize_home_paths("/Users/alex/project/file", Some(home));

        assert_eq!(normalized, "~/project/file");
    }

    #[test]
    fn redact_text_should_cover_compound_secret_keys() {
        let cases = [
            "client_secret=abc123",
            "aws_secret_access_key = wJalrXUtnFEMIabcdEFGHIJKLMNOP",
            "AWS_SECRET_ACCESS_KEY: \"deadbeefdeadbeefdeadbeef\"",
            "private_key=verylongprivatekeyvalue",
            "refresh_token: rt_abcdefghijklmnop",
        ];
        for case in cases {
            let redacted = redact_text(case, None).expect("redaction works");
            assert!(
                redacted.contains("<redacted>"),
                "expected redaction for {case:?}, got {redacted:?}"
            );
            assert!(
                !redacted.contains("abc123")
                    && !redacted.contains("wJalrXUtnFEMI")
                    && !redacted.contains("deadbeef")
                    && !redacted.contains("verylongprivatekeyvalue")
                    && !redacted.contains("rt_abcdefghijklmnop"),
                "value leaked for {case:?}: {redacted:?}"
            );
        }
    }

    #[test]
    fn redact_text_should_cover_bearer_and_pem_and_anthropic() {
        let anthropic = redact_text("key sk-ant-api03-AABBCCDDEEFFGGHHIIJJKKLL", None).unwrap();
        assert_eq!(anthropic, "key <redacted>");

        let bearer = redact_text("Authorization: Bearer aB3xZ9.kk-77_secrettoken", None).unwrap();
        assert!(
            bearer.contains("<redacted>") && !bearer.contains("secrettoken"),
            "bearer not redacted: {bearer:?}"
        );

        let pem = "before\n-----BEGIN OPENSSH PRIVATE KEY-----\nAAAAbbbbCCCC\nDDDDeeee\n-----END OPENSSH PRIVATE KEY-----\nafter";
        let red = redact_text(pem, None).unwrap();
        assert!(
            red.contains("before") && red.contains("after") && red.contains("<redacted>"),
            "pem block not redacted: {red:?}"
        );
        assert!(!red.contains("AAAAbbbbCCCC"), "pem body leaked: {red:?}");
    }
}
