//! Configuration loading for `claude-chron`.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use toml::Value;

const DEFAULT_CONFIG_PATH: &str = "~/.claude-chron/config.toml";
const PROJECT_CONFIG_NAME: &str = ".claude-chron.toml";

/// Errors that can occur while loading configuration.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// The process current directory could not be read.
    #[error("failed to read current directory")]
    CurrentDir(#[source] std::io::Error),
    /// A configuration file could not be read.
    #[error("failed to read config file at {path}")]
    Read {
        /// Path that failed.
        path: PathBuf,
        /// Source I/O error.
        source: std::io::Error,
    },
    /// A TOML document could not be parsed.
    #[error("failed to parse TOML config at {path}")]
    Parse {
        /// Path that failed.
        path: PathBuf,
        /// Source TOML parse error.
        source: toml::de::Error,
    },
    /// Merged configuration could not be converted into typed config.
    #[error("invalid config shape")]
    InvalidShape(#[source] toml::de::Error),
}

/// Embedding provider settings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Embedding provider identifier.
    pub provider: String,
    /// Embedding model identifier.
    pub model: String,
    /// Embedding vector dimension.
    pub dimension: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: "onnx".to_string(),
            model: "gpahal/bge-m3-onnx-int8".to_string(),
            dimension: 1024,
        }
    }
}

/// SQLite database settings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// SQLite database path.
    pub path: PathBuf,
    /// Busy timeout in milliseconds.
    pub busy_timeout_ms: u64,
    /// Whether to enable WAL mode.
    pub wal: bool,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from("~/.claude-chron/memory.db"),
            busy_timeout_ms: 5000,
            wal: true,
        }
    }
}

/// Transcript scanner settings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScannerConfig {
    /// Root containing Claude Code project transcript folders.
    pub root: PathBuf,
}

impl Default for ScannerConfig {
    fn default() -> Self {
        Self {
            root: PathBuf::from("~/.claude/projects"),
        }
    }
}

/// Conversation chunking settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkingConfig {
    /// Maximum chunk size in characters before splitting.
    pub max_chars: usize,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self { max_chars: 800 }
    }
}

/// Complete application configuration.
///
/// Redaction is intentionally not configurable: it is mandatory and always on
/// for every stored field, so there is no knob to disable it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Config {
    /// Embedding provider settings.
    pub embedding: EmbeddingConfig,
    /// SQLite database settings.
    pub db: DatabaseConfig,
    /// Transcript scanner settings.
    pub scanner: ScannerConfig,
    /// Chunking settings.
    pub chunking: ChunkingConfig,
}

#[derive(Debug, Default, Deserialize)]
struct PartialConfig {
    embedding: Option<PartialEmbeddingConfig>,
    db: Option<PartialDatabaseConfig>,
    scanner: Option<PartialScannerConfig>,
    chunking: Option<PartialChunkingConfig>,
}

#[derive(Debug, Deserialize)]
struct PartialEmbeddingConfig {
    provider: Option<String>,
    model: Option<String>,
    dimension: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct PartialDatabaseConfig {
    path: Option<PathBuf>,
    busy_timeout_ms: Option<u64>,
    wal: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct PartialScannerConfig {
    root: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct PartialChunkingConfig {
    max_chars: Option<usize>,
}

/// Loads global and project configuration, with project values taking precedence.
///
/// # Errors
/// Returns [`ConfigError`] when an existing config file cannot be read or parsed.
pub fn load_config(project_dir: Option<&Path>) -> Result<Config, ConfigError> {
    load_config_from_paths(
        &crate::util::expand_path(Path::new(DEFAULT_CONFIG_PATH)),
        project_config_path(project_dir)?,
    )
}

/// Loads configuration from explicit global and project paths.
///
/// # Errors
/// Returns [`ConfigError`] when an existing config file cannot be read or parsed.
pub fn load_config_from_paths(
    global_path: &Path,
    project_path: PathBuf,
) -> Result<Config, ConfigError> {
    let mut merged = Value::Table(toml::map::Map::new());
    merge_value(&mut merged, read_toml_value(global_path)?);
    merge_value(&mut merged, read_toml_value(&project_path)?);
    config_from_value(merged)
}

/// Builds typed configuration from a TOML value merged over defaults.
///
/// # Errors
/// Returns [`ConfigError`] when the TOML value has an invalid shape.
pub fn config_from_value(value: Value) -> Result<Config, ConfigError> {
    let partial: PartialConfig = value.try_into().map_err(ConfigError::InvalidShape)?;
    let mut config = Config::default();

    if let Some(embedding) = partial.embedding {
        if let Some(provider) = embedding.provider {
            config.embedding.provider = provider;
        }
        if let Some(model) = embedding.model {
            config.embedding.model = model;
        }
        if let Some(dimension) = embedding.dimension {
            config.embedding.dimension = dimension;
        }
    }

    if let Some(db) = partial.db {
        if let Some(path) = db.path {
            config.db.path = path;
        }
        if let Some(busy_timeout_ms) = db.busy_timeout_ms {
            config.db.busy_timeout_ms = busy_timeout_ms;
        }
        if let Some(wal) = db.wal {
            config.db.wal = wal;
        }
    }

    if let Some(scanner) = partial.scanner
        && let Some(root) = scanner.root
    {
        config.scanner.root = root;
    }

    if let Some(chunking) = partial.chunking
        && let Some(max_chars) = chunking.max_chars
    {
        config.chunking.max_chars = max_chars;
    }

    expand_config_paths(&mut config);
    Ok(config)
}

fn project_config_path(project_dir: Option<&Path>) -> Result<PathBuf, ConfigError> {
    let base = match project_dir {
        Some(path) => path.to_path_buf(),
        None => std::env::current_dir().map_err(ConfigError::CurrentDir)?,
    };
    Ok(base.join(PROJECT_CONFIG_NAME))
}

fn read_toml_value(path: &Path) -> Result<Value, ConfigError> {
    let path = crate::util::expand_path(path);
    if !path.exists() {
        return Ok(Value::Table(toml::map::Map::new()));
    }
    let text = fs::read_to_string(&path).map_err(|source| ConfigError::Read {
        path: path.clone(),
        source,
    })?;
    toml::from_str::<Value>(&text).map_err(|source| ConfigError::Parse { path, source })
}

fn merge_value(left: &mut Value, right: Value) {
    match (left, right) {
        (Value::Table(left_table), Value::Table(right_table)) => {
            for (key, right_value) in right_table {
                match left_table.get_mut(&key) {
                    Some(left_value) => merge_value(left_value, right_value),
                    None => {
                        left_table.insert(key, right_value);
                    }
                }
            }
        }
        (left_value, right_value) => *left_value = right_value,
    }
}

fn expand_config_paths(config: &mut Config) {
    config.db.path = crate::util::expand_path(&config.db.path);
    config.scanner.root = crate::util::expand_path(&config.scanner.root);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_from_value_should_overlay_partial_sections() {
        let value = r#"
            [embedding]
            provider = "onnx"
            dimension = 1536

            [chunking]
            max_chars = 1200
        "#
        .to_string();
        let value = toml::from_str::<Value>(&value).expect("test TOML should parse");

        let config = config_from_value(value).expect("config should load");

        assert_eq!(config.embedding.provider, "onnx");
        assert_eq!(config.embedding.dimension, 1536);
        assert_eq!(config.chunking.max_chars, 1200);
        assert_eq!(config.db.busy_timeout_ms, 5000);
    }

    #[test]
    fn load_config_from_paths_should_merge_project_over_global() {
        let temp =
            std::env::temp_dir().join(format!("claude_chron_config_test_{}", std::process::id()));
        fs::create_dir_all(&temp).expect("temp dir should be created");
        let global_path = temp.join("global.toml");
        let project_path = temp.join("project.toml");
        fs::write(&global_path, "[scanner]\nroot = \"/global\"\n")
            .expect("global config should write");
        fs::write(&project_path, "[scanner]\nroot = \"/project\"\n")
            .expect("project config should write");

        let config =
            load_config_from_paths(&global_path, project_path).expect("config should load");

        assert_eq!(config.scanner.root, PathBuf::from("/project"));
        fs::remove_dir_all(temp).expect("temp dir should be removed");
    }
}
