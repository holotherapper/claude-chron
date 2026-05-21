//! Indexing pipeline from JSONL transcripts into SQLite.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::config::Config;
use crate::embeddings::{EmbeddingError, EmbeddingProvider};
use crate::storage::{NewSession, Storage, StorageError};
use crate::transcript::{
    ChunkError, ParserError, REDACTION_VERSION, ScannerError, SessionFile, chunk_turns,
    parse_transcript, scan_sessions,
};

/// Signature of the indexing pipeline. A session is only considered up to date
/// when this matches, so changing the embedding model, dimension, chunk size,
/// or redaction rules forces affected sessions to be re-indexed rather than
/// skipped.
fn index_signature(config: &Config) -> String {
    format!(
        "{}|{}|{}|r{}",
        config.embedding.model,
        config.embedding.dimension,
        config.chunking.max_chars,
        REDACTION_VERSION,
    )
}

/// A session that failed to index, retained for diagnostics.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FailedSession {
    /// Session identifier.
    pub session_id: String,
    /// Error description.
    pub error: String,
}

/// Indexing summary.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexStats {
    /// Number of sessions indexed.
    pub indexed: usize,
    /// Number of sessions skipped because they are already indexed.
    pub skipped: usize,
    /// Number of sessions that failed to index.
    pub failed: usize,
    /// Sessions that failed, paired with their error message, for diagnostics.
    pub failures: Vec<FailedSession>,
}

/// Errors emitted by indexing operations.
#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    /// Scanner failed.
    #[error(transparent)]
    Scanner(#[from] ScannerError),
    /// Transcript parsing failed.
    #[error(transparent)]
    Parser(#[from] ParserError),
    /// Chunking failed.
    #[error(transparent)]
    Chunk(#[from] ChunkError),
    /// Embedding provider failed.
    #[error(transparent)]
    Embedding(#[from] EmbeddingError),
    /// Storage failed.
    #[error(transparent)]
    Storage(#[from] StorageError),
    /// Filesystem operation failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Indexes all discovered sessions under the configured scanner root.
///
/// # Errors
/// Returns [`IndexError`] when scanning fails. Per-session failures are counted
/// in [`IndexStats::failed`] and do not abort the whole run.
pub fn index_all(
    storage: &mut Storage,
    provider: &impl EmbeddingProvider,
    config: &Config,
    force: bool,
) -> Result<IndexStats, IndexError> {
    let sessions = scan_sessions(&config.scanner.root)?;
    let total = sessions.len();
    let mut stats = IndexStats::default();

    for (i, session) in sessions.iter().enumerate() {
        eprint!(
            "\r[{}/{}] {}...",
            i + 1,
            total,
            &session.session_id[..session.session_id.len().min(12)]
        );
        match index_session(storage, provider, session, config, force) {
            Ok(IndexOutcome::Indexed) => stats.indexed += 1,
            Ok(IndexOutcome::Skipped) => stats.skipped += 1,
            Err(e) => {
                stats.failed += 1;
                stats.failures.push(FailedSession {
                    session_id: session.session_id.clone(),
                    error: e.to_string(),
                });
            }
        }
    }
    eprintln!("\r[{total}/{total}] done.                    ");

    Ok(stats)
}

/// Indexes a single discovered session.
///
/// # Errors
/// Returns [`IndexError`] when parsing, embedding, or storage writes fail.
pub fn index_session(
    storage: &mut Storage,
    provider: &impl EmbeddingProvider,
    session: &SessionFile,
    config: &Config,
    force: bool,
) -> Result<IndexOutcome, IndexError> {
    let sha256 = sha256_file(&session.transcript_path)?;
    let signature = index_signature(config);
    if !force && storage.session_exists(&session.session_id, &sha256, session.mtime, &signature)? {
        return Ok(IndexOutcome::Skipped);
    }

    let turns = parse_transcript(&session.transcript_path)?;
    let chunks = chunk_turns(
        &turns,
        &session.session_id,
        Path::new(&session.project_path),
        &config.embedding.model,
        config.chunking.max_chars,
    )?;
    let text_refs: Vec<&str> = chunks
        .iter()
        .map(|chunk| chunk.text_for_embedding.as_str())
        .collect();
    let embeddings = embed_in_batches(provider, &text_refs, 16)?;

    let new_session = NewSession {
        session_id: session.session_id.clone(),
        project_path: session.project_path.clone(),
        transcript_path: session.transcript_path.clone(),
        sha256,
        mtime: session.mtime,
        size: session.size,
    };
    storage.index_session_atomic(&new_session, &signature, &turns, &chunks, &embeddings)?;
    Ok(IndexOutcome::Indexed)
}

/// Result of attempting to index one session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexOutcome {
    /// Session was indexed.
    Indexed,
    /// Session was already indexed and was skipped.
    Skipped,
}

fn embed_in_batches(
    provider: &impl EmbeddingProvider,
    texts: &[&str],
    batch_size: usize,
) -> Result<Vec<Vec<f32>>, EmbeddingError> {
    let mut all_embeddings = Vec::with_capacity(texts.len());
    for batch in texts.chunks(batch_size) {
        let batch_embeddings = provider.embed_texts(batch)?;
        all_embeddings.extend(batch_embeddings);
    }
    Ok(all_embeddings)
}

fn sha256_file(path: &Path) -> Result<String, std::io::Error> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 16 * 1024];
    loop {
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }
    Ok(hex::encode(hasher.finalize()))
}
