//! Hybrid search over sqlite-vec and FTS5.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::rrf::rrf_merge;
use crate::storage::{Storage, StorageError, StoredChunk};

/// Ranked hybrid search result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchResult {
    /// Stored chunk data.
    pub chunk: StoredChunk,
    /// Reciprocal Rank Fusion score.
    pub score: f64,
    /// One-based dense search rank.
    pub dense_rank: Option<usize>,
    /// One-based sparse search rank.
    pub sparse_rank: Option<usize>,
    /// Dense sqlite-vec distance, lower is better.
    pub dense_distance: Option<f64>,
    /// Sparse BM25 score, lower is better.
    pub sparse_bm25: Option<f64>,
}

/// Runs dense vector search and sparse FTS search, then merges them with RRF.
///
/// # Errors
/// Returns [`StorageError`] when the underlying storage queries fail.
pub fn hybrid_search(
    storage: &Storage,
    query: &str,
    query_embedding: &[f32],
    limit: usize,
    project_path: Option<&str>,
) -> Result<Vec<SearchResult>, StorageError> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let candidate_limit = limit.saturating_mul(4).max(20);
    let dense = storage.dense_candidates(query_embedding, candidate_limit, project_path)?;
    let sparse = storage.sparse_candidates(query, candidate_limit, project_path)?;

    let dense_ids = dense
        .iter()
        .map(|candidate| candidate.chunk_id)
        .collect::<Vec<_>>();
    let sparse_ids = sparse
        .iter()
        .map(|candidate| candidate.chunk_id)
        .collect::<Vec<_>>();
    let fused = rrf_merge(&dense_ids, &sparse_ids, limit);
    let chunk_ids = fused.iter().map(|score| score.chunk_id).collect::<Vec<_>>();
    let mut chunks = storage.chunks_by_ids(&chunk_ids)?;
    let dense_scores: HashMap<i64, f64> = dense.iter().map(|c| (c.chunk_id, c.distance)).collect();
    let sparse_scores: HashMap<i64, f64> = sparse.iter().map(|c| (c.chunk_id, c.bm25)).collect();

    let results = fused
        .into_iter()
        .filter_map(|score| {
            chunks.remove(&score.chunk_id).map(|chunk| SearchResult {
                chunk,
                score: score.score,
                dense_rank: score.dense_rank,
                sparse_rank: score.sparse_rank,
                dense_distance: dense_scores.get(&score.chunk_id).copied(),
                sparse_bm25: sparse_scores.get(&score.chunk_id).copied(),
            })
        })
        .collect();
    Ok(results)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use chrono::{TimeZone, Utc};

    use super::*;
    use crate::storage::{NewSession, init_db};
    use crate::transcript::Chunk;

    fn temp_db_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("claude_chron_{name}_{}.db", std::process::id()))
    }

    fn sample_session() -> NewSession {
        NewSession {
            session_id: "session-1".to_string(),
            project_path: "/tmp/project".to_string(),
            transcript_path: PathBuf::from("/tmp/project/session-1.jsonl"),
            sha256: "abc123".to_string(),
            mtime: Utc
                .with_ymd_and_hms(2026, 5, 13, 1, 2, 3)
                .single()
                .expect("valid test datetime"),
            size: 42,
        }
    }

    fn chunk(uid: &str, text: &str) -> Chunk {
        Chunk {
            chunk_uid: uid.to_string(),
            session_id: "session-1".to_string(),
            project_path: "/tmp/project".to_string(),
            title: Some(text.to_string()),
            text: text.to_string(),
            text_for_embedding: text.to_string(),
            line_start: Some(1),
            line_end: Some(2),
            token_count: 2,
            created_at: None,
            embedding_model: "mock".to_string(),
            content_hash: uid.to_string(),
        }
    }

    #[test]
    fn hybrid_search_should_return_matching_chunks() {
        let path = temp_db_path("hybrid");
        let _ = fs::remove_file(&path);
        let mut storage = init_db(Path::new(&path), 1000, true, 3).expect("db should init");
        let session = sample_session();
        storage
            .index_session_atomic(
                &session,
                "sig-test",
                &[],
                &[
                    chunk("chunk-1", "cargo test"),
                    chunk("chunk-2", "release notes"),
                ],
                &[vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
            )
            .expect("session should index");

        let results = hybrid_search(&storage, "cargo", &[1.0, 0.0, 0.0], 5, None)
            .expect("hybrid search should work");

        assert_eq!(results[0].chunk.chunk_uid, "chunk-1");
        fs::remove_file(path).expect("temp db should be removed");
    }
}
