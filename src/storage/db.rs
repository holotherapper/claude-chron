//! SQLite database access.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Once;
use std::time::Duration;

use chrono::{DateTime, Utc};
use rusqlite::ffi::sqlite3_auto_extension;
use rusqlite::{Connection, OptionalExtension, params, params_from_iter};
use serde::{Deserialize, Serialize};
use sqlite_vec::sqlite3_vec_init;

use crate::transcript::{Chunk, Turn};

static SQLITE_VEC_INIT: Once = Once::new();

/// Storage operation result type.
pub type StorageResult<T> = Result<T, StorageError>;

/// Errors emitted by the storage layer.
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    /// SQLite returned an error.
    #[error(transparent)]
    Sqlite(#[from] rusqlite::Error),
    /// Filesystem operation failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// JSON serialization failed.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// The number of chunks and embeddings differed.
    #[error("chunk count {chunks} does not match embedding count {embeddings}")]
    MismatchedEmbeddings {
        /// Number of chunks.
        chunks: usize,
        /// Number of embeddings.
        embeddings: usize,
    },
    /// An embedding vector had the wrong dimensionality.
    #[error("embedding dimension mismatch: expected {expected}, got {actual}")]
    InvalidEmbeddingDimension {
        /// Expected vector dimension.
        expected: usize,
        /// Actual vector dimension.
        actual: usize,
    },
    /// An integer could not be represented in SQLite's signed integer type.
    #[error("{field} value {value} is too large for SQLite INTEGER")]
    IntegerOverflow {
        /// Field name.
        field: &'static str,
        /// Field value.
        value: usize,
    },
}

/// Session data required when inserting a discovered transcript.
#[derive(Debug, Clone)]
pub struct NewSession {
    /// Session identifier.
    pub session_id: String,
    /// Decoded project path.
    pub project_path: String,
    /// JSONL transcript path.
    pub transcript_path: PathBuf,
    /// SHA-256 hash of the transcript file.
    pub sha256: String,
    /// Transcript modification time.
    pub mtime: DateTime<Utc>,
    /// Transcript size in bytes.
    pub size: u64,
}

/// Stored session metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredSession {
    /// Session identifier.
    pub session_id: String,
    /// Decoded project path.
    pub project_path: String,
    /// JSONL transcript path.
    pub transcript_path: PathBuf,
    /// SHA-256 hash of the transcript file.
    pub sha256: String,
    /// Transcript modification time.
    pub mtime: String,
    /// Transcript size in bytes.
    pub size: u64,
    /// Index insertion timestamp.
    pub indexed_at: String,
}

/// Stored searchable chunk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredChunk {
    /// SQLite row id.
    pub id: i64,
    /// Stable chunk identifier.
    pub chunk_uid: String,
    /// Source Claude Code session identifier.
    pub session_id: String,
    /// Decoded project path.
    pub project_path: String,
    /// Short title derived from the user request.
    pub title: Option<String>,
    /// Redacted full chunk text.
    pub text: String,
    /// Cleaned embedding text.
    pub text_for_embedding: String,
    /// One-based source start line.
    pub line_start: Option<i64>,
    /// One-based source end line.
    pub line_end: Option<i64>,
    /// Approximate token count.
    pub token_count: i64,
    /// Source creation timestamp.
    pub created_at: Option<String>,
    /// Embedding model used for this chunk.
    pub embedding_model: String,
    /// First 16 hex characters of the embedding text hash.
    pub content_hash: String,
}

/// Dense vector-search candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DenseCandidate {
    /// Chunk row id.
    pub chunk_id: i64,
    /// sqlite-vec distance, lower is better.
    pub distance: f64,
}

/// Sparse FTS candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparseCandidate {
    /// Chunk row id.
    pub chunk_id: i64,
    /// FTS5 BM25 score, lower is better.
    pub bm25: f64,
}

/// Result of a project path migration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MigrateStats {
    /// Number of sessions updated.
    pub sessions_updated: usize,
    /// Number of chunks updated.
    pub chunks_updated: usize,
}

/// Database statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorageStats {
    /// Number of indexed sessions.
    pub session_count: i64,
    /// Number of indexed chunks.
    pub chunk_count: i64,
    /// Database file size in bytes.
    pub db_size_bytes: u64,
}

/// Open SQLite storage handle.
pub struct Storage {
    conn: Connection,
    path: PathBuf,
    embedding_dimension: usize,
}

/// Opens and initializes a SQLite database.
///
/// # Errors
/// Returns [`StorageError`] when the database cannot be created or initialized.
pub fn init_db(
    path: &Path,
    busy_timeout_ms: u64,
    wal: bool,
    embedding_dimension: usize,
) -> StorageResult<Storage> {
    register_sqlite_vec();
    if let Some(parent) = path.parent() {
        // Tighten only directories this call actually creates. An existing
        // parent (a shared temp dir, a user-chosen location) is left as-is so
        // the database can also live outside a cchron-owned directory.
        let created_parent = !parent.as_os_str().is_empty() && !parent.exists();
        fs::create_dir_all(parent)?;
        if created_parent {
            set_private_dir_permissions(parent)?;
        }
    }

    let conn = Connection::open(path)?;
    set_private_permissions(path)?;
    conn.busy_timeout(Duration::from_millis(busy_timeout_ms))?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    if wal {
        conn.pragma_update(None, "journal_mode", "WAL")?;
    }
    conn.pragma_update(None, "synchronous", "NORMAL")?;

    let storage = Storage {
        conn,
        path: path.to_path_buf(),
        embedding_dimension,
    };
    storage.create_schema()?;
    Ok(storage)
}

impl Storage {
    /// Atomically replaces a session and all associated data in a single transaction.
    ///
    /// # Errors
    /// Returns [`StorageError`] when any insert fails. On error, the entire
    /// transaction is rolled back and existing data is preserved.
    pub fn index_session_atomic(
        &mut self,
        session: &NewSession,
        index_sig: &str,
        turns: &[Turn],
        chunks: &[Chunk],
        embeddings: &[Vec<f32>],
    ) -> StorageResult<()> {
        if chunks.len() != embeddings.len() {
            return Err(StorageError::MismatchedEmbeddings {
                chunks: chunks.len(),
                embeddings: embeddings.len(),
            });
        }
        let tx = self.conn.transaction()?;
        delete_session_rows(&tx, &session.session_id)?;
        insert_session_row(&tx, session, index_sig)?;
        insert_turn_rows(&tx, &session.session_id, turns)?;
        insert_chunk_rows(&tx, chunks, embeddings, self.embedding_dimension)?;
        tx.commit()?;
        Ok(())
    }

    /// Returns whether a session with the given transcript hash and mtime exists.
    ///
    /// # Errors
    /// Returns [`StorageError`] when the query fails.
    pub fn session_exists(
        &self,
        session_id: &str,
        sha256: &str,
        mtime: DateTime<Utc>,
        index_sig: &str,
    ) -> StorageResult<bool> {
        let exists = self
            .conn
            .query_row(
                "SELECT 1 FROM sessions
                 WHERE id = ?1 AND sha256 = ?2 AND mtime = ?3 AND index_sig = ?4 LIMIT 1",
                params![session_id, sha256, mtime.to_rfc3339(), index_sig],
                |_| Ok(()),
            )
            .optional()?
            .is_some();
        Ok(exists)
    }

    /// Loads a stored session by id.
    ///
    /// # Errors
    /// Returns [`StorageError`] when the query fails.
    pub fn get_session(&self, session_id: &str) -> StorageResult<Option<StoredSession>> {
        self.conn
            .query_row(
                "SELECT id, project_path, transcript_path, sha256, mtime, size, indexed_at
                 FROM sessions
                 WHERE id = ?1",
                params![session_id],
                |row| {
                    Ok(StoredSession {
                        session_id: row.get(0)?,
                        project_path: row.get(1)?,
                        transcript_path: PathBuf::from(row.get::<_, String>(2)?),
                        sha256: row.get(3)?,
                        mtime: row.get(4)?,
                        size: i64_to_u64(row.get(5)?),
                        indexed_at: row.get(6)?,
                    })
                },
            )
            .optional()
            .map_err(StorageError::from)
    }

    /// Returns the stored (already-redacted) turns for a session, in order.
    ///
    /// Reading from the database means recall does not depend on the original
    /// transcript file still existing, and the content is the redacted copy.
    ///
    /// # Errors
    /// Returns [`StorageError`] when the query fails.
    pub fn get_turns(&self, session_id: &str) -> StorageResult<Vec<Turn>> {
        let mut stmt = self.conn.prepare(
            "SELECT user, assistant, tool_summary, line_start, line_end, ts
             FROM turns WHERE session_id = ?1 ORDER BY turn_index",
        )?;
        let rows = stmt.query_map(params![session_id], |row| {
            let tool_json: String = row.get(2)?;
            Ok(Turn {
                user: row.get(0)?,
                assistant: row.get(1)?,
                tool_summary: serde_json::from_str(&tool_json).unwrap_or_default(),
                line_start: usize::try_from(row.get::<_, i64>(3)?).unwrap_or(0),
                line_end: usize::try_from(row.get::<_, i64>(4)?).unwrap_or(0),
                ts: row.get(5)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(StorageError::from)
    }

    /// Returns dense vector candidates ordered by ascending distance.
    ///
    /// # Errors
    /// Returns [`StorageError`] when dimensions do not match or the query fails.
    pub fn dense_candidates(
        &self,
        query_embedding: &[f32],
        limit: usize,
        project_path: Option<&str>,
    ) -> StorageResult<Vec<DenseCandidate>> {
        validate_embedding_dimension(query_embedding, self.embedding_dimension)?;
        if limit == 0 {
            return Ok(Vec::new());
        }

        let query_blob = f32s_to_le_bytes(query_embedding);
        let result_limit = usize_to_i64(limit, "limit")?;
        // sqlite-vec KNN is global (not partitioned by project), so a project
        // filter is applied after the fact. Over-fetch widely — up to the
        // project's own chunk count, capped — so a project whose matches sit
        // outside a small global window is not silently dropped.
        let knn_limit = if let Some(project_path) = project_path {
            let project_chunks: i64 = self.conn.query_row(
                "SELECT COUNT(*) FROM chunks WHERE project_path = ?1",
                params![project_path],
                |row| row.get(0),
            )?;
            let want = limit.saturating_mul(50).max(2000);
            project_chunks
                .min(usize_to_i64(want, "limit")?)
                .max(result_limit)
        } else {
            result_limit
        };

        let (project_filter, limit_placeholder) = match project_path {
            Some(_) => ("WHERE chunks.project_path = ?3", "?4"),
            None => ("", "?3"),
        };
        let sql = format!(
            "WITH matches AS (
                SELECT rowid, distance
                FROM chunk_vec
                WHERE embedding MATCH ?1 AND k = ?2
             )
             SELECT chunks.id, matches.distance
             FROM matches
             JOIN chunks ON chunks.id = matches.rowid
             {project_filter}
             ORDER BY matches.distance ASC
             LIMIT {limit_placeholder}"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let mut sql_params: Vec<Box<dyn rusqlite::types::ToSql>> =
            vec![Box::new(query_blob), Box::new(knn_limit)];
        if let Some(project_path) = project_path {
            sql_params.push(Box::new(project_path.to_string()));
        }
        sql_params.push(Box::new(result_limit));
        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            sql_params.iter().map(|p| p.as_ref()).collect();
        query_dense_rows(&mut stmt, param_refs.as_slice())
    }

    /// Returns sparse FTS candidates ordered by ascending BM25 score.
    ///
    /// # Errors
    /// Returns [`StorageError`] when the query fails.
    pub fn sparse_candidates(
        &self,
        query: &str,
        limit: usize,
        project_path: Option<&str>,
    ) -> StorageResult<Vec<SparseCandidate>> {
        if limit == 0 || query.trim().is_empty() {
            return Ok(Vec::new());
        }
        let fts_query = fts5_query(query);
        let result_limit = usize_to_i64(limit, "limit")?;

        let (project_filter, limit_placeholder) = match project_path {
            Some(_) => ("AND chunks.project_path = ?2", "?3"),
            None => ("", "?2"),
        };
        let sql = format!(
            "SELECT chunks.id, bm25(chunk_fts) AS score
             FROM chunk_fts
             JOIN chunks ON chunks.id = chunk_fts.rowid
             WHERE chunk_fts MATCH ?1 {project_filter}
             ORDER BY score ASC
             LIMIT {limit_placeholder}"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let mut sql_params: Vec<Box<dyn rusqlite::types::ToSql>> = vec![Box::new(fts_query)];
        if let Some(project_path) = project_path {
            sql_params.push(Box::new(project_path.to_string()));
        }
        sql_params.push(Box::new(result_limit));
        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            sql_params.iter().map(|p| p.as_ref()).collect();
        query_sparse_rows(&mut stmt, param_refs.as_slice())
    }

    /// Loads chunks by row id.
    ///
    /// # Errors
    /// Returns [`StorageError`] when the query fails.
    pub fn chunks_by_ids(&self, ids: &[i64]) -> StorageResult<HashMap<i64, StoredChunk>> {
        if ids.is_empty() {
            return Ok(HashMap::new());
        }
        let placeholders = (1..=ids.len())
            .map(|index| format!("?{index}"))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "SELECT id, chunk_uid, session_id, project_path, title, text,
                    text_for_embedding, line_start, line_end, token_count, created_at,
                    embedding_model, content_hash
             FROM chunks
             WHERE id IN ({placeholders})"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(ids.iter()), stored_chunk_from_row)?;
        let mut chunks = HashMap::with_capacity(ids.len());
        for row in rows {
            let chunk = row?;
            chunks.insert(chunk.id, chunk);
        }
        Ok(chunks)
    }

    /// Lists sessions, optionally filtered by project path.
    ///
    /// # Errors
    /// Returns [`StorageError`] when the database cannot be queried.
    pub fn list_sessions(
        &self,
        project: Option<&str>,
        limit: usize,
    ) -> StorageResult<Vec<StoredSession>> {
        let limit = usize_to_i64(limit, "limit")?;
        let (sql, params_vec): (String, Vec<Box<dyn rusqlite::types::ToSql>>) = match project {
            Some(p) => (
                "SELECT id, project_path, transcript_path, \
                 mtime, size, indexed_at, sha256 \
                 FROM sessions WHERE project_path = ?1 ORDER BY indexed_at DESC LIMIT ?2"
                    .to_string(),
                vec![Box::new(p.to_string()), Box::new(limit)],
            ),
            None => (
                "SELECT id, project_path, transcript_path, \
                 mtime, size, indexed_at, sha256 \
                 FROM sessions ORDER BY indexed_at DESC LIMIT ?1"
                    .to_string(),
                vec![Box::new(limit)],
            ),
        };
        let mut stmt = self.conn.prepare(&sql)?;
        let params_refs: Vec<&dyn rusqlite::types::ToSql> =
            params_vec.iter().map(|p| p.as_ref()).collect();
        let rows = stmt.query_map(params_refs.as_slice(), |row| {
            let tp: String = row.get(2)?;
            let sz: i64 = row.get(4)?;
            Ok(StoredSession {
                session_id: row.get(0)?,
                project_path: row.get(1)?,
                transcript_path: PathBuf::from(tp),
                mtime: row.get(3)?,
                size: i64_to_u64(sz),
                indexed_at: row.get(5)?,
                sha256: row.get(6)?,
            })
        })?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(StorageError::from)
    }

    /// Migrates all sessions and chunks from one project path to another.
    ///
    /// # Errors
    /// Returns [`StorageError`] when the update fails.
    pub fn migrate_project(&mut self, from: &str, to: &str) -> StorageResult<MigrateStats> {
        let tx = self.conn.transaction()?;
        let sessions_updated = tx.execute(
            "UPDATE sessions SET project_path = ?1 WHERE project_path = ?2",
            params![to, from],
        )?;
        let chunks_updated = tx.execute(
            "UPDATE chunks SET project_path = ?1 WHERE project_path = ?2",
            params![to, from],
        )?;
        tx.commit()?;
        Ok(MigrateStats {
            sessions_updated,
            chunks_updated,
        })
    }

    /// Returns basic database statistics.
    ///
    /// # Errors
    /// Returns [`StorageError`] when the database cannot be queried or statted.
    pub fn stats(&self) -> StorageResult<StorageStats> {
        let (session_count, chunk_count) = self.conn.query_row(
            "SELECT (SELECT COUNT(*) FROM sessions), (SELECT COUNT(*) FROM chunks)",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        let db_size_bytes: u64 = fs::metadata(&self.path).map_or(0, |metadata| metadata.len());
        Ok(StorageStats {
            session_count,
            chunk_count,
            db_size_bytes,
        })
    }

    /// Re-applies redaction to every stored turn and chunk in place,
    /// returning `(turns_updated, chunks_updated)`. Chunk text, embedding
    /// text, and titles plus the FTS shadow row are re-redacted; the stored
    /// vector is intentionally left as-is, since a redaction-only text delta
    /// does not meaningfully change the embedding and re-embedding would
    /// require the model. Idempotent: already-redacted rows are left
    /// untouched.
    ///
    /// # Errors
    /// Returns [`StorageError`] when reading or writing fails.
    pub fn reredact_stored(&mut self) -> StorageResult<(usize, usize)> {
        use crate::transcript::redact_or_placeholder;

        let turn_rows: Vec<(i64, String, String, String)> = {
            let mut stmt = self
                .conn
                .prepare("SELECT id, user, assistant, tool_summary FROM turns")?;
            stmt.query_map([], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            })?
            .collect::<Result<Vec<_>, _>>()?
        };
        let chunk_rows: Vec<(i64, String, Option<String>, String, String)> = {
            let mut stmt = self
                .conn
                .prepare("SELECT id, chunk_uid, title, text, text_for_embedding FROM chunks")?;
            stmt.query_map([], |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?
        };
        let tx = self.conn.transaction()?;
        let mut turns_updated = 0usize;
        let mut chunks_updated = 0usize;
        {
            let mut turn_stmt = tx.prepare(
                "UPDATE turns SET user = ?1, assistant = ?2, tool_summary = ?3 WHERE id = ?4",
            )?;
            for (id, user, assistant, tool_summary) in turn_rows {
                let new_user = redact_or_placeholder(&user);
                let new_assistant = redact_or_placeholder(&assistant);
                let new_tool = redact_tool_summary_json(&tool_summary);
                if new_user != user || new_assistant != assistant || new_tool != tool_summary {
                    turn_stmt.execute(params![new_user, new_assistant, new_tool, id])?;
                    turns_updated += 1;
                }
            }

            let mut chunk_stmt = tx.prepare(
                "UPDATE chunks SET title = ?1, text = ?2, text_for_embedding = ?3 WHERE id = ?4",
            )?;
            let mut fts_del = tx.prepare("DELETE FROM chunk_fts WHERE rowid = ?1")?;
            let mut fts_ins =
                tx.prepare("INSERT INTO chunk_fts(rowid, chunk_uid, text) VALUES (?1, ?2, ?3)")?;
            for (id, chunk_uid, title, text, embed) in chunk_rows {
                let new_title = title.as_ref().map(|t| redact_or_placeholder(t));
                let new_text = redact_or_placeholder(&text);
                let new_embed = redact_or_placeholder(&embed);
                if new_title != title || new_text != text || new_embed != embed {
                    chunk_stmt.execute(params![new_title, new_text, new_embed, id])?;
                    fts_del.execute(params![id])?;
                    // Keep the FTS shadow row in sync with the re-redacted text.
                    fts_ins.execute(params![id, chunk_uid, new_embed])?;
                    chunks_updated += 1;
                }
            }
        }
        tx.commit()?;
        Ok((turns_updated, chunks_updated))
    }

    fn create_schema(&self) -> StorageResult<()> {
        self.conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                project_path TEXT NOT NULL,
                transcript_path TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                mtime TEXT NOT NULL,
                size INTEGER NOT NULL,
                indexed_at TEXT NOT NULL,
                index_sig TEXT,
                UNIQUE(sha256, mtime)
            );

            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                turn_index INTEGER NOT NULL,
                user TEXT NOT NULL,
                assistant TEXT NOT NULL,
                tool_summary TEXT NOT NULL,
                line_start INTEGER NOT NULL,
                line_end INTEGER NOT NULL,
                ts TEXT,
                UNIQUE(session_id, turn_index)
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_uid TEXT NOT NULL UNIQUE,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                project_path TEXT NOT NULL,
                title TEXT,
                text TEXT NOT NULL,
                text_for_embedding TEXT NOT NULL,
                line_start INTEGER,
                line_end INTEGER,
                token_count INTEGER NOT NULL,
                created_at TEXT,
                embedding_model TEXT NOT NULL,
                content_hash TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_hash_mtime
                ON sessions(sha256, mtime);
            CREATE INDEX IF NOT EXISTS idx_chunks_session
                ON chunks(session_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_project
                ON chunks(project_path);
            ",
        )?;

        // Migrate databases created before index signatures existed. Adding a
        // column in SQLite is metadata-only (no row rewrite), so this is safe
        // even on a large database. Rows keep NULL `index_sig`, which never
        // equals a real signature, so they are re-indexed on the next run.
        let has_index_sig = self
            .conn
            .query_row(
                "SELECT 1 FROM pragma_table_info('sessions') WHERE name = 'index_sig'",
                [],
                |_| Ok(()),
            )
            .optional()?
            .is_some();
        if !has_index_sig {
            self.conn
                .execute("ALTER TABLE sessions ADD COLUMN index_sig TEXT", [])?;
        }

        self.conn.execute(
            &format!(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vec
                 USING vec0(embedding float[{}] distance_metric=cosine)",
                self.embedding_dimension
            ),
            [],
        )?;
        self.conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts
             USING fts5(chunk_uid UNINDEXED, text, tokenize = 'trigram')",
            [],
        )?;
        Ok(())
    }
}

fn delete_session_rows(conn: &Connection, session_id: &str) -> StorageResult<()> {
    let chunk_ids: Vec<i64> = {
        let mut stmt = conn.prepare("SELECT id FROM chunks WHERE session_id = ?1")?;
        stmt.query_map(params![session_id], |row| row.get::<_, i64>(0))?
            .collect::<Result<Vec<_>, _>>()?
    };
    {
        let mut vec_stmt = conn.prepare("DELETE FROM chunk_vec WHERE rowid = ?1")?;
        let mut fts_stmt = conn.prepare("DELETE FROM chunk_fts WHERE rowid = ?1")?;
        for chunk_id in &chunk_ids {
            vec_stmt.execute(params![chunk_id])?;
            fts_stmt.execute(params![chunk_id])?;
        }
    }
    conn.execute(
        "DELETE FROM chunks WHERE session_id = ?1",
        params![session_id],
    )?;
    conn.execute(
        "DELETE FROM turns WHERE session_id = ?1",
        params![session_id],
    )?;
    conn.execute("DELETE FROM sessions WHERE id = ?1", params![session_id])?;
    Ok(())
}

fn insert_session_row(
    conn: &Connection,
    session: &NewSession,
    index_sig: &str,
) -> StorageResult<()> {
    conn.execute(
        "INSERT INTO sessions
            (id, project_path, transcript_path, sha256, mtime, size, indexed_at, index_sig)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            session.session_id,
            session.project_path,
            session.transcript_path.to_string_lossy(),
            session.sha256,
            session.mtime.to_rfc3339(),
            u64_to_i64(session.size),
            Utc::now().to_rfc3339(),
            index_sig,
        ],
    )?;
    Ok(())
}

fn insert_turn_rows(conn: &Connection, session_id: &str, turns: &[Turn]) -> StorageResult<()> {
    use crate::transcript::redact_or_placeholder;

    let mut stmt = conn.prepare(
        "INSERT INTO turns (session_id, turn_index, user, assistant, tool_summary, line_start, line_end, ts)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
    )?;
    for (index, turn) in turns.iter().enumerate() {
        let user = redact_or_placeholder(&turn.user);
        let assistant = redact_or_placeholder(&turn.assistant);
        let redacted_summaries: Vec<String> = turn
            .tool_summary
            .iter()
            .map(|s| redact_or_placeholder(s))
            .collect();
        let tool_summary = serde_json::to_string(&redacted_summaries)?;
        stmt.execute(params![
            session_id,
            usize_to_i64(index, "turn_index")?,
            user,
            assistant,
            tool_summary,
            usize_to_i64(turn.line_start, "line_start")?,
            usize_to_i64(turn.line_end, "line_end")?,
            turn.ts,
        ])?;
    }
    Ok(())
}

/// Re-redacts a stored `tool_summary` JSON array element-wise, preserving the
/// JSON structure. Falls back to the original string if it does not parse.
fn redact_tool_summary_json(json: &str) -> String {
    use crate::transcript::redact_or_placeholder;
    match serde_json::from_str::<Vec<String>>(json) {
        Ok(items) => {
            let redacted: Vec<String> = items
                .iter()
                .map(|item| redact_or_placeholder(item))
                .collect();
            serde_json::to_string(&redacted).unwrap_or_else(|_| json.to_string())
        }
        Err(_) => json.to_string(),
    }
}

fn insert_chunk_rows(
    conn: &Connection,
    chunks: &[Chunk],
    embeddings: &[Vec<f32>],
    embedding_dimension: usize,
) -> StorageResult<()> {
    let mut chunk_stmt = conn.prepare(
        "INSERT INTO chunks (chunk_uid, session_id, project_path, title, text,
            text_for_embedding, line_start, line_end, token_count, created_at,
            embedding_model, content_hash)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
    )?;
    let mut vec_stmt = conn.prepare("INSERT INTO chunk_vec(rowid, embedding) VALUES (?1, ?2)")?;
    let mut fts_stmt =
        conn.prepare("INSERT INTO chunk_fts(rowid, chunk_uid, text) VALUES (?1, ?2, ?3)")?;

    for (chunk, embedding) in chunks.iter().zip(embeddings) {
        validate_embedding_dimension(embedding, embedding_dimension)?;
        chunk_stmt.execute(params![
            chunk.chunk_uid,
            chunk.session_id,
            chunk.project_path,
            chunk.title,
            chunk.text,
            chunk.text_for_embedding,
            optional_usize_to_i64(chunk.line_start, "line_start")?,
            optional_usize_to_i64(chunk.line_end, "line_end")?,
            usize_to_i64(chunk.token_count, "token_count")?,
            chunk.created_at,
            chunk.embedding_model,
            chunk.content_hash,
        ])?;
        let rowid = conn.last_insert_rowid();
        let embedding_blob = f32s_to_le_bytes(embedding);
        vec_stmt.execute(params![rowid, embedding_blob])?;
        fts_stmt.execute(params![rowid, chunk.chunk_uid, chunk.text_for_embedding])?;
    }
    Ok(())
}

fn register_sqlite_vec() {
    SQLITE_VEC_INIT.call_once(|| unsafe {
        // SAFETY: sqlite-vec exposes a SQLite extension entrypoint with the C ABI
        // expected by sqlite3_auto_extension. Registering it once makes vec0
        // available to every subsequent rusqlite connection in this process.
        sqlite3_auto_extension(Some(std::mem::transmute::<
            *const (),
            unsafe extern "C" fn(
                *mut rusqlite::ffi::sqlite3,
                *mut *mut i8,
                *const rusqlite::ffi::sqlite3_api_routines,
            ) -> i32,
        >(sqlite3_vec_init as *const ())));
    });
}

fn query_dense_rows<P>(
    stmt: &mut rusqlite::Statement<'_>,
    params: P,
) -> StorageResult<Vec<DenseCandidate>>
where
    P: rusqlite::Params,
{
    let rows = stmt.query_map(params, |row| {
        Ok(DenseCandidate {
            chunk_id: row.get(0)?,
            distance: row.get(1)?,
        })
    })?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(StorageError::from)
}

fn query_sparse_rows<P>(
    stmt: &mut rusqlite::Statement<'_>,
    params: P,
) -> StorageResult<Vec<SparseCandidate>>
where
    P: rusqlite::Params,
{
    let rows = stmt.query_map(params, |row| {
        Ok(SparseCandidate {
            chunk_id: row.get(0)?,
            bm25: row.get(1)?,
        })
    })?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(StorageError::from)
}

fn stored_chunk_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<StoredChunk> {
    Ok(StoredChunk {
        id: row.get(0)?,
        chunk_uid: row.get(1)?,
        session_id: row.get(2)?,
        project_path: row.get(3)?,
        title: row.get(4)?,
        text: row.get(5)?,
        text_for_embedding: row.get(6)?,
        line_start: row.get(7)?,
        line_end: row.get(8)?,
        token_count: row.get(9)?,
        created_at: row.get(10)?,
        embedding_model: row.get(11)?,
        content_hash: row.get(12)?,
    })
}

fn validate_embedding_dimension(embedding: &[f32], expected: usize) -> StorageResult<()> {
    if embedding.len() == expected {
        return Ok(());
    }
    Err(StorageError::InvalidEmbeddingDimension {
        expected,
        actual: embedding.len(),
    })
}

fn f32s_to_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Builds an FTS5 query from free text. Each whitespace-separated term is
/// quoted independently and the terms are ANDed (FTS5 implicit AND), so the
/// trigram tokenizer still applies per term and multi-word queries match on
/// term overlap.
fn fts5_query(query: &str) -> String {
    query
        .split_whitespace()
        .map(|term| format!("\"{}\"", term.replace('"', "\"\"")))
        .collect::<Vec<_>>()
        .join(" ")
}

fn usize_to_i64(value: usize, field: &'static str) -> StorageResult<i64> {
    i64::try_from(value).map_err(|_| StorageError::IntegerOverflow { field, value })
}

fn optional_usize_to_i64(value: Option<usize>, field: &'static str) -> StorageResult<Option<i64>> {
    value.map(|value| usize_to_i64(value, field)).transpose()
}

fn u64_to_i64(value: u64) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

fn i64_to_u64(value: i64) -> u64 {
    u64::try_from(value).unwrap_or_default()
}

#[cfg(unix)]
fn set_private_permissions(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let mut permissions = fs::metadata(path)?.permissions();
    permissions.set_mode(0o600);
    fs::set_permissions(path, permissions)
}

#[cfg(not(unix))]
fn set_private_permissions(_path: &Path) -> std::io::Result<()> {
    Ok(())
}

/// Restricts the database directory to its owner. The SQLite WAL and SHM
/// sidecar files hold the same content as the database, so the directory —
/// not just the database file — must be owner-only.
#[cfg(unix)]
fn set_private_dir_permissions(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let mut permissions = fs::metadata(path)?.permissions();
    permissions.set_mode(0o700);
    fs::set_permissions(path, permissions)
}

#[cfg(not(unix))]
fn set_private_dir_permissions(_path: &Path) -> std::io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use chrono::TimeZone;

    use super::*;
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

    fn sample_turn() -> Turn {
        Turn {
            user: "How do I run tests?".to_string(),
            assistant: "Use cargo test.".to_string(),
            tool_summary: vec!["Bash: cargo test".to_string()],
            line_start: 1,
            line_end: 2,
            ts: None,
        }
    }

    fn sample_chunk(text: &str) -> Chunk {
        Chunk {
            chunk_uid: format!("session-1:1:1:1:{text}"),
            session_id: "session-1".to_string(),
            project_path: "/tmp/project".to_string(),
            title: Some("Run tests".to_string()),
            text: text.to_string(),
            text_for_embedding: text.to_string(),
            line_start: Some(1),
            line_end: Some(2),
            token_count: 2,
            created_at: None,
            embedding_model: "mock".to_string(),
            content_hash: "hash".to_string(),
        }
    }

    #[test]
    fn storage_should_insert_and_query_session_data() {
        let path = temp_db_path("storage_insert");
        let _ = fs::remove_file(&path);
        let mut storage = init_db(Path::new(&path), 1000, true, 3).expect("db should init");
        let session = sample_session();

        storage
            .index_session_atomic(
                &session,
                "sig-v1",
                &[sample_turn()],
                &[sample_chunk("cargo test")],
                &[vec![1.0, 0.0, 0.0]],
            )
            .expect("session should index");

        assert!(
            storage
                .session_exists(
                    &session.session_id,
                    &session.sha256,
                    session.mtime,
                    "sig-v1"
                )
                .expect("session lookup should work"),
            "fresh when id + signature match"
        );
        assert!(
            !storage
                .session_exists(
                    &session.session_id,
                    &session.sha256,
                    session.mtime,
                    "sig-v2"
                )
                .expect("session lookup should work"),
            "stale when signature differs (model/chunk/redaction changed)"
        );
        let stored = storage
            .get_session(&session.session_id)
            .expect("session query should work")
            .expect("session should exist");
        assert_eq!(stored.session_id, session.session_id);
        fs::remove_file(path).expect("temp db should be removed");
    }

    #[test]
    fn reredact_stored_should_scrub_raw_rows_and_be_idempotent() {
        let path = temp_db_path("reredact");
        let _ = fs::remove_file(&path);
        let mut storage = init_db(Path::new(&path), 1000, true, 3).expect("db should init");
        storage
            .index_session_atomic(&sample_session(), "sig-v1", &[], &[], &[])
            .expect("session insert");

        // Raw rows are built from the *runtime* home so the home -> ~
        // normalization in reredact_stored applies on any machine, CI included.
        let home = dirs::home_dir()
            .expect("home dir should resolve")
            .to_string_lossy()
            .into_owned();
        let raw_user = format!("see {home}/secret");
        let assistant_raw = "token sk-abcdefghijklmnopqrstuvwxyz012345";
        let raw_tool = format!("[\"Bash: ls {home}/x\"]");
        let raw_title = format!("fix {home}/.env");
        let raw_text = format!("raw {home}/a sk-abcdefghijklmnopqrstuvwxyz012345");
        let raw_embed = format!("raw {home}/a");

        // Insert raw rows directly, bypassing the redacting insert path, so
        // reredact_stored has unredacted data to scrub.
        storage
            .conn
            .execute(
                "INSERT INTO turns
                   (session_id, turn_index, user, assistant, tool_summary, line_start, line_end, ts)
                 VALUES ('session-1', 0, ?1, ?2, ?3, 1, 2, NULL)",
                params![raw_user, assistant_raw, raw_tool],
            )
            .expect("raw turn insert");
        storage
            .conn
            .execute(
                "INSERT INTO chunks
                   (chunk_uid, session_id, project_path, title, text,
                    text_for_embedding, token_count, embedding_model, content_hash)
                 VALUES ('c1','session-1','/p', ?1, ?2, ?3, 1,'m','h')",
                params![raw_title, raw_text, raw_embed],
            )
            .expect("raw chunk insert");
        storage
            .conn
            .execute(
                "INSERT INTO chunk_fts(rowid, chunk_uid, text) VALUES
                 ((SELECT id FROM chunks WHERE chunk_uid='c1'), 'c1', ?1)",
                params![raw_text],
            )
            .expect("raw fts insert");

        assert_eq!(storage.reredact_stored().expect("reredact"), (1, 1));
        let (user, assistant, tool): (String, String, String) = storage
            .conn
            .query_row(
                "SELECT user, assistant, tool_summary FROM turns WHERE session_id='session-1'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .expect("turn query");
        assert_eq!(user, "see ~/secret");
        assert_eq!(assistant, "token <redacted>");
        assert_eq!(tool, "[\"Bash: ls ~/x\"]");
        let (title, ctext): (String, String) = storage
            .conn
            .query_row(
                "SELECT title, text FROM chunks WHERE chunk_uid='c1'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .expect("chunk query");
        assert_eq!(title, "fix ~/.env");
        assert_eq!(ctext, "raw ~/a <redacted>");
        let fts_text: String = storage
            .conn
            .query_row(
                "SELECT text FROM chunk_fts WHERE chunk_uid='c1'",
                [],
                |row| row.get(0),
            )
            .expect("fts query");
        assert_eq!(
            fts_text, "raw ~/a",
            "FTS shadow holds the re-redacted text_for_embedding"
        );

        assert_eq!(
            storage.reredact_stored().expect("reredact again"),
            (0, 0),
            "idempotent: clean rows are not rewritten"
        );
        fs::remove_file(path).expect("temp db should be removed");
    }
}
