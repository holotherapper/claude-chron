//! SQLite storage for indexed Claude Code sessions.

pub mod db;

pub use db::{
    DenseCandidate, MigrateStats, NewSession, SparseCandidate, Storage, StorageError,
    StorageResult, StorageStats, StoredChunk, StoredSession, init_db,
};
