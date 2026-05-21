//! End-to-end pipeline integration tests.
//!
//! These exercise the public API the way the CLI and MCP server do —
//! parse -> chunk -> redact -> store -> search — without ever constructing
//! an embedding provider. They behave identically with or without the
//! `onnx` feature and never download a model, so they are safe in CI.

use std::path::Path;

use chrono::{TimeZone, Utc};
use tempfile::tempdir;

use claude_chron::search::hybrid_search;
use claude_chron::storage::{NewSession, Storage, init_db};
use claude_chron::transcript::{chunk_turns, parse_jsonl_lines, parse_transcript};

/// Embedding dimension used for the synthetic vectors. Small and fixed so
/// the tests never touch a real model.
const DIM: usize = 4;

fn new_session(id: &str, project: &str, transcript: &Path) -> NewSession {
    NewSession {
        session_id: id.to_string(),
        project_path: project.to_string(),
        transcript_path: transcript.to_path_buf(),
        sha256: format!("sha-{id}"),
        mtime: Utc
            .with_ymd_and_hms(2026, 5, 19, 1, 2, 3)
            .single()
            .expect("valid datetime"),
        size: 1,
    }
}

/// Indexes a synthetic session sharing the keyword `kwfindme` into `project`.
fn index_synthetic(storage: &mut Storage, id: &str, project: &str) {
    let jsonl = format!(
        "{}\n{}\n",
        r#"{"type":"user","message":{"role":"user","content":"shared kwfindme topic"}}"#,
        r#"{"type":"assistant","message":{"role":"assistant","content":"shared kwfindme answer"}}"#,
    );
    let turns = parse_jsonl_lines(&jsonl).expect("synthetic transcript should parse");
    let chunks =
        chunk_turns(&turns, id, Path::new(project), "mock", 800).expect("chunking should succeed");
    let embeddings: Vec<Vec<f32>> = chunks.iter().map(|_| vec![0.5_f32; DIM]).collect();
    storage
        .index_session_atomic(
            &new_session(id, project, Path::new("/tmp/x.jsonl")),
            "sig",
            &[],
            &chunks,
            &embeddings,
        )
        .expect("session should index");
}

#[test]
fn pipeline_parses_redacts_chunks_stores_and_searches() {
    // A synthetic transcript whose user turn leaks an Anthropic-style key
    // and whose assistant turn carries a distinctive searchable token.
    let jsonl = concat!(
        r#"{"type":"user","timestamp":"2026-05-19T01:00:00Z","message":{"role":"user","content":"my key is sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA please help"}}"#,
        "\n",
        r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Apply the zzqplasma performance optimization to the loop."},{"type":"tool_use","name":"Bash","input":{"command":"cargo build --release"}}]}}"#,
        "\n",
    );

    let turns = parse_jsonl_lines(jsonl).expect("transcript should parse");
    assert_eq!(turns.len(), 1, "user + assistant grouped into one turn");
    assert!(
        turns[0].user.contains("sk-ant-"),
        "the parser keeps raw text; redaction happens downstream"
    );
    assert!(turns[0].assistant.contains("zzqplasma"));
    assert_eq!(
        turns[0].tool_summary,
        vec!["Bash: cargo build --release".to_string()]
    );

    let project = "/tmp/claude-chron-it/proj-a";
    let chunks = chunk_turns(&turns, "session-it-1", Path::new(project), "mock", 800)
        .expect("chunking should succeed");
    assert!(!chunks.is_empty());
    assert!(
        chunks.iter().all(|c| !c.text.contains("sk-ant-")),
        "chunk text is redacted before storage"
    );

    let dir = tempdir().expect("temp dir");
    let db_path = dir.path().join("memory.db");
    let mut storage = init_db(&db_path, 1000, true, DIM).expect("db should init");

    let session = new_session("session-it-1", project, Path::new("/tmp/x.jsonl"));
    let embeddings: Vec<Vec<f32>> = chunks.iter().map(|_| vec![0.25_f32; DIM]).collect();
    storage
        .index_session_atomic(&session, "sig-it", &turns, &chunks, &embeddings)
        .expect("session should index");

    // Redaction-at-rest: stored turns must not carry the raw key.
    let stored = storage.get_turns(&session.session_id).expect("get_turns");
    assert_eq!(stored.len(), 1);
    assert!(
        !stored[0].user.contains("sk-ant-"),
        "secret scrubbed at rest"
    );
    assert!(
        stored[0].user.contains("<redacted>"),
        "secret replaced with the placeholder"
    );

    // Hybrid search resolves the chunk via the FTS path (no model needed).
    let query_embedding = vec![0.25_f32; DIM];
    let results = hybrid_search(&storage, "zzqplasma", &query_embedding, 5, None)
        .expect("hybrid search should work");
    assert!(
        results.iter().any(|r| r.chunk.session_id == "session-it-1"),
        "the indexed chunk is findable by keyword"
    );

    let stats = storage.stats().expect("stats");
    assert_eq!(stats.session_count, 1);
    assert_eq!(
        usize::try_from(stats.chunk_count).expect("non-negative count"),
        chunks.len()
    );
}

#[test]
fn verbatim_recall_survives_transcript_deletion() {
    let dir = tempdir().expect("temp dir");
    let transcript = dir.path().join("session.jsonl");
    std::fs::write(
        &transcript,
        concat!(
            r#"{"type":"user","message":{"role":"user","content":"remember the qqxylo decision"}}"#,
            "\n",
            r#"{"type":"assistant","message":{"role":"assistant","content":"we chose qqxylo for caching"}}"#,
            "\n",
        ),
    )
    .expect("write transcript");

    let turns = parse_transcript(&transcript).expect("parse from file");
    assert_eq!(turns.len(), 1);

    let db_path = dir.path().join("memory.db");
    let mut storage = init_db(&db_path, 1000, true, DIM).expect("db init");
    let session = new_session("recall-1", "/tmp/claude-chron-it/proj-r", &transcript);
    storage
        .index_session_atomic(&session, "sig", &turns, &[], &[])
        .expect("session should index");

    // The original transcript disappears (pruned or moved): recall must
    // still work because turns are served from the database.
    std::fs::remove_file(&transcript).expect("remove transcript");
    assert!(!transcript.exists());

    let recalled = storage.get_turns(&session.session_id).expect("get_turns");
    assert_eq!(recalled.len(), 1);
    assert!(recalled[0].assistant.contains("qqxylo"));
}

#[test]
fn project_scoped_search_isolates_and_migrates() {
    let dir = tempdir().expect("temp dir");
    let db_path = dir.path().join("memory.db");
    let mut storage = init_db(&db_path, 1000, true, DIM).expect("db init");

    let proj_a = "/tmp/claude-chron-it/A";
    let proj_b = "/tmp/claude-chron-it/B";
    index_synthetic(&mut storage, "s-a", proj_a);
    index_synthetic(&mut storage, "s-b", proj_b);

    let query_embedding = vec![0.5_f32; DIM];
    let scoped = hybrid_search(&storage, "kwfindme", &query_embedding, 10, Some(proj_a))
        .expect("scoped search");
    assert!(!scoped.is_empty());
    assert!(
        scoped.iter().all(|r| r.chunk.project_path == proj_a),
        "the project filter excludes other projects"
    );

    let proj_c = "/tmp/claude-chron-it/C";
    let migrated = storage.migrate_project(proj_a, proj_c).expect("migrate");
    assert_eq!(migrated.sessions_updated, 1);
    let listed = storage.list_sessions(Some(proj_c), 10).expect("list");
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].session_id, "s-a");
}
