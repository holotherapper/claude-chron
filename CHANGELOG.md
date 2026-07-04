# Changelog

## [Unreleased]

### Fixed
- Runaway CPU from hook indexing. Three compounding causes, all fixed:
  - Chunk-level incremental indexing: re-indexing a grown transcript now embeds only new chunks (diffed by `chunk_uid`) and removes stale ones, instead of re-embedding the entire transcript on every `Stop` hook
  - Per-session single-flight lock (`~/.claude-chron/locks/<session_id>.lock`, advisory flock): overlapping hook invocations exit immediately instead of indexing the same session concurrently
  - ONNX Runtime intra-op threads bounded to 4 by default (was: all cores); configurable via `embedding.intra_threads` (0 = all cores)
- `SessionEnd` hook no longer force-re-embeds the whole session; it indexes incrementally like `Stop`

### Changed
- MSRV raised to 1.89 (std file locking)

## [0.1.0] - 2026-05-21

Local RAG search over Claude Code session history. Verbatim (no LLM
summarization), nothing injected into sessions, secrets redacted before storage.

### Added
- Hybrid search: ONNX bge-m3 semantic embedding + FTS5 trigram keyword search fused with RRF
- MCP server with `search`, `expand`, `read_session`, and `list_sessions` tools
- Claude Code `Stop`/`SessionEnd` hooks that only index sessions silently — nothing is summarized or injected back into any session
- Verbatim recall: indexes cleaned transcript turns (no LLM summarization step); `read_session`/`expand` serve the stored, redacted turns so recall works even after the original transcript is pruned or moved
- Secret redaction before storage across all content (user/assistant text, tool-call summaries, chunk text and titles): API keys/tokens (incl. Anthropic `sk-ant-`), compound secret keys (`client_secret`, `aws_secret_access_key`, …), `Bearer` tokens, PEM private keys, JWT, env-var assignments, home-path normalization. Redaction is mandatory (not configurable)
- `redact-db` command: re-applies redaction to all stored turns and chunks (incl. titles and the FTS shadow) in place, idempotently
- Project filtering for scoped searches; `migrate` command for project path renames
- Incremental indexing keyed on session id, SHA-256, mtime, and an index signature (embedding model, dimension, chunk size, redaction version) so changing any of them re-indexes affected sessions instead of skipping them
- Atomic session indexing (single transaction, rollback on failure); the database directory and file are created owner-only (`0700`/`0600`); MCP server loads the embedding model once per process
- Configurable embedding model, database path, and chunking via TOML
