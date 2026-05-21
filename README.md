# claude-chron

[![CI](https://github.com/holotherapper/claude-chron/actions/workflows/ci.yml/badge.svg)](https://github.com/holotherapper/claude-chron/actions)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![Rust 1.85+](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)

Verbatim, local search over your Claude Code session history — served as an MCP server so Claude recalls past work on its own.

## Overview

There are tools that give Claude persistent memory — summarized context, injected prompts, automatic recall. claude-chron is not one of them. Memory features are better handled by the model itself (Claude already has built-in memory), and summaries lose the detail that matters most: the back-and-forth of an investigation, the dead ends, the exact reasoning that led to a decision.

What I wanted was simpler: full-text search over every session I've ever run, verbatim, so I can find that conversation from three weeks ago where I figured out the caching strategy, or the exact command sequence that fixed the deploy. claude-chron indexes every transcript into a local SQLite database and serves it as an MCP server. When you ask Claude about past work, it searches the index — nothing summarized, nothing injected, nothing sent anywhere.

### What it does

- **Indexes cleaned transcript turns** — your questions, the assistant's replies, tool-call summaries. Long output and fenced blocks are truncated; conversational content is kept as written.
- **Redacts secrets before storage** — API keys, tokens, credentials, and home paths are scrubbed from all stored fields. Redaction fails closed.
- **Searches with hybrid retrieval** — semantic embedding (bge-m3 via ONNX, local) + FTS5 trigram keywords, fused with Reciprocal Rank Fusion.
- **Runs entirely locally** — one Rust binary, one SQLite file, no Python/Node runtime, no external APIs.

### What it does not do

- Summarize or compress your history
- Inject context, hints, or memory into sessions
- Send data anywhere — all processing and storage is local

## Getting started

### Prerequisites

- macOS or Linux
- [Rust 1.85+](https://rustup.rs/) (for source install) or [Homebrew](https://brew.sh/) (macOS)
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code)

### Install

```sh
brew install holotherapper/tap/cchron          # macOS (Homebrew)
cargo install --git https://github.com/holotherapper/claude-chron  # from source
```

### Set up

```sh
cchron setup
```

This registers the MCP server with Claude Code and runs an initial index. The first run downloads the bge-m3 ONNX model (~558 MB, cached locally).

> [!TIP]
> To index automatically after every session, install the Claude Code plugin:
> ```
> /plugin marketplace add holotherapper/claude-chron
> /plugin install claude-chron
> ```
> Or run `cchron install-hooks` and paste the output into `~/.claude/settings.json`.

## Usage

You don't interact with claude-chron directly. Once set up, Claude Code reaches it through four MCP tools:

| Tool | What it does |
|------|-------------|
| `search` | Hybrid semantic + keyword search. Optionally scoped to a project. Returns ranked chunks with `session_id` and `line_start`. |
| `expand` | Returns surrounding turns for a search hit, given `session_id` and `line_start`. |
| `read_session` | Returns a full session transcript from the stored (redacted) copy. Works even after the original file is pruned. |
| `list_sessions` | Browse indexed sessions, optionally filtered by project. |

In practice you keep working normally. When past context would help, ask Claude — it will search and cite what it finds.

## How it works

```
scan → parse → redact → chunk → embed → store → search (dense + sparse → RRF)
```

1. Scans `~/.claude/projects/` for JSONL transcripts
2. Parses them into conversation turns
3. Redacts secrets and home paths from all content
4. Chunks turns into ~800-character units
5. Embeds each chunk with local bge-m3 ONNX (1024-dim)
6. Stores into SQLite — sqlite-vec for vectors, FTS5 trigram for keywords
7. Searches both indexes and fuses results with Reciprocal Rank Fusion

Indexing is incremental: keyed on session id, SHA-256, mtime, and an index signature (model, dimension, chunk size, redaction version). Changing any of those re-indexes the affected sessions. Each session is written in a single atomic transaction.

## CLI reference

The MCP tools are the primary interface. CLI subcommands are for setup and maintenance:

| Command | Purpose |
|---------|---------|
| `cchron setup` | Register MCP server and run initial index |
| `cchron index [--force]` | Index new sessions (`--force`: re-index all) |
| `cchron search <query>` | Run the same search the MCP tool uses |
| `cchron status` | Session count, chunk count, database size |
| `cchron redact-db` | Re-apply redaction to stored rows in place |
| `cchron migrate --from <old> --to <new>` | Update a renamed project path |
| `cchron config` | Print merged configuration |
| `cchron install-hooks` | Print hook JSON for manual installation |
| `cchron serve` | Run MCP server on stdio (called by Claude Code) |

## Configuration

Optional. Global config at `~/.claude-chron/config.toml`, project-level at `.claude-chron.toml` (project overrides global):

```toml
[embedding]
provider = "onnx"
model = "gpahal/bge-m3-onnx-int8"
dimension = 1024

[db]
path = "~/.claude-chron/memory.db"
busy_timeout_ms = 5000
wal = true

[scanner]
root = "~/.claude/projects"

[chunking]
max_chars = 800
```

Redaction is always on — there is no knob to disable it.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.
