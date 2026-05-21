//! Transcript parsing, redaction, scanning, and chunking.

pub mod chunker;
pub mod parser;
pub mod redactor;
pub mod scanner;

pub use chunker::{Chunk, ChunkError, chunk_turns, clean_for_embedding};
pub use parser::{ParserError, Turn, parse_jsonl_lines, parse_transcript};
pub use redactor::{
    REDACTION_VERSION, RedactionError, normalize_home_paths, redact_or_placeholder, redact_text,
};
pub use scanner::{ScannerError, SessionFile, scan_sessions, session_file_from_path};
