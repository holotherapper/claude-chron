//! Embedding providers.

pub mod onnx;
pub mod registry;

pub use onnx::{
    EmbeddingError, EmbeddingProvider, EmbeddingResult, OnnxEmbeddingProvider, l2_normalize,
};
pub use registry::create_provider;
