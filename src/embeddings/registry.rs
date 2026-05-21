//! Embedding provider registry.

use super::onnx::{EmbeddingError, EmbeddingResult, OnnxEmbeddingProvider};

/// Creates an embedding provider based on the provider name.
///
/// Currently only `"onnx"` is supported. Other provider names return an error
/// with a message listing available providers.
///
/// # Errors
/// Returns [`EmbeddingError`] when the provider name is unknown or initialization fails.
pub fn create_provider(
    provider: &str,
    model: &str,
    dimension: usize,
) -> EmbeddingResult<OnnxEmbeddingProvider> {
    match provider {
        "onnx" => OnnxEmbeddingProvider::new(model, dimension),
        other => Err(EmbeddingError::Provider(format!(
            "unknown embedding provider: {other}. Available: onnx"
        ))),
    }
}
