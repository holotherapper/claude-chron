//! ONNX embedding provider backed by bge-m3 (or compatible) models.

/// Embedding operation result type.
pub type EmbeddingResult<T> = Result<T, EmbeddingError>;

/// Errors emitted by embedding providers.
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    /// Embedding dimension must be greater than zero.
    #[error("embedding dimension must be greater than zero")]
    InvalidDimension,
    /// Provider setup or inference failed.
    #[error("embedding provider failed: {0}")]
    Provider(String),
}

/// Batch text embedding provider.
pub trait EmbeddingProvider {
    /// Embeds a batch of texts.
    ///
    /// # Errors
    /// Returns [`EmbeddingError`] when provider setup or inference fails.
    fn embed_texts(&self, texts: &[&str]) -> EmbeddingResult<Vec<Vec<f32>>>;
}

/// L2-normalizes a vector in place.
pub fn l2_normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm == 0.0 {
        return;
    }
    for v in vector {
        *v /= norm;
    }
}

// ---------------------------------------------------------------------------
// Stub implementation (no `onnx` feature)
// ---------------------------------------------------------------------------

#[cfg(not(feature = "onnx"))]
mod stub {
    use sha2::{Digest, Sha256};

    use super::{EmbeddingProvider, EmbeddingResult, l2_normalize};

    #[derive(Debug, Clone)]
    pub struct OnnxEmbeddingProvider {
        dimension: usize,
    }

    impl OnnxEmbeddingProvider {
        /// # Errors
        /// Returns error when `dimension` is zero.
        pub fn new(
            _model: impl Into<String>,
            dimension: usize,
            _intra_threads: usize,
        ) -> EmbeddingResult<Self> {
            if dimension == 0 {
                return Err(super::EmbeddingError::InvalidDimension);
            }
            Ok(Self { dimension })
        }
    }

    impl EmbeddingProvider for OnnxEmbeddingProvider {
        fn embed_texts(&self, texts: &[&str]) -> EmbeddingResult<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|text| deterministic_embedding(text, self.dimension))
                .collect())
        }
    }

    fn deterministic_embedding(text: &str, dimension: usize) -> Vec<f32> {
        let mut vector = vec![0.0; dimension];
        if text.trim().is_empty() {
            vector[0] = 1.0;
            return vector;
        }
        for (index, window) in text.as_bytes().windows(8).enumerate() {
            let mut hasher = Sha256::new();
            hasher.update(window);
            hasher.update(index.to_le_bytes());
            let digest = hasher.finalize();
            let slot = usize::from(digest[0]) % dimension;
            let sign = if digest[1] & 1 == 0 { 1.0 } else { -1.0 };
            vector[slot] += sign * (f32::from(digest[2]) / 255.0);
        }
        if vector.iter().all(|v| *v == 0.0) {
            let slot = text
                .as_bytes()
                .first()
                .map_or(0, |byte| usize::from(*byte) % dimension);
            vector[slot] = 1.0;
        }
        l2_normalize(&mut vector);
        vector
    }
}

// ---------------------------------------------------------------------------
// Real ONNX implementation (requires `onnx` feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "onnx")]
mod real {
    use std::cell::RefCell;
    use std::path::PathBuf;

    use super::{EmbeddingError, EmbeddingProvider, EmbeddingResult, l2_normalize};

    /// ONNX embedding provider using `ort` for inference.
    pub struct OnnxEmbeddingProvider {
        model_name: String,
        dimension: usize,
        session: RefCell<ort::session::Session>,
        tokenizer: tokenizers::Tokenizer,
        output_names: Vec<String>,
    }

    impl std::fmt::Debug for OnnxEmbeddingProvider {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("OnnxEmbeddingProvider")
                .field("model_name", &self.model_name)
                .field("dimension", &self.dimension)
                .finish()
        }
    }

    impl OnnxEmbeddingProvider {
        /// Downloads the model from HuggingFace (if not cached) and initialises
        /// the ONNX Runtime session. `intra_threads` bounds ONNX Runtime's
        /// intra-op thread pool (0 = all cores).
        ///
        /// # Errors
        /// Returns [`EmbeddingError`] on dimension, download, or session errors.
        pub fn new(
            model: impl Into<String>,
            dimension: usize,
            intra_threads: usize,
        ) -> EmbeddingResult<Self> {
            if dimension == 0 {
                return Err(EmbeddingError::InvalidDimension);
            }
            let model_name = model.into();
            let (tokenizer_path, model_path) =
                download_model_files(&model_name).map_err(EmbeddingError::Provider)?;

            let mut tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| EmbeddingError::Provider(format!("tokenizer load: {e}")))?;
            tokenizer
                .with_padding(Some(tokenizers::PaddingParams {
                    pad_id: 1,
                    pad_token: "<pad>".into(),
                    ..Default::default()
                }))
                .with_truncation(Some(tokenizers::TruncationParams {
                    max_length: 512,
                    ..Default::default()
                }))
                .map_err(|e| EmbeddingError::Provider(format!("tokenizer config: {e}")))?;

            let session = ort::session::Session::builder()
                .map_err(|e| EmbeddingError::Provider(format!("session builder: {e}")))?
                .with_intra_threads(intra_threads)
                .map_err(|e| EmbeddingError::Provider(format!("threads: {e}")))?
                .commit_from_file(&model_path)
                .map_err(|e| EmbeddingError::Provider(format!("session load: {e}")))?;

            let output_names: Vec<String> = session
                .outputs()
                .iter()
                .map(|output| output.name().to_string())
                .collect();

            Ok(Self {
                model_name,
                dimension,
                session: RefCell::new(session),
                tokenizer,
                output_names,
            })
        }
    }

    impl EmbeddingProvider for OnnxEmbeddingProvider {
        fn embed_texts(&self, texts: &[&str]) -> EmbeddingResult<Vec<Vec<f32>>> {
            if texts.is_empty() {
                return Ok(Vec::new());
            }

            let encodings = self
                .tokenizer
                .encode_batch(texts.to_vec(), true)
                .map_err(|e| EmbeddingError::Provider(format!("tokenize: {e}")))?;

            let batch_size = encodings.len();
            let max_len = encodings
                .iter()
                .map(|e| e.get_ids().len())
                .max()
                .unwrap_or(0);

            let mut input_ids_flat = vec![0i64; batch_size * max_len];
            let mut attention_mask_flat = vec![0i64; batch_size * max_len];
            for (row, encoding) in encodings.iter().enumerate() {
                for (column, &id) in encoding.get_ids().iter().enumerate() {
                    input_ids_flat[row * max_len + column] = i64::from(id);
                }
                for (column, &mask) in encoding.get_attention_mask().iter().enumerate() {
                    attention_mask_flat[row * max_len + column] = i64::from(mask);
                }
            }

            let shape = vec![checked_i64(batch_size)?, checked_i64(max_len)?];
            let input_ids_tensor = ort::value::Tensor::from_array((shape.clone(), input_ids_flat))
                .map_err(|e| EmbeddingError::Provider(format!("input_ids tensor: {e}")))?;
            let attention_mask_tensor =
                ort::value::Tensor::from_array((shape, attention_mask_flat))
                    .map_err(|e| EmbeddingError::Provider(format!("attention_mask tensor: {e}")))?;

            let mut session = self.session.borrow_mut();
            let outputs = session
                .run(ort::inputs![input_ids_tensor, attention_mask_tensor])
                .map_err(|e| EmbeddingError::Provider(format!("inference: {e}")))?;

            let dense_output_index = self
                .output_names
                .iter()
                .position(|name| name == "dense_vecs");
            let output_index = dense_output_index.unwrap_or(0);

            let (shape, data) = outputs[output_index]
                .try_extract_tensor::<f32>()
                .map_err(|e| EmbeddingError::Provider(format!("extract: {e}")))?;
            let dims: &[i64] = shape.as_ref();

            // A custom embedding model's output shape is not guaranteed, so the
            // per-row stride and width are derived through checked accessors.
            let (row_stride, row_len) = if dense_output_index.is_some() || dims.len() == 2 {
                let vec_dim = output_axis(dims, 1)?;
                (vec_dim, vec_dim)
            } else if dims.len() == 3 {
                // CLS pooling: take row 0 of each sequence.
                let seq_len = output_axis(dims, 1)?;
                let hidden = output_axis(dims, 2)?;
                (seq_len * hidden, hidden)
            } else {
                let vec_dim = data.len() / batch_size;
                (vec_dim, vec_dim)
            };

            let mut results = Vec::with_capacity(batch_size);
            for row in 0..batch_size {
                let start = row * row_stride;
                let slice = data.get(start..start + row_len).ok_or_else(|| {
                    EmbeddingError::Provider(format!(
                        "model output buffer too short: need {} values, got {}",
                        start + row_len,
                        data.len()
                    ))
                })?;
                let mut vec: Vec<f32> = slice.iter().copied().take(self.dimension).collect();
                l2_normalize(&mut vec);
                results.push(vec);
            }

            Ok(results)
        }
    }

    /// The model this build pins to an immutable revision so the downloaded
    /// weights cannot change underneath users. Must stay in sync with the
    /// config default (`config::EmbeddingConfig::default`). A custom model set
    /// via config is fetched at `main` — the user owns that choice.
    const PINNED_MODEL: &str = "gpahal/bge-m3-onnx-int8";
    const PINNED_MODEL_REVISION: &str = "2b34e84df040034d4b9eabb62383a87c18955822";

    /// Reads a non-negative axis length from a model output shape, failing
    /// when the shape lacks that axis — a custom model whose output rank is
    /// not what the dense/CLS pooling code expects.
    fn output_axis(dims: &[i64], axis: usize) -> EmbeddingResult<usize> {
        let value = dims.get(axis).copied().ok_or_else(|| {
            EmbeddingError::Provider(format!("model output shape {dims:?} has no axis {axis}"))
        })?;
        usize::try_from(value).map_err(|_| {
            EmbeddingError::Provider(format!("model output axis {axis} is negative ({value})"))
        })
    }

    /// Converts a tensor dimension to `i64`, failing rather than wrapping.
    fn checked_i64(value: usize) -> EmbeddingResult<i64> {
        i64::try_from(value)
            .map_err(|_| EmbeddingError::Provider(format!("dimension {value} exceeds i64 range")))
    }

    fn download_model_files(model: &str) -> Result<(PathBuf, PathBuf), String> {
        let api = hf_hub::api::sync::Api::new().map_err(|e| format!("hf-hub init: {e}"))?;
        let repo = if model == PINNED_MODEL {
            api.repo(hf_hub::Repo::with_revision(
                model.to_string(),
                hf_hub::RepoType::Model,
                PINNED_MODEL_REVISION.to_string(),
            ))
        } else {
            api.model(model.to_string())
        };

        let tok_path = repo
            .get("tokenizer.json")
            .map_err(|e| format!("tokenizer download: {e}"))?;

        let onnx_candidates = ["model_quantized.onnx", "model.onnx"];
        let mut model_path = None;
        for candidate in onnx_candidates {
            if let Ok(path) = repo.get(candidate) {
                model_path = Some(path);
                break;
            }
        }
        let model_path = model_path.ok_or_else(|| {
            format!("no ONNX model file found in {model} (tried: {onnx_candidates:?})")
        })?;

        Ok((tok_path, model_path))
    }

    #[cfg(test)]
    mod tests {
        use super::output_axis;

        #[test]
        fn output_axis_reads_a_present_axis() {
            assert_eq!(output_axis(&[2, 768], 1).expect("axis 1 is present"), 768);
        }

        #[test]
        fn output_axis_rejects_a_missing_axis() {
            assert!(output_axis(&[768], 1).is_err());
        }

        #[test]
        fn output_axis_rejects_a_negative_length() {
            assert!(output_axis(&[2, -5], 1).is_err());
        }
    }
}

#[cfg(not(feature = "onnx"))]
pub use stub::OnnxEmbeddingProvider;

#[cfg(feature = "onnx")]
pub use real::OnnxEmbeddingProvider;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_normalize_should_scale_nonzero_vector_to_unit_length() {
        let mut vector = vec![3.0, 4.0];
        l2_normalize(&mut vector);
        assert!((vector[0] - 0.6).abs() < 0.0001);
    }

    #[cfg(not(feature = "onnx"))]
    #[test]
    fn stub_embed_texts_should_return_one_vector_per_text() {
        let provider = OnnxEmbeddingProvider::new("mock", 4, 4).expect("provider should build");
        let embeddings = provider
            .embed_texts(&["hello", "world"])
            .expect("embedding should work");
        assert_eq!(embeddings.len(), 2);
    }

    // Exercises the real ONNX path (download + tokenizer + session + output
    // selection + dimension). Ignored by default because it fetches the
    // ~558MB pinned model; run with `cargo test --features onnx -- --ignored`.
    #[cfg(feature = "onnx")]
    #[test]
    #[ignore = "downloads the pinned ONNX model (~558MB)"]
    fn pinned_onnx_provider_embeds_with_configured_dimension() {
        let dimension = 1024;
        let provider = OnnxEmbeddingProvider::new("gpahal/bge-m3-onnx-int8", dimension, 4)
            .expect("pinned model should load");
        let vectors = provider
            .embed_texts(&["hello world", "claude chron"])
            .expect("embedding should succeed");
        assert_eq!(vectors.len(), 2);
        for vector in &vectors {
            assert_eq!(vector.len(), dimension);
            assert!(vector.iter().all(|value| value.is_finite()));
        }
    }
}
