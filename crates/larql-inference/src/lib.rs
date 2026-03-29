extern crate blas_src;

pub mod attention;
pub mod capture;
pub mod error;
pub mod ffn;
pub mod forward;
pub mod model;
pub mod residual;
pub mod tokenizer;
pub mod walker;

// Re-export dependencies for downstream crates.
pub use larql_models;
pub use ndarray;
pub use safetensors;
pub use tokenizers;

// Re-export essentials at crate root.
pub use capture::{
    CaptureCallbacks, CaptureConfig, InferenceModel, TopKEntry, VectorFileHeader, VectorRecord,
};
pub use error::InferenceError;
pub use forward::{capture_residuals, predict, trace_forward, PredictResult, TraceResult};
pub use model::{load_model_dir, resolve_model_path, ModelWeights};
pub use tokenizer::{decode_token, load_tokenizer};

// Walker re-exports.
pub use walker::attention_walker::{AttentionLayerResult, AttentionWalker};
pub use walker::vector_extractor::{
    ExtractCallbacks, ExtractConfig, ExtractSummary, VectorExtractor,
};
pub use walker::weight_walker::{
    walk_model, LayerResult, LayerStats, WalkCallbacks, WalkConfig, WeightWalker,
};
