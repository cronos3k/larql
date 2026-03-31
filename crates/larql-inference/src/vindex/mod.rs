//! Vindex build pipeline, weight management, and walk FFN.
//!
//! Core vindex types live in `larql-vindex`. This module adds:
//! - Build pipeline (EXTRACT: model weights → vindex)
//! - Weight IO (model_weights.bin read/write)
//! - WalkFfn (FFN backend using vindex KNN for interpretability)

mod build;
mod build_from_vectors;
mod walk_ffn;
mod weights;

pub use build::{build_vindex, build_vindex_resume, IndexBuildCallbacks, SilentBuildCallbacks};
pub use build_from_vectors::build_vindex_from_vectors;
pub use walk_ffn::WalkFfn;
pub use weights::{find_tokenizer_path, load_model_weights_from_vindex, write_model_weights};
