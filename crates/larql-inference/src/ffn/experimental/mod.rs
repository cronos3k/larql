//! Experimental FFN backends — research artifacts, NOT for production.
//!
//! All backends in this module have known accuracy issues (0% or near-0%).
//! They are preserved for reproducibility of research results.
//!
//! Results:
//! - `graph`: Embedding-based feature selection. Wrong features (1.5% overlap).
//! - `entity_routed`: Preselected features per entity. 4x FFN speed, 0% accuracy.
//! - `clustered`: K-means gate clusters. Activations are distributed, not clustered.
//! - `cached`: Precomputed FFN outputs. Bit-identical, 1us/layer. Not scalable.
//! - `down_clustered`: Output-directed clusters. 0% accuracy.
//! - `feature_list`: Precomputed feature lists. Cascade drift kills accuracy.
//!
//! **Use instead:** `WalkFfn` (sparse mmap'd walk) or `WeightFfn` (dense, exact).

pub mod cached;
pub mod clustered;
pub mod down_clustered;
pub mod entity_routed;
pub mod feature_list;
pub mod graph;
