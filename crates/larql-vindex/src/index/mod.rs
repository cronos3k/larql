//! VectorIndex — the in-memory KNN engine, mutation interface, and MoE router.

pub mod core;
pub mod mutate;
pub mod router;

pub use core::*;
pub use router::RouterIndex;
