//! Patch system — lightweight, shareable knowledge diffs.

pub mod core;
pub mod knn_store;
pub mod refine;

pub use core::*;
pub use knn_store::{KnnStore, KnnEntry};
pub use refine::{refine_gates, RefineInput, RefineResult, RefinedGate};
