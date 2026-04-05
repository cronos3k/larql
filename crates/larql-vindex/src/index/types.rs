//! Shared types and traits for the vindex index.

use ndarray::{Array1, Array2};
use larql_models::TopKEntry;

/// Metadata for a single FFN feature (from extraction).
#[derive(Clone)]
pub struct FeatureMeta {
    pub top_token: String,
    pub top_token_id: u32,
    pub c_score: f32,
    pub top_k: Vec<TopKEntry>,
}

/// A single step in the walk trace — one feature that fired at one layer.
pub struct WalkHit {
    pub layer: usize,
    pub feature: usize,
    pub gate_score: f32,
    pub meta: FeatureMeta,
}

/// Result of a walk — per-layer feature activations with full metadata.
pub struct WalkTrace {
    pub layers: Vec<(usize, Vec<WalkHit>)>,
}

/// Trait for gate-based feature lookup.
///
/// Both `VectorIndex` (base, readonly) and `PatchedVindex` (with overlay)
/// implement this trait, allowing `WalkFfn` and other consumers to work
/// transparently with patched or unpatched indexes.
pub trait GateIndex {
    fn gate_knn(&self, layer: usize, residual: &Array1<f32>, top_k: usize) -> Vec<(usize, f32)>;
    fn feature_meta(&self, layer: usize, feature: usize) -> Option<FeatureMeta>;
    fn num_features(&self, layer: usize) -> usize;
    fn down_override(&self, _layer: usize, _feature: usize) -> Option<&[f32]> { None }
    /// Check if any down vector overrides or gate overrides exist at this layer.
    fn has_overrides_at(&self, _layer: usize) -> bool { false }
    fn down_feature_vector(&self, _layer: usize, _feature: usize) -> Option<&[f32]> { None }
    fn has_down_features(&self) -> bool { false }
    fn down_layer_matrix(&self, _layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> { None }
    fn gate_scores_batch(&self, _layer: usize, _x: &Array2<f32>) -> Option<Array2<f32>> { None }
    fn up_layer_matrix(&self, _layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> { None }
    fn has_full_mmap_ffn(&self) -> bool { false }
    fn has_interleaved(&self) -> bool { false }
    fn interleaved_gate(&self, _layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> { None }
    fn interleaved_up(&self, _layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> { None }
    fn interleaved_down(&self, _layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> { None }
    fn prefetch_interleaved_layer(&self, _layer: usize) {}
    fn has_interleaved_q4(&self) -> bool { false }
    fn interleaved_q4_gate(&self, _layer: usize) -> Option<ndarray::Array2<f32>> { None }
    fn interleaved_q4_up(&self, _layer: usize) -> Option<ndarray::Array2<f32>> { None }
    fn interleaved_q4_down(&self, _layer: usize) -> Option<ndarray::Array2<f32>> { None }
    fn prefetch_interleaved_q4_layer(&self, _layer: usize) {}
    fn interleaved_q4_mmap_ref(&self) -> Option<&[u8]> { None }

    /// Per-feature gate scoring: iterate all features, dot product each one.
    /// No matrix multiplication — each feature scored individually.
    /// Returns (feature_index, score) sorted by absolute score descending.
    fn gate_walk(&self, _layer: usize, _residual: &Array1<f32>, _top_k: usize) -> Option<Vec<(usize, f32)>> {
        None // Override in VectorIndex to use mmap
    }

    fn gate_knn_batch(&self, layer: usize, x: &Array2<f32>, top_k: usize) -> Vec<usize> {
        let seq_len = x.shape()[0];
        let mut all = std::collections::BTreeSet::new();
        for s in 0..seq_len {
            let row = x.row(s).to_owned();
            for (feat, _) in self.gate_knn(layer, &row, top_k) {
                all.insert(feat);
            }
        }
        all.into_iter().collect()
    }
}

/// Progress callbacks for index loading.
pub trait IndexLoadCallbacks {
    fn on_file_start(&mut self, _component: &str, _path: &str) {}
    fn on_progress(&mut self, _records: usize) {}
    fn on_file_done(&mut self, _component: &str, _records: usize, _elapsed_ms: f64) {}
}

pub struct SilentLoadCallbacks;
impl IndexLoadCallbacks for SilentLoadCallbacks {}

/// Per-layer gate vector offset info for mmap mode.
#[derive(Clone)]
pub struct GateLayerSlice {
    pub float_offset: usize,
    pub num_features: usize,
}

/// Per-layer Q4 gate data offset info.
#[derive(Clone)]
pub struct GateQ4Slice {
    pub byte_offset: usize,
    pub byte_len: usize,
    pub num_features: usize,
}

/// Mmap'd down_meta.bin — reads individual feature records on demand.
#[derive(Clone)]
pub struct DownMetaMmap {
    pub(crate) mmap: std::sync::Arc<memmap2::Mmap>,
    pub(crate) layer_offsets: Vec<usize>,
    pub(crate) layer_num_features: Vec<usize>,
    pub(crate) top_k_count: usize,
    pub(crate) tokenizer: std::sync::Arc<tokenizers::Tokenizer>,
}

impl DownMetaMmap {
    fn record_size(&self) -> usize {
        8 + self.top_k_count * 8
    }

    pub fn feature_meta(&self, layer: usize, feature: usize) -> Option<FeatureMeta> {
        if layer >= self.layer_offsets.len() { return None; }
        let num_features = self.layer_num_features[layer];
        if num_features == 0 || feature >= num_features { return None; }

        let offset = self.layer_offsets[layer] + feature * self.record_size();
        let rec_size = self.record_size();
        if offset + rec_size > self.mmap.len() { return None; }

        let b = &self.mmap[offset..offset + rec_size];
        let top_token_id = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        let c_score = f32::from_le_bytes([b[4], b[5], b[6], b[7]]);

        if top_token_id == 0 && c_score == 0.0 { return None; }

        let mut top_k = Vec::new();
        for i in 0..self.top_k_count {
            let o = 8 + i * 8;
            let tid = u32::from_le_bytes([b[o], b[o+1], b[o+2], b[o+3]]);
            let logit = f32::from_le_bytes([b[o+4], b[o+5], b[o+6], b[o+7]]);
            if tid > 0 || logit != 0.0 {
                let token = self.tokenizer.decode(&[tid], true)
                    .unwrap_or_else(|_| format!("T{tid}")).trim().to_string();
                top_k.push(TopKEntry { token, token_id: tid, logit });
            }
        }

        let top_token = self.tokenizer.decode(&[top_token_id], true)
            .unwrap_or_else(|_| format!("T{top_token_id}")).trim().to_string();

        Some(FeatureMeta { top_token, top_token_id, c_score, top_k })
    }

    pub fn num_features(&self, layer: usize) -> usize {
        self.layer_num_features.get(layer).copied().unwrap_or(0)
    }

    pub fn total_features(&self) -> usize {
        self.layer_num_features.iter().sum()
    }
}
