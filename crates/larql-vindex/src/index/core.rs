//! VectorIndex struct and core operations.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::{Arc, Mutex};

use ndarray::{Array1, Array2};

use crate::error::VindexError;
use larql_models::TopKEntry;

// Re-export all shared types from types.rs.
pub use super::types::*;

/// The full model as a local vector index.
///
/// Gate vectors for KNN matching + down token metadata for output lookup.
/// Supports two storage modes:
/// - **Heap**: gate vectors copied into per-layer Array2 (in-memory builds, mutations)
/// - **Mmap**: gate vectors sliced directly from mmap'd file (zero-copy, zero heap)
pub struct VectorIndex {
    /// Per-layer gate vectors (heap mode): gate_vectors[layer] is (num_features, hidden_size).
    pub(crate) gate_vectors: Vec<Option<Array2<f32>>>,

    /// Mmap'd gate vector bytes (zero-copy mode). When set, gate_knn slices
    /// directly from this instead of using gate_vectors heap arrays.
    /// For f32: bytes are reinterpreted as &[f32] directly (zero-copy).
    /// For f16: bytes are decoded per-layer on demand.
    /// Arc for Clone support — the mmap is shared, not copied.
    pub(crate) gate_mmap_bytes: Option<Arc<memmap2::Mmap>>,

    /// Storage dtype for mmap'd data (needed for f16 decoding).
    pub(crate) gate_mmap_dtype: crate::config::dtype::StorageDtype,

    /// Per-layer slice info for mmap mode.
    pub(crate) gate_mmap_slices: Vec<GateLayerSlice>,

    /// Per-layer, per-feature output token metadata from down projections.
    /// down_meta[layer][feature] = FeatureMeta with top tokens.
    /// Heap mode: populated during builds or when loaded from JSONL.
    pub(crate) down_meta: Vec<Option<Vec<Option<FeatureMeta>>>>,

    /// Mmap'd down_meta.bin bytes (zero-copy mode).
    /// When set, feature_meta() reads records on demand from the mmap.
    pub(crate) down_meta_mmap: Option<Arc<DownMetaMmap>>,

    /// Number of layers in the model.
    pub num_layers: usize,

    /// Hidden dimension.
    pub hidden_size: usize,

    /// Down vector overrides: custom output vectors for specific features.
    /// When set, sparse_ffn_forward uses this instead of the model's down weight row.
    /// Key: (layer, feature), Value: hidden_size f32 vector.
    pub(crate) down_overrides: HashMap<(usize, usize), Vec<f32>>,

    /// Lazy decode cache for f16 gate vectors. Each layer decoded once on first
    /// KNN call, then reused. Eliminates repeated f16→f32 conversion.
    pub(crate) f16_decode_cache: Mutex<Vec<Option<Vec<f32>>>>,
    pub(crate) warmed_gates: std::sync::RwLock<Vec<Option<Vec<f32>>>>,
    pub(crate) down_features_mmap: Option<Arc<memmap2::Mmap>>,
    pub(crate) up_features_mmap: Option<Arc<memmap2::Mmap>>,
    pub(crate) hnsw_cache: Mutex<Vec<Option<super::hnsw::HnswLayer>>>,
    pub(crate) hnsw_enabled: std::sync::atomic::AtomicBool,
    pub(crate) hnsw_ef_search: std::sync::atomic::AtomicUsize,
    /// Mmap'd lm_head (output projection): [vocab_size, hidden_size], f32.
    pub(crate) lm_head_mmap: Option<Arc<memmap2::Mmap>>,
    pub vocab_size: usize,
    /// Interleaved FFN data: [gate|up|down] per layer in one contiguous file.
    pub(crate) interleaved_mmap: Option<Arc<memmap2::Mmap>>,
    /// Q4_0 quantized interleaved FFN data (7x smaller, dequant on read).
    pub(crate) interleaved_q4_mmap: Option<Arc<memmap2::Mmap>>,

    /// Q4_0 gate vectors mmap — for fast Q4 KNN via larql-compute.
    /// Layout: layers packed contiguously, each [num_features, hidden] in Q4_0 format.
    pub(crate) gate_q4_mmap: Option<Arc<memmap2::Mmap>>,
    /// Per-layer byte offset + byte length in gate_q4_mmap.
    pub(crate) gate_q4_slices: Vec<GateQ4Slice>,
}

impl Clone for VectorIndex {
    fn clone(&self) -> Self {
        use std::sync::atomic::Ordering;
        Self {
            gate_vectors: self.gate_vectors.clone(),
            gate_mmap_bytes: self.gate_mmap_bytes.clone(),
            gate_mmap_dtype: self.gate_mmap_dtype,
            gate_mmap_slices: self.gate_mmap_slices.clone(),
            down_meta: self.down_meta.clone(),
            down_meta_mmap: self.down_meta_mmap.clone(),
            num_layers: self.num_layers,
            hidden_size: self.hidden_size,
            down_overrides: self.down_overrides.clone(),
            f16_decode_cache: Mutex::new(vec![None; self.num_layers]),
            warmed_gates: std::sync::RwLock::new(vec![None; self.num_layers]),
            down_features_mmap: self.down_features_mmap.clone(),
            up_features_mmap: self.up_features_mmap.clone(),
            hnsw_cache: Mutex::new((0..self.num_layers).map(|_| None).collect()),
            hnsw_enabled: std::sync::atomic::AtomicBool::new(
                self.hnsw_enabled.load(Ordering::Relaxed)
            ),
            hnsw_ef_search: std::sync::atomic::AtomicUsize::new(
                self.hnsw_ef_search.load(Ordering::Relaxed)
            ),
            lm_head_mmap: self.lm_head_mmap.clone(),
            vocab_size: self.vocab_size,
            interleaved_mmap: self.interleaved_mmap.clone(),
            interleaved_q4_mmap: self.interleaved_q4_mmap.clone(),
            gate_q4_mmap: self.gate_q4_mmap.clone(),
            gate_q4_slices: self.gate_q4_slices.clone(),
        }
    }
}

impl VectorIndex {
    /// Create a new VectorIndex from heap-allocated components (in-memory builds).
    pub fn new(
        gate_vectors: Vec<Option<Array2<f32>>>,
        down_meta: Vec<Option<Vec<Option<FeatureMeta>>>>,
        num_layers: usize,
        hidden_size: usize,
    ) -> Self {
        Self {
            gate_vectors,
            gate_mmap_bytes: None,
            gate_mmap_dtype: crate::config::dtype::StorageDtype::F32,
            gate_mmap_slices: Vec::new(),
            down_meta,
            down_meta_mmap: None,
            num_layers,
            hidden_size,
            down_overrides: HashMap::new(),
            f16_decode_cache: Mutex::new(vec![None; num_layers]),
            warmed_gates: std::sync::RwLock::new(vec![None; num_layers]),
            down_features_mmap: None,
            up_features_mmap: None,
            hnsw_cache: Mutex::new((0..num_layers).map(|_| None).collect()),
            hnsw_enabled: std::sync::atomic::AtomicBool::new(false),
            hnsw_ef_search: std::sync::atomic::AtomicUsize::new(200),
            lm_head_mmap: None,
            vocab_size: 0,
            interleaved_mmap: None,
            interleaved_q4_mmap: None,
            gate_q4_mmap: None,
            gate_q4_slices: Vec::new(),
        }
    }

    /// Create a VectorIndex with zero-copy mmap'd gate vectors and down_meta.
    /// No heap allocation — everything read on demand from mmap'd files.
    pub fn new_mmap(
        gate_mmap: memmap2::Mmap,
        gate_slices: Vec<GateLayerSlice>,
        dtype: crate::config::dtype::StorageDtype,
        down_meta_mmap: Option<DownMetaMmap>,
        num_layers: usize,
        hidden_size: usize,
    ) -> Self {
        Self {
            gate_vectors: vec![None; num_layers],
            gate_mmap_bytes: Some(Arc::new(gate_mmap)),
            gate_mmap_dtype: dtype,
            gate_mmap_slices: gate_slices,
            down_meta: vec![None; num_layers],
            down_meta_mmap: down_meta_mmap.map(Arc::new),
            num_layers,
            hidden_size,
            down_overrides: HashMap::new(),
            f16_decode_cache: Mutex::new(vec![None; num_layers]),
            warmed_gates: std::sync::RwLock::new(vec![None; num_layers]),
            down_features_mmap: None,
            up_features_mmap: None,
            hnsw_cache: Mutex::new((0..num_layers).map(|_| None).collect()),
            hnsw_enabled: std::sync::atomic::AtomicBool::new(false),
            hnsw_ef_search: std::sync::atomic::AtomicUsize::new(200),
            lm_head_mmap: None,
            vocab_size: 0,
            interleaved_mmap: None,
            interleaved_q4_mmap: None,
            gate_q4_mmap: None,
            gate_q4_slices: Vec::new(),
        }
    }

    /// Returns true if this index uses mmap'd gate vectors (zero heap copy).
    pub fn is_mmap(&self) -> bool {
        self.gate_mmap_bytes.is_some()
    }

    /// Estimated heap bytes used by gate vectors (0 if mmap'd).
    pub fn gate_heap_bytes(&self) -> usize {
        if self.is_mmap() {
            return 0;
        }
        self.gate_vectors.iter()
            .filter_map(|v| v.as_ref())
            .map(|m| m.len() * std::mem::size_of::<f32>())
            .sum()
    }

    /// Load gate vectors from an NDJSON file (ffn_gate.vectors.jsonl).
    ///
    /// Each line is a VectorRecord with layer, feature, vector, top_token, etc.
    /// Vectors are packed into per-layer Array2 matrices for BLAS matmul.
    pub fn load_gates(
        path: &Path,
        callbacks: &mut dyn IndexLoadCallbacks,
    ) -> Result<Self, VindexError> {
        callbacks.on_file_start("ffn_gate", &path.display().to_string());
        let start = std::time::Instant::now();

        let file = std::fs::File::open(path)?;
        let reader = BufReader::with_capacity(1 << 20, file);

        // First pass: collect all records to determine dimensions
        let mut records: Vec<(usize, usize, Vec<f32>, FeatureMeta)> = Vec::new();
        let mut hidden_size = 0;
        let mut max_layer = 0;
        let mut count = 0;

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let obj: serde_json::Value =
                serde_json::from_str(line).map_err(|e| VindexError::Parse(e.to_string()))?;

            if obj.get("_header").is_some() {
                if let Some(dim) = obj.get("dimension").and_then(|v| v.as_u64()) {
                    hidden_size = dim as usize;
                }
                continue;
            }

            let layer = obj["layer"].as_u64().unwrap() as usize;
            let feature = obj["feature"].as_u64().unwrap() as usize;

            let vector: Vec<f32> = obj["vector"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();

            if hidden_size == 0 {
                hidden_size = vector.len();
            }

            let top_token = obj["top_token"].as_str().unwrap_or("").to_string();
            let top_token_id = obj["top_token_id"].as_u64().unwrap_or(0) as u32;
            let c_score = obj["c_score"].as_f64().unwrap_or(0.0) as f32;

            let top_k: Vec<TopKEntry> = match obj.get("top_k").and_then(|v| v.as_array()) {
                Some(arr) => arr
                    .iter()
                    .filter_map(|entry| {
                        Some(TopKEntry {
                            token: entry.get("token")?.as_str()?.to_string(),
                            token_id: entry.get("token_id")?.as_u64()? as u32,
                            logit: entry.get("logit")?.as_f64()? as f32,
                        })
                    })
                    .collect(),
                None => vec![],
            };

            let meta = FeatureMeta {
                top_token,
                top_token_id,
                c_score,
                top_k,
            };

            if layer > max_layer {
                max_layer = layer;
            }

            records.push((layer, feature, vector, meta));

            count += 1;
            if count % 10000 == 0 {
                callbacks.on_progress(count);
            }
        }

        let num_layers = max_layer + 1;

        // Group by layer, find max feature per layer
        let mut layer_sizes: HashMap<usize, usize> = HashMap::new();
        for &(layer, feature, _, _) in &records {
            let entry = layer_sizes.entry(layer).or_insert(0);
            if feature + 1 > *entry {
                *entry = feature + 1;
            }
        }

        // Build per-layer matrices
        let mut gate_vectors: Vec<Option<Array2<f32>>> = vec![None; num_layers];
        let mut gate_meta: Vec<Option<Vec<Option<FeatureMeta>>>> = vec![None; num_layers];

        // Pre-allocate
        for (&layer, &num_features) in &layer_sizes {
            gate_vectors[layer] = Some(Array2::zeros((num_features, hidden_size)));
            gate_meta[layer] = Some(vec![None; num_features]);
        }

        // Fill
        for (layer, feature, vector, meta) in records {
            if let Some(ref mut matrix) = gate_vectors[layer] {
                for (j, &val) in vector.iter().enumerate() {
                    matrix[[feature, j]] = val;
                }
            }
            if let Some(ref mut metas) = gate_meta[layer] {
                metas[feature] = Some(meta);
            }
        }

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        callbacks.on_file_done("ffn_gate", count, elapsed_ms);

        Ok(VectorIndex {
            gate_vectors,
            gate_mmap_bytes: None,
            gate_mmap_dtype: crate::config::dtype::StorageDtype::F32,
            gate_mmap_slices: Vec::new(),
            down_meta: gate_meta,
            down_meta_mmap: None,
            down_overrides: HashMap::new(),
            f16_decode_cache: Mutex::new(vec![None; num_layers]),
            warmed_gates: std::sync::RwLock::new(vec![None; num_layers]),
            down_features_mmap: None,
            up_features_mmap: None,
            hnsw_cache: Mutex::new((0..num_layers).map(|_| None).collect()),
            hnsw_enabled: std::sync::atomic::AtomicBool::new(false),
            hnsw_ef_search: std::sync::atomic::AtomicUsize::new(200),
            lm_head_mmap: None,
            vocab_size: 0,
            interleaved_mmap: None,
            interleaved_q4_mmap: None,
            gate_q4_mmap: None,
            gate_q4_slices: Vec::new(),
            num_layers,
            hidden_size,
        })
    }

    /// Load down-projection token metadata from an NDJSON file (ffn_down.vectors.jsonl).
    ///
    /// Only loads the metadata (top_token, top_k, c_score), NOT the full vectors.
    /// This replaces any gate-file metadata with the down-projection metadata,
    /// which tells you what each feature *outputs* rather than what it *responds to*.
    pub fn load_down_meta(
        &mut self,
        path: &Path,
        callbacks: &mut dyn IndexLoadCallbacks,
    ) -> Result<usize, VindexError> {
        callbacks.on_file_start("ffn_down", &path.display().to_string());
        let start = std::time::Instant::now();

        let file = std::fs::File::open(path)?;
        let reader = BufReader::with_capacity(1 << 20, file);
        let mut count = 0;

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let obj: serde_json::Value =
                serde_json::from_str(line).map_err(|e| VindexError::Parse(e.to_string()))?;

            if obj.get("_header").is_some() {
                continue;
            }

            let layer = obj["layer"].as_u64().unwrap() as usize;
            let feature = obj["feature"].as_u64().unwrap() as usize;

            let top_token = obj["top_token"].as_str().unwrap_or("").to_string();
            let top_token_id = obj["top_token_id"].as_u64().unwrap_or(0) as u32;
            let c_score = obj["c_score"].as_f64().unwrap_or(0.0) as f32;

            let top_k: Vec<TopKEntry> = match obj.get("top_k").and_then(|v| v.as_array()) {
                Some(arr) => arr
                    .iter()
                    .filter_map(|entry| {
                        Some(TopKEntry {
                            token: entry.get("token")?.as_str()?.to_string(),
                            token_id: entry.get("token_id")?.as_u64()? as u32,
                            logit: entry.get("logit")?.as_f64()? as f32,
                        })
                    })
                    .collect(),
                None => vec![],
            };

            let meta = FeatureMeta {
                top_token,
                top_token_id,
                c_score,
                top_k,
            };

            if layer < self.num_layers {
                // Ensure layer slot exists
                while self.down_meta.len() <= layer {
                    self.down_meta.push(None);
                }
                if self.down_meta[layer].is_none() {
                    self.down_meta[layer] = Some(Vec::new());
                }
                if let Some(ref mut metas) = self.down_meta[layer] {
                    while metas.len() <= feature {
                        metas.push(None);
                    }
                    metas[feature] = Some(meta);
                }
            }

            count += 1;
            if count % 10000 == 0 {
                callbacks.on_progress(count);
            }
        }

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        callbacks.on_file_done("ffn_down", count, elapsed_ms);

        Ok(count)
    }

}

impl GateIndex for VectorIndex {
    fn gate_knn(&self, layer: usize, residual: &Array1<f32>, top_k: usize) -> Vec<(usize, f32)> {
        self.gate_knn(layer, residual, top_k)
    }

    fn feature_meta(&self, layer: usize, feature: usize) -> Option<FeatureMeta> {
        self.feature_meta(layer, feature)
    }

    fn num_features(&self, layer: usize) -> usize {
        self.num_features(layer)
    }

    fn down_override(&self, layer: usize, feature: usize) -> Option<&[f32]> {
        self.down_overrides.get(&(layer, feature)).map(|v| v.as_slice())
    }

    fn has_overrides_at(&self, layer: usize) -> bool {
        self.down_overrides.keys().any(|(l, _)| *l == layer)
    }

    fn gate_knn_batch(&self, layer: usize, x: &Array2<f32>, top_k: usize) -> Vec<usize> {
        self.gate_knn_batch(layer, x, top_k)
    }

    fn down_feature_vector(&self, layer: usize, feature: usize) -> Option<&[f32]> {
        self.down_feature_vector(layer, feature)
    }

    fn has_down_features(&self) -> bool {
        self.has_down_features()
    }

    fn down_layer_matrix(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.down_layer_matrix(layer)
    }

    fn gate_scores_batch(&self, layer: usize, x: &Array2<f32>) -> Option<Array2<f32>> {
        self.gate_scores_batch(layer, x)
    }

    fn up_layer_matrix(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.up_layer_matrix(layer)
    }

    fn has_full_mmap_ffn(&self) -> bool {
        self.has_full_mmap_ffn()
    }

    fn has_interleaved(&self) -> bool {
        self.has_interleaved()
    }

    fn interleaved_gate(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.interleaved_gate(layer)
    }

    fn interleaved_up(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.interleaved_up(layer)
    }

    fn interleaved_down(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.interleaved_down(layer)
    }

    fn prefetch_interleaved_layer(&self, layer: usize) {
        self.prefetch_interleaved_layer(layer)
    }

    fn has_interleaved_q4(&self) -> bool {
        self.has_interleaved_q4()
    }

    fn interleaved_q4_gate(&self, layer: usize) -> Option<ndarray::Array2<f32>> {
        self.interleaved_q4_gate(layer)
    }

    fn interleaved_q4_up(&self, layer: usize) -> Option<ndarray::Array2<f32>> {
        self.interleaved_q4_up(layer)
    }

    fn interleaved_q4_down(&self, layer: usize) -> Option<ndarray::Array2<f32>> {
        self.interleaved_q4_down(layer)
    }

    fn prefetch_interleaved_q4_layer(&self, layer: usize) {
        self.prefetch_interleaved_q4_layer(layer)
    }

    fn interleaved_q4_mmap_ref(&self) -> Option<&[u8]> {
        self.interleaved_q4_mmap.as_ref().map(|m| m.as_ref() as &[u8])
    }
}
