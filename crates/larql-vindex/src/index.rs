//! VectorIndex struct and core operations: load_gates, load_down_meta, gate_knn, walk.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::Path;

use ndarray::{Array1, Array2};

use crate::error::VindexError;

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
    /// Per-layer hits, sorted by gate score descending.
    pub layers: Vec<(usize, Vec<WalkHit>)>,
}

/// Progress callbacks for index loading.
pub trait IndexLoadCallbacks {
    fn on_file_start(&mut self, _component: &str, _path: &str) {}
    fn on_progress(&mut self, _records: usize) {}
    fn on_file_done(&mut self, _component: &str, _records: usize, _elapsed_ms: f64) {}
}

pub struct SilentLoadCallbacks;
impl IndexLoadCallbacks for SilentLoadCallbacks {}

/// The full model as a local vector index.
///
/// Gate vectors for KNN matching + down token metadata for output lookup.
/// Loaded from the NDJSON files produced by `vector-extract`.
pub struct VectorIndex {
    /// Per-layer gate vectors: gate_vectors[layer] is (num_features, hidden_size).
    /// Used for KNN via BLAS matmul.
    pub(crate) gate_vectors: Vec<Option<Array2<f32>>>,

    /// Per-layer, per-feature output token metadata from down projections.
    /// down_meta[layer][feature] = FeatureMeta with top tokens.
    pub(crate) down_meta: Vec<Option<Vec<Option<FeatureMeta>>>>,

    /// Number of layers in the model.
    pub num_layers: usize,

    /// Hidden dimension.
    pub hidden_size: usize,
}

impl VectorIndex {
    /// Create a new VectorIndex from components.
    pub fn new(
        gate_vectors: Vec<Option<Array2<f32>>>,
        down_meta: Vec<Option<Vec<Option<FeatureMeta>>>>,
        num_layers: usize,
        hidden_size: usize,
    ) -> Self {
        Self {
            gate_vectors,
            down_meta,
            num_layers,
            hidden_size,
        }
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
            down_meta: gate_meta, // gate meta initially; down loaded separately
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

    /// Gate KNN: find the top-K features at a layer whose gate vectors have
    /// the highest dot product with the input residual. Uses BLAS matmul.
    ///
    /// Returns (feature_index, dot_product) sorted by absolute magnitude descending.
    pub fn gate_knn(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        let gate_matrix = match self.gate_vectors.get(layer).and_then(|v| v.as_ref()) {
            Some(m) => m,
            None => return vec![],
        };

        // gate_proj = gate_matrix @ residual → (num_features,)
        let scores = gate_matrix.dot(residual);

        // Top-K by dot product magnitude
        let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        let k = top_k.min(indexed.len());
        if k > 0 && k < indexed.len() {
            indexed.select_nth_unstable_by(k, |a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
            indexed.truncate(k);
        }
        indexed.sort_unstable_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        indexed
    }

    /// Full walk: gate KNN at each layer, annotated with down token metadata.
    pub fn walk(
        &self,
        residual: &Array1<f32>,
        layers: &[usize],
        top_k: usize,
    ) -> WalkTrace {
        let mut trace_layers = Vec::with_capacity(layers.len());

        for &layer in layers {
            let hits = self.gate_knn(layer, residual, top_k);
            let walk_hits: Vec<WalkHit> = hits
                .into_iter()
                .filter_map(|(feature, gate_score)| {
                    let meta = self
                        .down_meta
                        .get(layer)
                        .and_then(|v| v.as_ref())
                        .and_then(|metas| metas.get(feature))
                        .and_then(|m| m.as_ref())
                        .cloned()?;
                    Some(WalkHit {
                        layer,
                        feature,
                        gate_score,
                        meta,
                    })
                })
                .collect();
            trace_layers.push((layer, walk_hits));
        }

        WalkTrace {
            layers: trace_layers,
        }
    }

    /// Look up metadata for a specific feature.
    pub fn feature_meta(&self, layer: usize, feature: usize) -> Option<&FeatureMeta> {
        self.down_meta
            .get(layer)
            .and_then(|v| v.as_ref())
            .and_then(|metas| metas.get(feature))
            .and_then(|m| m.as_ref())
    }

    /// Number of features indexed at a layer.
    pub fn num_features(&self, layer: usize) -> usize {
        self.gate_vectors
            .get(layer)
            .and_then(|v| v.as_ref())
            .map(|m| m.shape()[0])
            .unwrap_or(0)
    }

    /// Total gate vectors loaded across all layers.
    pub fn total_gate_vectors(&self) -> usize {
        self.gate_vectors
            .iter()
            .filter_map(|v| v.as_ref())
            .map(|m| m.shape()[0])
            .sum()
    }

    /// Total down metadata entries loaded across all layers.
    pub fn total_down_meta(&self) -> usize {
        self.down_meta
            .iter()
            .filter_map(|v| v.as_ref())
            .map(|metas| metas.iter().filter(|m| m.is_some()).count())
            .sum()
    }

    /// Layers that have gate vectors loaded.
    pub fn loaded_layers(&self) -> Vec<usize> {
        self.gate_vectors
            .iter()
            .enumerate()
            .filter_map(|(i, v)| v.as_ref().map(|_| i))
            .collect()
    }

    /// Access down metadata for a specific layer.
    pub fn down_meta_at(&self, layer: usize) -> Option<&[Option<FeatureMeta>]> {
        self.down_meta
            .get(layer)
            .and_then(|v| v.as_ref())
            .map(|v| v.as_slice())
    }

    /// Access gate vectors matrix for a specific layer.
    pub fn gate_vectors_at(&self, layer: usize) -> Option<&Array2<f32>> {
        self.gate_vectors.get(layer).and_then(|v| v.as_ref())
    }
}
