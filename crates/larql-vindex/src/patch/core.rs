#![allow(dead_code)]
//! Vindex patch system — lightweight, shareable knowledge diffs.
//!
//! A patch (.vlp file) captures INSERT, DELETE, and UPDATE operations
//! as a portable JSON file. Patches overlay an immutable base vindex
//! without modifying its files on disk.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::VindexError;
use crate::index::{FeatureMeta, GateIndex, VectorIndex, WalkHit, WalkTrace};

use ndarray::Array1;

// ═══════════════════════════════════════════════════════════════
// Patch data types
// ═══════════════════════════════════════════════════════════════

/// A vindex patch — a set of operations to apply to a base vindex.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VindexPatch {
    pub version: u32,
    pub base_model: String,
    #[serde(default)]
    pub base_checksum: Option<String>,
    pub created_at: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub author: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    pub operations: Vec<PatchOp>,
}

/// A single patch operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "lowercase")]
pub enum PatchOp {
    Insert {
        layer: usize,
        feature: usize,
        #[serde(default)]
        relation: Option<String>,
        entity: String,
        target: String,
        #[serde(default)]
        confidence: Option<f32>,
        /// Base64-encoded f32 gate vector.
        #[serde(default)]
        gate_vector_b64: Option<String>,
        #[serde(default)]
        down_meta: Option<PatchDownMeta>,
    },
    Update {
        layer: usize,
        feature: usize,
        #[serde(default)]
        gate_vector_b64: Option<String>,
        #[serde(default)]
        down_meta: Option<PatchDownMeta>,
    },
    Delete {
        layer: usize,
        feature: usize,
        #[serde(default)]
        reason: Option<String>,
    },
}

/// Compact down_meta for a patch operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchDownMeta {
    #[serde(rename = "t")]
    pub top_token: String,
    #[serde(rename = "i")]
    pub top_token_id: u32,
    #[serde(rename = "c")]
    pub c_score: f32,
}

impl PatchOp {
    /// The (layer, feature) this operation targets.
    pub fn key(&self) -> (usize, usize) {
        match self {
            PatchOp::Insert { layer, feature, .. } => (*layer, *feature),
            PatchOp::Update { layer, feature, .. } => (*layer, *feature),
            PatchOp::Delete { layer, feature, .. } => (*layer, *feature),
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Patch file I/O
// ═══════════════════════════════════════════════════════════════

impl VindexPatch {
    /// Write patch to a .vlp file.
    pub fn save(&self, path: &Path) -> Result<(), VindexError> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| VindexError::Parse(e.to_string()))?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load patch from a .vlp file.
    pub fn load(path: &Path) -> Result<Self, VindexError> {
        let text = std::fs::read_to_string(path)?;
        let patch: VindexPatch = serde_json::from_str(&text)
            .map_err(|e| VindexError::Parse(e.to_string()))?;
        Ok(patch)
    }

    /// Number of operations in this patch.
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// Summary counts: (inserts, updates, deletes).
    pub fn counts(&self) -> (usize, usize, usize) {
        let mut ins = 0;
        let mut upd = 0;
        let mut del = 0;
        for op in &self.operations {
            match op {
                PatchOp::Insert { .. } => ins += 1,
                PatchOp::Update { .. } => upd += 1,
                PatchOp::Delete { .. } => del += 1,
            }
        }
        (ins, upd, del)
    }
}

// ═══════════════════════════════════════════════════════════════
// Base64 gate vector encoding
// ═══════════════════════════════════════════════════════════════

/// Encode a gate vector (f32 slice) as base64 string.
pub fn encode_gate_vector(vec: &[f32]) -> String {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(vec.as_ptr() as *const u8, vec.len() * 4)
    };
    base64_encode(bytes)
}

/// Decode a base64 string back to f32 vector.
pub fn decode_gate_vector(b64: &str) -> Result<Vec<f32>, VindexError> {
    let bytes = base64_decode(b64)?;
    if bytes.len() % 4 != 0 {
        return Err(VindexError::Parse("gate vector bytes not aligned to f32".into()));
    }
    let floats: Vec<f32> = unsafe {
        std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4)
    }
    .to_vec();
    Ok(floats)
}

// Simple base64 (no external dependency)
#[allow(dead_code)]
fn base64_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        result.push(CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 { result.push(CHARS[((triple >> 6) & 0x3F) as usize] as char); } else { result.push('='); }
        if chunk.len() > 2 { result.push(CHARS[(triple & 0x3F) as usize] as char); } else { result.push('='); }
    }
    result
}

fn base64_decode(input: &str) -> Result<Vec<u8>, VindexError> {
    fn val(c: u8) -> Result<u32, VindexError> {
        match c {
            b'A'..=b'Z' => Ok((c - b'A') as u32),
            b'a'..=b'z' => Ok((c - b'a' + 26) as u32),
            b'0'..=b'9' => Ok((c - b'0' + 52) as u32),
            b'+' => Ok(62),
            b'/' => Ok(63),
            b'=' => Ok(0),
            _ => Err(VindexError::Parse(format!("invalid base64 char: {c}"))),
        }
    }
    let input = input.as_bytes();
    let mut result = Vec::with_capacity(input.len() * 3 / 4);
    for chunk in input.chunks(4) {
        if chunk.len() < 4 { break; }
        let a = val(chunk[0])?;
        let b = val(chunk[1])?;
        let c = val(chunk[2])?;
        let d = val(chunk[3])?;
        let triple = (a << 18) | (b << 12) | (c << 6) | d;
        result.push(((triple >> 16) & 0xFF) as u8);
        if chunk[2] != b'=' { result.push(((triple >> 8) & 0xFF) as u8); }
        if chunk[3] != b'=' { result.push((triple & 0xFF) as u8); }
    }
    Ok(result)
}

// ═══════════════════════════════════════════════════════════════
// PatchedVindex — overlay on immutable base
// ═══════════════════════════════════════════════════════════════

/// A vindex with patches applied as an overlay.
/// The base files on disk are never modified.
pub struct PatchedVindex {
    /// Immutable base index.
    pub base: VectorIndex,
    /// Applied patches (in order).
    pub patches: Vec<VindexPatch>,
    /// Resolved overrides: (layer, feature) → effective operation.
    /// Later patches override earlier ones for the same feature.
    pub(crate) overrides_meta: HashMap<(usize, usize), Option<FeatureMeta>>,
    pub(crate) overrides_gate: HashMap<(usize, usize), Vec<f32>>,
    pub(crate) deleted: std::collections::HashSet<(usize, usize)>,
}

impl PatchedVindex {
    /// Create a patched vindex from a base index.
    pub fn new(base: VectorIndex) -> Self {
        Self {
            base,
            patches: Vec::new(),
            overrides_meta: HashMap::new(),
            overrides_gate: HashMap::new(),
            deleted: std::collections::HashSet::new(),
        }
    }

    /// Insert a feature directly into the overlay (auto-patch mode).
    pub fn insert_feature(
        &mut self,
        layer: usize,
        feature: usize,
        gate_vec: Vec<f32>,
        meta: FeatureMeta,
    ) {
        let key = (layer, feature);
        self.overrides_meta.insert(key, Some(meta));
        self.overrides_gate.insert(key, gate_vec);
        self.deleted.remove(&key);
    }

    /// Delete a feature via the overlay.
    pub fn delete_feature(&mut self, layer: usize, feature: usize) {
        let key = (layer, feature);
        self.overrides_meta.insert(key, None);
        self.deleted.insert(key);
        self.overrides_gate.remove(&key);
    }

    /// Update feature metadata via the overlay.
    pub fn update_feature_meta(&mut self, layer: usize, feature: usize, meta: FeatureMeta) {
        let key = (layer, feature);
        self.overrides_meta.insert(key, Some(meta));
    }

    /// Check if a (layer, feature) has been overridden.
    pub fn is_overridden(&self, layer: usize, feature: usize) -> bool {
        self.overrides_meta.contains_key(&(layer, feature))
    }

    /// Access the underlying base index (readonly).
    pub fn base(&self) -> &VectorIndex {
        &self.base
    }

    /// Find a free feature slot in the base (weakest or empty).
    pub fn find_free_feature(&self, layer: usize) -> Option<usize> {
        self.base.find_free_feature(layer)
    }

    /// Apply a patch. Operations are resolved into the override maps.
    pub fn apply_patch(&mut self, patch: VindexPatch) {
        for op in &patch.operations {
            let key = op.key();
            match op {
                PatchOp::Insert { target, confidence, gate_vector_b64, down_meta, .. } => {
                    let meta = if let Some(dm) = down_meta {
                        FeatureMeta {
                            top_token: dm.top_token.clone(),
                            top_token_id: dm.top_token_id,
                            c_score: dm.c_score,
                            top_k: vec![larql_models::TopKEntry {
                                token: dm.top_token.clone(),
                                token_id: dm.top_token_id,
                                logit: dm.c_score,
                            }],
                        }
                    } else {
                        FeatureMeta {
                            top_token: target.clone(),
                            top_token_id: 0,
                            c_score: confidence.unwrap_or(0.9),
                            top_k: vec![],
                        }
                    };
                    self.overrides_meta.insert(key, Some(meta));
                    self.deleted.remove(&key);
                    if let Some(b64) = gate_vector_b64 {
                        if let Ok(vec) = decode_gate_vector(b64) {
                            self.overrides_gate.insert(key, vec);
                        }
                    }
                }
                PatchOp::Update { gate_vector_b64, down_meta, .. } => {
                    if let Some(dm) = down_meta {
                        let meta = FeatureMeta {
                            top_token: dm.top_token.clone(),
                            top_token_id: dm.top_token_id,
                            c_score: dm.c_score,
                            top_k: vec![larql_models::TopKEntry {
                                token: dm.top_token.clone(),
                                token_id: dm.top_token_id,
                                logit: dm.c_score,
                            }],
                        };
                        self.overrides_meta.insert(key, Some(meta));
                    }
                    if let Some(b64) = gate_vector_b64 {
                        if let Ok(vec) = decode_gate_vector(b64) {
                            self.overrides_gate.insert(key, vec);
                        }
                    }
                }
                PatchOp::Delete { .. } => {
                    self.overrides_meta.insert(key, None);
                    self.deleted.insert(key);
                    self.overrides_gate.remove(&key);
                }
            }
        }
        self.patches.push(patch);
    }

    /// Remove the last applied patch and rebuild overrides.
    pub fn remove_patch(&mut self, index: usize) {
        if index < self.patches.len() {
            self.patches.remove(index);
            self.rebuild_overrides();
        }
    }

    /// Rebuild override maps from scratch (after removing a patch).
    fn rebuild_overrides(&mut self) {
        self.overrides_meta.clear();
        self.overrides_gate.clear();
        self.deleted.clear();
        let patches: Vec<VindexPatch> = self.patches.drain(..).collect();
        for patch in patches {
            self.apply_patch(patch);
        }
    }

    /// Look up feature metadata, checking overrides first.
    pub fn feature_meta(&self, layer: usize, feature: usize) -> Option<FeatureMeta> {
        let key = (layer, feature);
        if let Some(override_meta) = self.overrides_meta.get(&key) {
            return override_meta.clone();
        }
        if self.deleted.contains(&key) {
            return None;
        }
        self.base.feature_meta(layer, feature)
    }

    /// Gate KNN with patched vectors.
    /// For features with overridden gate vectors, uses the patch vector.
    /// For deleted features, excludes them from results.
    pub fn gate_knn(&self, layer: usize, residual: &Array1<f32>, top_k: usize) -> Vec<(usize, f32)> {
        let mut hits = self.base.gate_knn(layer, residual, top_k * 2); // oversample

        // Apply gate vector overrides
        for (&(l, f), gate_vec) in &self.overrides_gate {
            if l != layer { continue; }
            let score: f32 = gate_vec.iter()
                .zip(residual.iter())
                .map(|(a, b)| a * b)
                .sum();
            // Update or insert
            if let Some(hit) = hits.iter_mut().find(|(feat, _)| *feat == f) {
                hit.1 = score;
            } else {
                hits.push((f, score));
            }
        }

        // Remove deleted features
        hits.retain(|(f, _)| !self.deleted.contains(&(layer, *f)));

        // Re-sort and truncate
        hits.sort_unstable_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        hits.truncate(top_k);
        hits
    }

    /// Walk with patch overrides.
    pub fn walk(&self, residual: &Array1<f32>, layers: &[usize], top_k: usize) -> WalkTrace {
        let mut trace_layers = Vec::with_capacity(layers.len());
        for &layer in layers {
            let hits = self.gate_knn(layer, residual, top_k);
            let walk_hits: Vec<WalkHit> = hits
                .into_iter()
                .filter_map(|(feature, gate_score)| {
                    let meta = self.feature_meta(layer, feature)?.clone();
                    Some(WalkHit { layer, feature, gate_score, meta })
                })
                .collect();
            trace_layers.push((layer, walk_hits));
        }
        WalkTrace { layers: trace_layers }
    }

    /// Flatten all patches into the base, producing a new clean VectorIndex (heap mode).
    pub fn bake_down(&self) -> VectorIndex {
        let mut new_gate = Vec::new();
        let mut new_meta = Vec::new();

        for layer in 0..self.base.num_layers {
            // Get base gate vectors (from heap or mmap)
            let base_gate = if let Some(g) = self.base.gate_vectors_at(layer) {
                Some(g.clone())
            } else if let Some(ref mmap) = self.base.gate_mmap_bytes {
                // Mmap mode — decode this layer's slice to an Array2
                self.base.gate_mmap_slices.get(layer).and_then(|slice| {
                    if slice.num_features == 0 { return None; }
                    let bpf = crate::config::dtype::bytes_per_float(self.base.gate_mmap_dtype);
                    let byte_offset = slice.float_offset * bpf;
                    let byte_count = slice.num_features * self.base.hidden_size * bpf;
                    let byte_end = byte_offset + byte_count;
                    if byte_end > mmap.len() { return None; }
                    let floats = crate::config::dtype::decode_floats(
                        &mmap[byte_offset..byte_end], self.base.gate_mmap_dtype
                    );
                    ndarray::Array2::from_shape_vec(
                        (slice.num_features, self.base.hidden_size), floats
                    ).ok()
                })
            } else {
                None
            };

            let gate = base_gate.map(|mut g| {
                // Apply gate vector overrides
                for (&(l, f), vec) in &self.overrides_gate {
                    if l != layer { continue; }
                    if f < g.shape()[0] && vec.len() == g.shape()[1] {
                        for (j, val) in vec.iter().enumerate() {
                            g[[f, j]] = *val;
                        }
                    }
                }
                g
            });
            new_gate.push(gate);

            // Build metadata from heap or mmap
            let num_features = self.base.num_features(layer);
            let mut new_metas: Vec<Option<FeatureMeta>> = if let Some(heap) = self.base.down_meta_at(layer) {
                heap.to_vec()
            } else if num_features > 0 {
                // Mmap: read each feature on demand
                (0..num_features).map(|f| self.base.feature_meta(layer, f)).collect()
            } else {
                Vec::new()
            };

            // Apply meta overrides
            for (&(l, f), override_meta) in &self.overrides_meta {
                if l != layer { continue; }
                while new_metas.len() <= f { new_metas.push(None); }
                new_metas[f] = override_meta.clone();
            }
            // Apply deletes
            for &(l, f) in &self.deleted {
                if l == layer && f < new_metas.len() { new_metas[f] = None; }
            }

            new_meta.push(if new_metas.is_empty() { None } else { Some(new_metas) });
        }

        VectorIndex::new(new_gate, new_meta, self.base.num_layers, self.base.hidden_size)
    }

    /// Number of active patches.
    pub fn num_patches(&self) -> usize {
        self.patches.len()
    }

    /// Total override count.
    pub fn num_overrides(&self) -> usize {
        self.overrides_meta.len()
    }

    // ── Forwarding methods to base (for compatibility) ──

    /// Layers that have gate vectors loaded (delegates to base).
    pub fn loaded_layers(&self) -> Vec<usize> {
        self.base.loaded_layers()
    }

    /// Number of features at a layer (delegates to base).
    pub fn num_features(&self, layer: usize) -> usize {
        self.base.num_features(layer)
    }

    /// Access down metadata for a layer (base only — does not include overrides).
    /// For override-aware lookups, use `feature_meta()`.
    pub fn down_meta_at(&self, layer: usize) -> Option<&[Option<FeatureMeta>]> {
        self.base.down_meta_at(layer)
    }

    /// Access gate vectors matrix for a layer (base only).
    pub fn gate_vectors_at(&self, layer: usize) -> Option<&ndarray::Array2<f32>> {
        self.base.gate_vectors_at(layer)
    }

    /// Number of layers (delegates to base).
    pub fn num_layers(&self) -> usize {
        self.base.num_layers
    }

    /// Hidden size (delegates to base).
    pub fn hidden_size(&self) -> usize {
        self.base.hidden_size
    }
}

impl GateIndex for PatchedVindex {
    fn gate_knn(&self, layer: usize, residual: &Array1<f32>, top_k: usize) -> Vec<(usize, f32)> {
        self.gate_knn(layer, residual, top_k)
    }

    fn feature_meta(&self, layer: usize, feature: usize) -> Option<FeatureMeta> {
        self.feature_meta(layer, feature)
    }

    fn num_features(&self, layer: usize) -> usize {
        self.num_features(layer)
    }
}
