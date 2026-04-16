//! Gate KNN search — brute-force, batched, and HNSW.
//!
//! All gate KNN methods for VectorIndex: single-query, batched, expert-scoped,
//! score computation, HNSW integration, and top-K selection.

use ndarray::{Array1, Array2, ArrayView2};
use larql_compute::ComputeBackend;

use super::core::VectorIndex;
use super::types::*;

/// Matrix-vector multiply: view[N, hidden] × vec[hidden] → scores[N].
/// All compute goes through larql-compute.
fn gemv(view: &ArrayView2<f32>, vec: &Array1<f32>) -> Array1<f32> {
    let hidden = vec.len();
    let x = vec.view().into_shape_with_order((1, hidden)).unwrap();
    let cpu = larql_compute::default_backend();
    // x[1, hidden] @ view[N, hidden]^T → [1, N]
    let result = cpu.matmul_transb(x, *view);
    Array1::from_vec(result.into_raw_vec_and_offset().0)
}

/// Gate scores batch: gate[N, hidden] × x[seq, hidden]^T → [N, seq].
/// Equivalent to original gate.dot(&x.t()).
fn gate_matmul(gate: &ArrayView2<f32>, x: &ArrayView2<f32>) -> Array2<f32> {
    let cpu = larql_compute::default_backend();
    // gate[N, hidden] @ x[seq, hidden]^T = matmul_transb(gate, x) → [N, seq]
    cpu.matmul_transb(*gate, *x)
}

/// Resolved gate matrix data — owned f32 with feature count.
struct GateData {
    data: Vec<f32>,
    num_features: usize,
}

impl GateData {
    fn view(&self, hidden_size: usize) -> ArrayView2<'_, f32> {
        ArrayView2::from_shape((self.num_features, hidden_size), &self.data).unwrap()
    }
}

/// Gate KNN methods for VectorIndex.
impl VectorIndex {
    /// Resolve the gate matrix for a layer as contiguous f32.
    /// Handles all storage paths: warmed → heap → mmap f32 → mmap f16.
    /// Returns owned data (zero-copy from mmap via to_vec on the hot path).
    fn resolve_gate(&self, layer: usize) -> Option<GateData> {
        // 1. Warmed cache
        {
            let warmed = self.warmed_gates.read().unwrap();
            if let Some(Some(ref data)) = warmed.get(layer) {
                let nf = self.gate_mmap_slices.get(layer).map(|s| s.num_features).unwrap_or(0);
                if nf > 0 {
                    return Some(GateData { data: data.clone(), num_features: nf });
                }
            }
        }

        // 2. Heap
        if let Some(Some(ref matrix)) = self.gate_vectors.get(layer) {
            return Some(GateData {
                data: matrix.as_slice().unwrap().to_vec(),
                num_features: matrix.shape()[0],
            });
        }

        // 3. Mmap
        if let Some(ref mmap) = self.gate_mmap_bytes {
            if let Some(slice) = self.gate_mmap_slices.get(layer) {
                if slice.num_features == 0 { return None; }
                let bpf = crate::config::dtype::bytes_per_float(self.gate_mmap_dtype);
                let byte_offset = slice.float_offset * bpf;
                let byte_count = slice.num_features * self.hidden_size * bpf;
                let byte_end = byte_offset + byte_count;
                if byte_end > mmap.len() { return None; }

                let data = match self.gate_mmap_dtype {
                    crate::config::dtype::StorageDtype::F32 => {
                        let float_count = slice.num_features * self.hidden_size;
                        unsafe {
                            let ptr = mmap[byte_offset..byte_end].as_ptr() as *const f32;
                            std::slice::from_raw_parts(ptr, float_count).to_vec()
                        }
                    }
                    crate::config::dtype::StorageDtype::F16 => {
                        let mut cache = self.f16_decode_cache.lock().unwrap();
                        if cache.len() <= layer { cache.resize(layer + 1, None); }
                        if cache[layer].is_none() {
                            let raw = &mmap[byte_offset..byte_end];
                            cache[layer] = Some(larql_models::quant::half::decode_f16(raw));
                        }
                        cache[layer].as_ref().unwrap().clone()
                    }
                };
                return Some(GateData { data, num_features: slice.num_features });
            }
        }

        None
    }

    /// Gate KNN: find the top-K features at a layer whose gate vectors have
    /// the highest dot product with the input residual. Uses BLAS matmul.
    ///
    /// In mmap mode, slices directly from the mmap'd file — zero heap allocation.
    /// Returns (feature_index, dot_product) sorted by absolute magnitude descending.
    pub fn gate_knn(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        // HNSW path
        if self.hnsw_enabled.load(std::sync::atomic::Ordering::Relaxed) {
            if let Some(results) = self.gate_knn_hnsw(layer, residual, top_k) {
                return results;
            }
        }

        // Fast path: f32 mmap zero-copy (no allocation, no clone)
        if let Some(scores) = self.gate_knn_mmap_fast(layer, residual) {
            return Self::top_k_from_scores(&scores, top_k);
        }

        // Fallback: resolve_gate (copies data for heap/f16 paths)
        let gate = match self.resolve_gate(layer) {
            Some(g) => g,
            None => return vec![],
        };
        let view = gate.view(self.hidden_size);
        let scores = gemv(&view, residual);
        Self::top_k_from_scores(&scores, top_k)
    }

    /// Zero-copy gate KNN for f32 mmap — no allocation, no clone.
    /// Returns None if not on the f32 mmap path (falls back to resolve_gate).
    fn gate_knn_mmap_fast(&self, layer: usize, residual: &Array1<f32>) -> Option<Array1<f32>> {
        // Warmed cache (RwLock read — lock-free when no writers)
        {
            let warmed = self.warmed_gates.read().unwrap();
            if let Some(Some(ref data)) = warmed.get(layer) {
                let nf = self.gate_mmap_slices.get(layer).map(|s| s.num_features).unwrap_or(0);
                if nf > 0 {
                    let view = ArrayView2::from_shape((nf, self.hidden_size), data.as_slice()).unwrap();
                    return Some(gemv(&view, residual));
                }
            }
        }

        // f32 mmap zero-copy
        if self.gate_mmap_dtype == crate::config::dtype::StorageDtype::F32 {
            if let Some(ref mmap) = self.gate_mmap_bytes {
                if let Some(slice) = self.gate_mmap_slices.get(layer) {
                    if slice.num_features == 0 { return None; }
                    let bpf = 4;
                    let byte_offset = slice.float_offset * bpf;
                    let byte_end = byte_offset + slice.num_features * self.hidden_size * bpf;
                    if byte_end > mmap.len() { return None; }
                    let data = unsafe {
                        let ptr = mmap[byte_offset..byte_end].as_ptr() as *const f32;
                        std::slice::from_raw_parts(ptr, slice.num_features * self.hidden_size)
                    };
                    let view = ArrayView2::from_shape((slice.num_features, self.hidden_size), data).unwrap();
                    return Some(gemv(&view, residual));
                }
            }
        }

        None // Not on fast path — caller will use resolve_gate
    }

    /// Per-feature gate walk: score each feature with an individual dot product.
    /// No matrix multiplication. Iterates gate vectors from mmap and computes
    /// dot products one feature at a time. Returns exact top-K.
    pub fn gate_walk(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        top_k: usize,
    ) -> Option<Vec<(usize, f32)>> {
        let num_features = self.num_features(layer);
        if num_features == 0 { return None; }

        // Get gate data as contiguous f32 (from mmap or warmed cache)
        let gate_data: &[f32];
        let _owned: Vec<f32>;

        // Try zero-copy f32 mmap first
        let mmap_slice = if self.gate_mmap_dtype == crate::config::dtype::StorageDtype::F32 {
            self.gate_mmap_bytes.as_ref().and_then(|mmap| {
                let slice = self.gate_mmap_slices.get(layer)?;
                if slice.num_features == 0 { return None; }
                let byte_offset = slice.float_offset * 4;
                let byte_end = byte_offset + slice.num_features * self.hidden_size * 4;
                if byte_end > mmap.len() { return None; }
                Some(unsafe {
                    std::slice::from_raw_parts(
                        mmap[byte_offset..byte_end].as_ptr() as *const f32,
                        slice.num_features * self.hidden_size,
                    )
                })
            })
        } else {
            None
        };

        if let Some(data) = mmap_slice {
            gate_data = data;
        } else {
            // Fallback: resolve gate (may clone)
            let gate = self.resolve_gate(layer)?;
            _owned = gate.data;
            gate_data = &_owned;
        }

        let hidden = self.hidden_size;

        // Single BLAS gemv: gate[N, hidden] × residual[hidden] → scores[N].
        let gate_view = ArrayView2::from_shape((num_features, hidden), gate_data).unwrap();
        let scores = gemv(&gate_view, residual);

        // Top-K selection
        let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        let k = top_k.min(indexed.len());
        if k > 0 && k < indexed.len() {
            indexed.select_nth_unstable_by(k, |a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
            indexed.truncate(k);
        }
        indexed.sort_unstable_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        Some(indexed)
    }

    /// Gate KNN within a specific feature range (for MoE expert-scoped queries).
    /// Only computes dot products for features [feat_start..feat_end].
    /// Returns (global_feature_index, score) pairs.
    pub fn gate_knn_expert(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        feat_start: usize,
        feat_end: usize,
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        // If promoted to heap, use heap path
        if let Some(Some(ref matrix)) = self.gate_vectors.get(layer) {
            let end = feat_end.min(matrix.shape()[0]);
            if feat_start >= end { return vec![]; }
            let slice = matrix.slice(ndarray::s![feat_start..end, ..]);
            let scores = gemv(&slice, residual);
            let mut hits = Self::top_k_from_scores(&scores, top_k);
            for hit in &mut hits { hit.0 += feat_start; }
            return hits;
        }

        if let Some(ref mmap) = self.gate_mmap_bytes {
            if let Some(slice) = self.gate_mmap_slices.get(layer) {
                if slice.num_features == 0 || feat_start >= slice.num_features { return vec![]; }
                let end = feat_end.min(slice.num_features);
                let bpf = crate::config::dtype::bytes_per_float(self.gate_mmap_dtype);

                // Compute byte range for just this expert's features
                let layer_byte_start = slice.float_offset * bpf;
                let expert_byte_start = layer_byte_start + feat_start * self.hidden_size * bpf;
                let expert_byte_end = layer_byte_start + end * self.hidden_size * bpf;
                let n_features = end - feat_start;

                if expert_byte_end > mmap.len() { return vec![]; }

                match self.gate_mmap_dtype {
                    crate::config::dtype::StorageDtype::F32 => {
                        let data = unsafe {
                            let ptr = mmap[expert_byte_start..expert_byte_end].as_ptr() as *const f32;
                            std::slice::from_raw_parts(ptr, n_features * self.hidden_size)
                        };
                        let view = ndarray::ArrayView2::from_shape(
                            (n_features, self.hidden_size), data
                        ).unwrap();
                        let scores = gemv(&view, residual);
                        let mut hits = Self::top_k_from_scores(&scores, top_k);
                        // Offset indices to global feature space
                        for hit in &mut hits { hit.0 += feat_start; }
                        return hits;
                    }
                    crate::config::dtype::StorageDtype::F16 => {
                        let raw = &mmap[expert_byte_start..expert_byte_end];
                        let floats = larql_models::quant::half::decode_f16(raw);
                        let view = ndarray::ArrayView2::from_shape(
                            (n_features, self.hidden_size), &floats
                        ).unwrap();
                        let scores = gemv(&view, residual);
                        let mut hits = Self::top_k_from_scores(&scores, top_k);
                        for hit in &mut hits { hit.0 += feat_start; }
                        return hits;
                    }
                }
            }
        }
        // Fallback: full KNN filtered (slower)
        self.gate_knn(layer, residual, top_k * 10)
            .into_iter()
            .filter(|(f, _)| *f >= feat_start && *f < feat_end)
            .take(top_k)
            .collect()
    }

    fn top_k_from_scores(scores: &Array1<f32>, top_k: usize) -> Vec<(usize, f32)> {
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
                    let meta = self.feature_meta(layer, feature)?;
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
    /// Checks heap first (mutation overrides), then mmap (production read path).
    pub fn feature_meta(&self, layer: usize, feature: usize) -> Option<FeatureMeta> {
        // Heap path first — catches mutation overrides (INSERT/UPDATE)
        if let Some(meta) = self.down_meta
            .get(layer)
            .and_then(|v| v.as_ref())
            .and_then(|metas| metas.get(feature))
            .and_then(|m| m.clone())
        {
            return Some(meta);
        }
        // Mmap path (production — zero heap, no mutations)
        if let Some(ref dm) = self.down_meta_mmap {
            return dm.feature_meta(layer, feature);
        }
        None
    }

    /// Number of features indexed at a layer.
    pub fn num_features(&self, layer: usize) -> usize {
        // Check mmap first
        if self.gate_mmap_bytes.is_some() {
            return self.gate_mmap_slices.get(layer)
                .map(|s| s.num_features)
                .unwrap_or(0);
        }
        self.gate_vectors
            .get(layer)
            .and_then(|v| v.as_ref())
            .map(|m| m.shape()[0])
            .unwrap_or(0)
    }

    /// Total gate vectors loaded across all layers.
    pub fn total_gate_vectors(&self) -> usize {
        if self.gate_mmap_bytes.is_some() {
            return self.gate_mmap_slices.iter().map(|s| s.num_features).sum();
        }
        self.gate_vectors
            .iter()
            .filter_map(|v| v.as_ref())
            .map(|m| m.shape()[0])
            .sum()
    }

    /// Total down metadata entries loaded across all layers.
    pub fn total_down_meta(&self) -> usize {
        if let Some(ref dm) = self.down_meta_mmap {
            return dm.total_features();
        }
        self.down_meta
            .iter()
            .filter_map(|v| v.as_ref())
            .map(|metas| metas.iter().filter(|m| m.is_some()).count())
            .sum()
    }

    /// Layers that have gate vectors loaded.
    pub fn loaded_layers(&self) -> Vec<usize> {
        if self.gate_mmap_bytes.is_some() {
            return self.gate_mmap_slices.iter()
                .enumerate()
                .filter(|(_, s)| s.num_features > 0)
                .map(|(i, _)| i)
                .collect();
        }
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

    /// Access gate vectors matrix for a specific layer (heap mode only).
    /// Returns None in mmap mode — use gate_knn() directly instead.
    pub fn gate_vectors_at(&self, layer: usize) -> Option<&Array2<f32>> {
        self.gate_vectors.get(layer).and_then(|v| v.as_ref())
    }

    /// Extract a single gate vector for a feature. Works in both heap and mmap mode.
    /// Returns the raw f32 vector (hidden_size elements).
    pub fn gate_vector(&self, layer: usize, feature: usize) -> Option<Vec<f32>> {
        // Heap path
        if let Some(Some(matrix)) = self.gate_vectors.get(layer) {
            if feature < matrix.shape()[0] {
                return Some(matrix.row(feature).to_vec());
            }
            return None;
        }
        // Mmap path
        if let Some(ref mmap) = self.gate_mmap_bytes {
            if let Some(slice) = self.gate_mmap_slices.get(layer) {
                if feature >= slice.num_features { return None; }
                let bpf = crate::config::dtype::bytes_per_float(self.gate_mmap_dtype);
                let byte_offset = (slice.float_offset + feature * self.hidden_size) * bpf;
                let byte_count = self.hidden_size * bpf;
                if byte_offset + byte_count > mmap.len() { return None; }
                let raw = &mmap[byte_offset..byte_offset + byte_count];
                return Some(crate::config::dtype::decode_floats(raw, self.gate_mmap_dtype));
            }
        }
        None
    }

    /// Extract all gate vectors at a layer as flat f32 data.
    /// Returns (flat_data, num_features, hidden_size). Works in both heap and mmap mode.
    /// Use for bulk operations (SVD, PCA, numpy export).
    pub fn gate_vectors_flat(&self, layer: usize) -> Option<(Vec<f32>, usize, usize)> {
        // Heap path
        if let Some(Some(matrix)) = self.gate_vectors.get(layer) {
            let (rows, cols) = (matrix.shape()[0], matrix.shape()[1]);
            if let Some(data) = matrix.as_slice() {
                return Some((data.to_vec(), rows, cols));
            }
            // Non-contiguous — copy row by row
            let mut data = Vec::with_capacity(rows * cols);
            for r in 0..rows {
                data.extend(matrix.row(r).iter());
            }
            return Some((data, rows, cols));
        }
        // Mmap path
        if let Some(ref mmap) = self.gate_mmap_bytes {
            if let Some(slice) = self.gate_mmap_slices.get(layer) {
                if slice.num_features == 0 { return None; }
                let bpf = crate::config::dtype::bytes_per_float(self.gate_mmap_dtype);
                let byte_offset = slice.float_offset * bpf;
                let byte_count = slice.num_features * self.hidden_size * bpf;
                if byte_offset + byte_count > mmap.len() { return None; }
                let raw = &mmap[byte_offset..byte_offset + byte_count];
                let data = crate::config::dtype::decode_floats(raw, self.gate_mmap_dtype);
                return Some((data, slice.num_features, self.hidden_size));
            }
        }
        None
    }

    /// Batched gate KNN: compute scores for ALL sequence positions in one BLAS gemm.
    ///
    /// Input: x is [seq_len, hidden]. Computes gate_vectors @ x^T = [features, seq_len].
    /// Returns the union of per-position top-K feature indices (sorted).
    /// One gemm replaces seq_len separate gemv calls.
    pub fn gate_knn_batch(
        &self,
        layer: usize,
        x: &Array2<f32>,
        top_k: usize,
    ) -> Vec<usize> {
        let seq_len = x.shape()[0];
        if seq_len == 0 { return vec![]; }

        // Fast path: zero-copy f32 mmap/warmed
        let scores_2d = if let Some(s) = self.gate_scores_2d_fast(layer, x) {
            s
        } else if let Some(gate) = self.resolve_gate(layer) {
            gate_matmul(&gate.view(self.hidden_size), &x.view())
        } else {
            return vec![];
        };

        // scores_2d is [num_features, seq_len]
        // For each position, take top-K features and union them
        let num_features = scores_2d.shape()[0];
        let mut feature_set = std::collections::BTreeSet::new();

        for s in 0..seq_len {
            let col = scores_2d.column(s);
            let mut indexed: Vec<(usize, f32)> = col.iter().copied().enumerate().collect();
            let k = top_k.min(num_features);
            if k > 0 && k < indexed.len() {
                indexed.select_nth_unstable_by(k, |a, b| {
                    b.1.abs().partial_cmp(&a.1.abs()).unwrap()
                });
                indexed.truncate(k);
            }
            feature_set.extend(indexed.iter().map(|(idx, _)| *idx));
        }

        feature_set.into_iter().collect()
    }

    // Feature store methods (load_down/up_features, down/up_layer_matrix, warmup)
    // are in feature_store.rs

    /// Compute gate scores for all features × all positions in one BLAS gemm.
    /// Returns [seq_len, intermediate] matrix = x @ gate_vectors^T.
    /// These scores are the gate projections — the same as x @ W_gate.T.
    pub fn gate_scores_batch(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> Option<Array2<f32>> {
        if x.shape()[0] == 0 { return None; }
        // Fast path first, then fallback
        let scores_2d = if let Some(s) = self.gate_scores_2d_fast(layer, x) {
            s
        } else {
            let gate = self.resolve_gate(layer)?;
            gate_matmul(&gate.view(self.hidden_size), &x.view())
        };
        Some(scores_2d.t().to_owned())
    }

    /// Zero-copy batch gate scores for f32 mmap/warmed — returns [features, seq].
    fn gate_scores_2d_fast(&self, layer: usize, x: &Array2<f32>) -> Option<Array2<f32>> {
        // Warmed cache
        {
            let warmed = self.warmed_gates.read().unwrap();
            if let Some(Some(ref data)) = warmed.get(layer) {
                let nf = self.gate_mmap_slices.get(layer).map(|s| s.num_features).unwrap_or(0);
                if nf > 0 {
                    let view = ArrayView2::from_shape((nf, self.hidden_size), data.as_slice()).unwrap();
                    return Some(gate_matmul(&view, &x.view()));
                }
            }
        }
        // f32 mmap
        if self.gate_mmap_dtype == crate::config::dtype::StorageDtype::F32 {
            if let Some(ref mmap) = self.gate_mmap_bytes {
                if let Some(slice) = self.gate_mmap_slices.get(layer) {
                    if slice.num_features == 0 { return None; }
                    let byte_offset = slice.float_offset * 4;
                    let byte_end = byte_offset + slice.num_features * self.hidden_size * 4;
                    if byte_end > mmap.len() { return None; }
                    let data = unsafe {
                        let ptr = mmap[byte_offset..byte_end].as_ptr() as *const f32;
                        std::slice::from_raw_parts(ptr, slice.num_features * self.hidden_size)
                    };
                    let view = ArrayView2::from_shape((slice.num_features, self.hidden_size), data).unwrap();
                    return Some(gate_matmul(&view, &x.view()));
                }
            }
        }
        None
    }

    /// Enable HNSW search. Indexes are built lazily on first query per layer.
    ///
    /// `ef_search`: beam width for search (50-200). Higher = better recall, slower.
    pub fn enable_hnsw(&self, ef_search: usize) {
        self.hnsw_enabled.store(true, std::sync::atomic::Ordering::Relaxed);
        self.hnsw_ef_search.store(ef_search, std::sync::atomic::Ordering::Relaxed);
    }

    /// Disable HNSW, revert to brute-force matmul.
    pub fn disable_hnsw(&self) {
        self.hnsw_enabled.store(false, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether HNSW is currently enabled.
    pub fn is_hnsw_enabled(&self) -> bool {
        self.hnsw_enabled.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get the gate vector matrix for a layer as owned contiguous f32.
    /// Used by HNSW build which needs owned data.
    fn gate_matrix_f32(&self, layer: usize) -> Option<(Vec<f32>, usize)> {
        let gate = self.resolve_gate(layer)?;
        Some((gate.data, gate.num_features))
    }

    /// Get or build the HNSW index for a layer (lazy).
    fn get_or_build_hnsw(&self, layer: usize) -> bool {
        let mut cache = self.hnsw_cache.lock().unwrap();
        if cache.len() <= layer { cache.resize_with(layer + 1, || None); }
        if cache[layer].is_some() { return true; }

        // Build from gate vectors
        if let Some((data, num_features)) = self.gate_matrix_f32(layer) {
            let view = ArrayView2::from_shape(
                (num_features, self.hidden_size), &data
            ).unwrap();
            let hnsw = super::hnsw::HnswLayer::build(&view, 8, 32);
            cache[layer] = Some(hnsw);
            true
        } else {
            false
        }
    }

    /// Gate KNN via HNSW: graph search instead of brute-force matmul.
    fn gate_knn_hnsw(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        top_k: usize,
    ) -> Option<Vec<(usize, f32)>> {
        if !self.get_or_build_hnsw(layer) { return None; }

        let ef = self.hnsw_ef_search.load(std::sync::atomic::Ordering::Relaxed);

        // We need both the HNSW index and the vectors for search
        let cache = self.hnsw_cache.lock().unwrap();
        let hnsw = cache[layer].as_ref()?;

        // Get gate matrix for dot product computation during search
        let (data, num_features) = self.gate_matrix_f32(layer)?;
        let view = ArrayView2::from_shape(
            (num_features, self.hidden_size), &data
        ).unwrap();

        let results = hnsw.search(&view, residual, top_k, ef);
        Some(results)
    }

    /// Adaptive gate KNN — automatically picks the fastest path per layer.
    ///
    /// Dispatch order:
    /// 1. Pinned Q4 → backend.q4_matvec (pre-loaded, no page faults)
    /// 2. Mmap Q4 → backend.q4_matvec (paged on demand)
    /// 3. f32 mmap/heap → BLAS brute-force (fallback)
    ///
    /// The residency manager tracks which layers are pinned.
    /// More memory budget → more pinned layers → faster walk.
    pub fn gate_knn_adaptive(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        top_k: usize,
        residency: &mut super::residency::ResidencyManager,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Vec<(usize, f32)> {
        residency.record_access(layer);

        // 1. Pinned Q4 (fastest — data already in RAM)
        if let Some(q4_data) = residency.pinned_q4(layer) {
            if backend.has_q4() {
                let x = residual.as_slice().unwrap();
                let (q8_x, q8_scales) = larql_compute::cpu::q4::quantize_to_q8(x);
                let num_features = self.num_features(layer);
                if let Some(scores_vec) = backend.q4_matvec(
                    q4_data, &q8_x, &q8_scales, num_features, self.hidden_size,
                ) {
                    return Self::top_k_from_scores(&Array1::from_vec(scores_vec), top_k);
                }
            }
        }

        // 2. Mmap Q4 (Q4 file loaded but not pinned — OS pages on demand)
        if let Some(hits) = self.gate_knn_q4(layer, residual, top_k, backend) {
            return hits;
        }

        // 3. f32 brute-force (fallback)
        self.gate_knn(layer, residual, top_k)
    }

    /// Gate KNN via Q4 matvec — scored by a ComputeBackend.
    ///
    /// The vindex provides the raw Q4 data. The backend scores it.
    /// Works with any backend: CPU C kernel, Metal GPU, CUDA, WASM.
    ///
    /// Returns None if Q4 gate data isn't loaded or backend doesn't support Q4.
    pub fn gate_knn_q4(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        top_k: usize,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Option<Vec<(usize, f32)>> {
        if !backend.has_q4() { return None; }
        let q4_data = self.gate_q4_data(layer)?;
        let slice = self.gate_q4_slices.get(layer)?;
        if slice.num_features == 0 { return None; }

        let (q8_x, q8_scales) = larql_compute::cpu::q4::quantize_to_q8(residual.as_slice().unwrap());
        let scores_vec = backend.q4_matvec(
            q4_data, &q8_x, &q8_scales,
            slice.num_features, self.hidden_size,
        )?;

        let scores = Array1::from_vec(scores_vec);
        Some(Self::top_k_from_scores(&scores, top_k))
    }

    /// Number of features at a layer (works in both heap and mmap mode).
    pub fn num_features_at(&self, layer: usize) -> usize {
        if self.gate_mmap_bytes.is_some() {
            self.gate_mmap_slices.get(layer).map(|s| s.num_features).unwrap_or(0)
        } else {
            self.num_features(layer)
        }
    }

    /// Pre-decode f16 gate vectors to f32 for lock-free access.
    /// For f32 vindexes this is a no-op — the mmap path is already zero-copy.
    pub fn warmup(&self) {
        if self.gate_mmap_dtype == crate::config::dtype::StorageDtype::F32 { return; }

        let Some(ref mmap) = self.gate_mmap_bytes else { return; };
        let mut warmed = self.warmed_gates.write().unwrap();
        if warmed.len() < self.num_layers {
            warmed.resize_with(self.num_layers, || None);
        }
        for layer in 0..self.num_layers {
            if warmed[layer].is_some() { continue; }
            if let Some(slice) = self.gate_mmap_slices.get(layer) {
                if slice.num_features == 0 { continue; }
                let bpf = crate::config::dtype::bytes_per_float(self.gate_mmap_dtype);
                let byte_offset = slice.float_offset * bpf;
                let byte_count = slice.num_features * self.hidden_size * bpf;
                let byte_end = byte_offset + byte_count;
                if byte_end > mmap.len() { continue; }
                let raw = &mmap[byte_offset..byte_end];
                warmed[layer] = Some(larql_models::quant::half::decode_f16(raw));
            }
        }
    }
}
