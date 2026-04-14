//! Walk FFN data — mmap'd feature-major down and up projection vectors.
//!
//! Manages down_features.bin and up_features.bin — [intermediate, hidden] per layer,
//! f32 files where each feature's vector is contiguous for zero-copy BLAS access.

use std::sync::Arc;

use crate::error::VindexError;

use super::core::VectorIndex;

use crate::mmap_util::mmap_optimized;

/// Feature store methods for VectorIndex.
impl VectorIndex {
    /// Load feature-major down vectors from down_features.bin.
    pub fn load_down_features(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join("down_features.bin");
        if !path.exists() {
            return Err(VindexError::Parse(
                "down_features.bin not found. Run: cargo run --release -p larql-vindex --example build_down_features -- <vindex>".into()
            ));
        }
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { mmap_optimized(&file)? };
        self.down_features_mmap = Some(Arc::new(mmap));
        Ok(())
    }

    /// Whether feature-major down vectors are loaded.
    pub fn has_down_features(&self) -> bool {
        self.down_features_mmap.is_some()
    }

    /// Get a feature's contiguous down vector from the mmap'd feature-major file.
    /// Returns `[hidden_size]` f32 slice — zero-copy from mmap.
    pub fn down_feature_vector(&self, layer: usize, feature: usize) -> Option<&[f32]> {
        let mmap = self.down_features_mmap.as_ref()?;
        let intermediate = self.num_features(layer);
        if intermediate == 0 || feature >= intermediate { return None; }

        let layer_floats = intermediate * self.hidden_size;
        let layer_offset = layer * layer_floats * 4;
        let feature_offset = feature * self.hidden_size * 4;
        let start = layer_offset + feature_offset;
        let end = start + self.hidden_size * 4;

        if end > mmap.len() { return None; }

        let data = unsafe {
            let ptr = mmap[start..end].as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, self.hidden_size)
        };
        Some(data)
    }

    /// Get the full down matrix for a layer: [intermediate, hidden] zero-copy view.
    pub fn down_layer_matrix(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        let mmap = self.down_features_mmap.as_ref()?;
        let intermediate = self.num_features(layer);
        if intermediate == 0 { return None; }

        let floats_per_layer = intermediate * self.hidden_size;
        let bytes_per_layer = floats_per_layer * 4;
        let start = layer * bytes_per_layer;
        let end = start + bytes_per_layer;
        if end > mmap.len() { return None; }

        let data = unsafe {
            let ptr = mmap[start..end].as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, floats_per_layer)
        };
        ndarray::ArrayView2::from_shape((intermediate, self.hidden_size), data).ok()
    }

    /// Load feature-major up vectors from up_features.bin.
    pub fn load_up_features(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join("up_features.bin");
        if !path.exists() {
            return Err(VindexError::Parse(
                "up_features.bin not found. Run: cargo run --release -p larql-vindex --example build_up_features -- <vindex>".into()
            ));
        }
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { mmap_optimized(&file)? };
        self.up_features_mmap = Some(Arc::new(mmap));
        Ok(())
    }

    /// Get the full up matrix for a layer: [intermediate, hidden] zero-copy view.
    pub fn up_layer_matrix(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        let mmap = self.up_features_mmap.as_ref()?;
        let intermediate = self.num_features(layer);
        if intermediate == 0 { return None; }
        let floats_per_layer = intermediate * self.hidden_size;
        let bytes_per_layer = floats_per_layer * 4;
        let start = layer * bytes_per_layer;
        let end = start + bytes_per_layer;
        if end > mmap.len() { return None; }
        let data = unsafe {
            let ptr = mmap[start..end].as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, floats_per_layer)
        };
        ndarray::ArrayView2::from_shape((intermediate, self.hidden_size), data).ok()
    }

    /// Whether both up and down feature-major mmaps are loaded.
    pub fn has_full_mmap_ffn(&self) -> bool {
        self.down_features_mmap.is_some() && self.up_features_mmap.is_some()
    }

    // ── Interleaved FFN data: gate+up+down packed per layer ──

    /// Load interleaved FFN data: [gate|up|down] per layer in one contiguous file.
    /// Eliminates TLB thrash from 3 separate mmap files.
    pub fn load_interleaved(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join("interleaved.bin");
        if !path.exists() {
            return Err(VindexError::Parse(
                "interleaved.bin not found. Run: cargo run --release -p larql-vindex --example build_interleaved -- <vindex>".into()
            ));
        }
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { mmap_optimized(&file)? };
        self.interleaved_mmap = Some(Arc::new(mmap));
        Ok(())
    }

    /// Whether interleaved FFN data is loaded.
    pub fn has_interleaved(&self) -> bool {
        self.interleaved_mmap.is_some()
    }

    /// Get gate matrix for a layer from the interleaved file: [intermediate, hidden].
    pub fn interleaved_gate(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        let mmap = self.interleaved_mmap.as_ref()?;
        let intermediate = self.num_features(layer);
        if intermediate == 0 { return None; }
        let matrix_floats = intermediate * self.hidden_size;
        let matrix_bytes = matrix_floats * 4;
        let layer_bytes = matrix_bytes * 3; // gate + up + down
        let start = layer * layer_bytes; // gate is first
        let end = start + matrix_bytes;
        if end > mmap.len() { return None; }
        let data = unsafe {
            let ptr = mmap[start..end].as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, matrix_floats)
        };
        ndarray::ArrayView2::from_shape((intermediate, self.hidden_size), data).ok()
    }

    /// Get up matrix for a layer from the interleaved file: [intermediate, hidden].
    pub fn interleaved_up(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        let mmap = self.interleaved_mmap.as_ref()?;
        let intermediate = self.num_features(layer);
        if intermediate == 0 { return None; }
        let matrix_floats = intermediate * self.hidden_size;
        let matrix_bytes = matrix_floats * 4;
        let layer_bytes = matrix_bytes * 3;
        let start = layer * layer_bytes + matrix_bytes; // up is second
        let end = start + matrix_bytes;
        if end > mmap.len() { return None; }
        let data = unsafe {
            let ptr = mmap[start..end].as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, matrix_floats)
        };
        ndarray::ArrayView2::from_shape((intermediate, self.hidden_size), data).ok()
    }

    /// Get down matrix for a layer from the interleaved file: [intermediate, hidden].
    pub fn interleaved_down(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        let mmap = self.interleaved_mmap.as_ref()?;
        let intermediate = self.num_features(layer);
        if intermediate == 0 { return None; }
        let matrix_floats = intermediate * self.hidden_size;
        let matrix_bytes = matrix_floats * 4;
        let layer_bytes = matrix_bytes * 3;
        let start = layer * layer_bytes + matrix_bytes * 2; // down is third
        let end = start + matrix_bytes;
        if end > mmap.len() { return None; }
        let data = unsafe {
            let ptr = mmap[start..end].as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, matrix_floats)
        };
        ndarray::ArrayView2::from_shape((intermediate, self.hidden_size), data).ok()
    }

    /// Prefetch next layer's interleaved data into page cache.
    pub fn prefetch_interleaved_layer(&self, layer: usize) {
        #[cfg(unix)]
        if let Some(ref mmap) = self.interleaved_mmap {
            let intermediate = self.num_features(layer);
            if intermediate == 0 { return; }
            let matrix_bytes = intermediate * self.hidden_size * 4;
            let layer_bytes = matrix_bytes * 3;
            let start = layer * layer_bytes;
            let end = (start + layer_bytes).min(mmap.len());
            if start >= mmap.len() { return; }
            unsafe {
                let ptr = mmap[start..].as_ptr() as *mut libc::c_void;
                libc::madvise(ptr, end - start, libc::MADV_WILLNEED);
            }
        }
    }

    // ── Q4 interleaved: quantized gate+up+down per layer ──

    /// Load Q4_0 interleaved FFN data.
    pub fn load_interleaved_q4(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join("interleaved_q4.bin");
        if !path.exists() {
            return Err(VindexError::Parse("interleaved_q4.bin not found".into()));
        }
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { mmap_optimized(&file)? };
        self.interleaved_q4_mmap = Some(Arc::new(mmap));
        Ok(())
    }

    pub fn has_interleaved_q4(&self) -> bool {
        self.interleaved_q4_mmap.is_some()
    }

    /// Load Q4_K/Q6_K interleaved FFN data (Ollama-compatible, matches attn format).
    pub fn load_interleaved_q4k(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join("interleaved_q4k.bin");
        if !path.exists() {
            return Err(VindexError::Parse("interleaved_q4k.bin not found".into()));
        }
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { mmap_optimized(&file)? };
        self.interleaved_q4k_mmap = Some(Arc::new(mmap));
        Ok(())
    }

    pub fn has_interleaved_q4k(&self) -> bool {
        self.interleaved_q4k_mmap.is_some()
    }

    /// Dequantize one matrix from Q4 interleaved file → f32 Array2.
    /// component: 0=gate, 1=up, 2=down
    fn dequant_q4_matrix(&self, layer: usize, component: usize) -> Option<ndarray::Array2<f32>> {
        let mmap = self.interleaved_q4_mmap.as_ref()?;
        let intermediate = self.num_features(layer);
        if intermediate == 0 { return None; }

        let floats_per_matrix = intermediate * self.hidden_size;
        let q4_bytes_per_matrix = floats_per_matrix / 32 * 18; // Q4_0: 18 bytes per 32 elements
        let q4_bytes_per_layer = q4_bytes_per_matrix * 3;

        let start = layer * q4_bytes_per_layer + component * q4_bytes_per_matrix;
        let end = start + q4_bytes_per_matrix;
        if end > mmap.len() { return None; }

        let q4_data = &mmap[start..end];
        let floats = larql_models::quant::ggml::dequantize_q4_0(q4_data, floats_per_matrix).ok()?;
        ndarray::Array2::from_shape_vec((intermediate, self.hidden_size), floats).ok()
    }

    /// Get gate matrix from Q4 interleaved file, dequantized to f32.
    pub fn interleaved_q4_gate(&self, layer: usize) -> Option<ndarray::Array2<f32>> {
        self.dequant_q4_matrix(layer, 0)
    }

    /// Get up matrix from Q4 interleaved file, dequantized to f32.
    pub fn interleaved_q4_up(&self, layer: usize) -> Option<ndarray::Array2<f32>> {
        self.dequant_q4_matrix(layer, 1)
    }

    /// Get down matrix from Q4 interleaved file, dequantized to f32.
    pub fn interleaved_q4_down(&self, layer: usize) -> Option<ndarray::Array2<f32>> {
        self.dequant_q4_matrix(layer, 2)
    }

    /// Prefetch next layer's Q4 data.
    pub fn prefetch_interleaved_q4_layer(&self, layer: usize) {
        #[cfg(unix)]
        if let Some(ref mmap) = self.interleaved_q4_mmap {
            let intermediate = self.num_features(layer);
            if intermediate == 0 { return; }
            let q4_bytes_per_matrix = intermediate * self.hidden_size / 32 * 18;
            let q4_bytes_per_layer = q4_bytes_per_matrix * 3;
            let start = layer * q4_bytes_per_layer;
            let end = (start + q4_bytes_per_layer).min(mmap.len());
            if start >= mmap.len() { return; }
            unsafe {
                let ptr = mmap[start..].as_ptr() as *mut libc::c_void;
                libc::madvise(ptr, end - start, libc::MADV_WILLNEED);
            }
        }
    }

    // warmup() is in gate.rs (it's a gate cache operation)

    // ── Q4 gate vectors for fast KNN via larql-compute ──

    /// Load Q4_0 gate vectors from gate_vectors_q4.bin.
    ///
    /// File layout: layers packed contiguously, each layer is
    /// [num_features × hidden] in Q4_0 format (18 bytes per 32 elements).
    /// The per-layer feature count comes from gate_mmap_slices (must load
    /// f32/f16 gates first for the slice metadata, or pass feature counts).
    pub fn load_gate_vectors_q4(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join("gate_vectors_q4.bin");
        if !path.exists() {
            return Err(VindexError::Parse("gate_vectors_q4.bin not found".into()));
        }
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { mmap_optimized(&file)? };

        // Compute per-layer byte offsets from feature counts
        let mut slices = Vec::with_capacity(self.num_layers);
        let mut offset = 0usize;
        for layer in 0..self.num_layers {
            let num_features = self.num_features(layer);
            let floats = num_features * self.hidden_size;
            let q4_bytes = floats / 32 * 18; // Q4_0: 18 bytes per 32 elements
            slices.push(super::types::GateQ4Slice {
                byte_offset: offset,
                byte_len: q4_bytes,
                num_features,
            });
            offset += q4_bytes;
        }

        self.gate_q4_mmap = Some(Arc::new(mmap));
        self.gate_q4_slices = slices;
        Ok(())
    }

    /// Whether Q4 gate vectors are loaded.
    pub fn has_gate_q4(&self) -> bool {
        self.gate_q4_mmap.is_some()
    }

    /// Get Q4 data slice for a layer's gate vectors. Returns the raw Q4_0 bytes.
    pub fn gate_q4_data(&self, layer: usize) -> Option<&[u8]> {
        let mmap = self.gate_q4_mmap.as_ref()?;
        let slice = self.gate_q4_slices.get(layer)?;
        if slice.byte_len == 0 { return None; }
        let end = slice.byte_offset + slice.byte_len;
        if end > mmap.len() { return None; }
        Some(&mmap[slice.byte_offset..end])
    }

    /// Load Q8 attention weights + manifest for GPU full pipeline.
    pub fn load_attn_q8(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join("attn_weights_q8.bin");
        if !path.exists() {
            return Err(VindexError::Parse("attn_weights_q8.bin not found".into()));
        }
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { mmap_optimized(&file)? };
        self.attn_q8_mmap = Some(Arc::new(mmap));

        let manifest_path = dir.join("attn_weights_q8_manifest.json");
        if manifest_path.exists() {
            let json: Vec<serde_json::Value> = serde_json::from_str(
                &std::fs::read_to_string(&manifest_path)
                    .map_err(|e| VindexError::Parse(e.to_string()))?
            ).map_err(|e| VindexError::Parse(e.to_string()))?;

            let entries: Vec<(usize, usize, usize)> = json.iter()
                .map(|e| {
                    let offset = e["q8_offset"].as_u64().unwrap_or(0) as usize;
                    let vals_len = e["q8_vals_len"].as_u64().unwrap_or(0) as usize;
                    let scales_len = e["q8_scales_len"].as_u64().unwrap_or(0) as usize;
                    (offset, vals_len, scales_len)
                })
                .collect();
            self.attn_q8_manifest = Some(entries);
        }
        Ok(())
    }

    /// Get per-layer Q8 attention slices: (q_vals, q_scales, k_vals, k_scales, v_vals, v_scales, o_vals, o_scales)
    pub fn attn_q8_layer_data(&self, layer: usize) -> Option<[(&[u8], &[f32]); 4]> {
        let mmap = self.attn_q8_mmap.as_ref()?;
        let manifest = self.attn_q8_manifest.as_ref()?;

        let base = layer * 4;
        if base + 3 >= manifest.len() { return None; }

        let mut result = [(&[] as &[u8], &[] as &[f32]); 4];
        for i in 0..4 {
            let (offset, vals_len, scales_len) = manifest[base + i];
            let vals = &mmap[offset..offset + vals_len];
            let scales_start = offset + vals_len;
            let scales_data = &mmap[scales_start..scales_start + scales_len];
            let scales = unsafe {
                std::slice::from_raw_parts(
                    scales_data.as_ptr() as *const f32,
                    scales_len / 4,
                )
            };
            result[i] = (vals, scales);
        }
        Some(result)
    }

    /// Load Q4_K/Q6_K attention weights for Ollama-compatible GPU pipeline.
    pub fn load_attn_q4k(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join("attn_weights_q4k.bin");
        if !path.exists() {
            return Err(VindexError::Parse("attn_weights_q4k.bin not found".into()));
        }
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { mmap_optimized(&file)? };

        let manifest_path = dir.join("attn_weights_q4k_manifest.json");
        if manifest_path.exists() {
            let json: Vec<serde_json::Value> = serde_json::from_str(
                &std::fs::read_to_string(&manifest_path)
                    .map_err(|e| VindexError::Parse(e.to_string()))?
            ).map_err(|e| VindexError::Parse(e.to_string()))?;

            // Each entry: {key, shape, format, offset, length}
            let entries: Vec<(usize, usize, String)> = json.iter()
                .map(|e| {
                    let offset = e["offset"].as_u64().unwrap_or(0) as usize;
                    let length = e["length"].as_u64().unwrap_or(0) as usize;
                    let format = e["format"].as_str().unwrap_or("Q4_K").to_string();
                    (offset, length, format)
                })
                .collect();
            self.attn_q4k_manifest = Some(entries);
        }
        self.attn_q4k_mmap = Some(Arc::new(mmap));
        Ok(())
    }

    /// Get per-layer Q4_K/Q6_K attention slices: (data, format) for Q, K, V, O.
    pub fn attn_q4k_layer_data(&self, layer: usize) -> Option<[(&[u8], &str); 4]> {
        let mmap = self.attn_q4k_mmap.as_ref()?;
        let manifest = self.attn_q4k_manifest.as_ref()?;
        let base = layer * 4;
        if base + 3 >= manifest.len() { return None; }

        let mut result: [(&[u8], &str); 4] = [(&[], ""); 4];
        for i in 0..4 {
            let (offset, length, ref format) = manifest[base + i];
            result[i] = (&mmap[offset..offset + length], format.as_str());
        }
        Some(result)
    }

    /// Load Q4 attention weights + manifest for GPU full pipeline.
    pub fn load_attn_q4(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join("attn_weights_q4.bin");
        if !path.exists() {
            return Err(VindexError::Parse("attn_weights_q4.bin not found".into()));
        }
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { mmap_optimized(&file)? };
        self.attn_q4_mmap = Some(Arc::new(mmap));

        // Load manifest with per-matrix offsets
        let manifest_path = dir.join("attn_weights_q4_manifest.json");
        if manifest_path.exists() {
            let json: Vec<serde_json::Value> = serde_json::from_str(
                &std::fs::read_to_string(&manifest_path)
                    .map_err(|e| VindexError::Parse(e.to_string()))?
            ).map_err(|e| VindexError::Parse(e.to_string()))?;

            let entries: Vec<(usize, usize)> = json.iter()
                .map(|e| {
                    let offset = e["q4_offset"].as_u64().unwrap_or(0) as usize;
                    let length = e["q4_length"].as_u64().unwrap_or(0) as usize;
                    (offset, length)
                })
                .collect();
            self.attn_q4_manifest = Some(entries);
        }
        Ok(())
    }

    /// Get raw Q4 attention weight bytes (all layers packed).
    pub fn attn_q4_data(&self) -> Option<&[u8]> {
        self.attn_q4_mmap.as_ref().map(|m| m.as_ref() as &[u8])
    }

    /// Get per-layer Q4 attention weight slices (Q, K, V, O) using the manifest.
    /// Returns None if manifest or Q4 attn data is not loaded.
    #[allow(clippy::type_complexity)]
    pub fn attn_q4_layer_slices(&self, layer: usize) -> Option<(&[u8], &[u8], &[u8], &[u8])> {
        let mmap = self.attn_q4_mmap.as_ref()?;
        let manifest = self.attn_q4_manifest.as_ref()?;

        // Each layer has 4 tensors: Q, K, V, O
        let base = layer * 4;
        if base + 3 >= manifest.len() { return None; }

        let q = &manifest[base];
        let k = &manifest[base + 1];
        let v = &manifest[base + 2];
        let o = &manifest[base + 3];

        let q_data = &mmap[q.0..q.0 + q.1];
        let k_data = &mmap[k.0..k.0 + k.1];
        let v_data = &mmap[v.0..v.0 + v.1];
        let o_data = &mmap[o.0..o.0 + o.1];

        Some((q_data, k_data, v_data, o_data))
    }

    /// Load Q4 lm_head for GPU logits (replaces CPU f32 lm_head KNN).
    pub fn load_lm_head_q4(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join("lm_head_q4.bin");
        if !path.exists() {
            return Err(VindexError::Parse("lm_head_q4.bin not found".into()));
        }
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { mmap_optimized(&file)? };
        self.lm_head_q4_mmap = Some(Arc::new(mmap));
        Ok(())
    }

    /// Whether Q4 lm_head is loaded.
    pub fn has_lm_head_q4(&self) -> bool {
        self.lm_head_q4_mmap.is_some()
    }

    // ── LM head (output projection) for vindex logits ──

    /// Load lm_head from lm_head.bin for KNN logit lookup.
    pub fn load_lm_head(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join("lm_head.bin");
        if !path.exists() {
            return Err(VindexError::Parse("lm_head.bin not found".into()));
        }
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { mmap_optimized(&file)? };
        // Detect vocab size from file size: vocab = file_bytes / (hidden_size * 4)
        let vocab = mmap.len() / (self.hidden_size * 4);
        self.vocab_size = vocab;
        self.lm_head_mmap = Some(Arc::new(mmap));
        Ok(())
    }

    /// Whether lm_head is loaded for vindex logits.
    pub fn has_lm_head(&self) -> bool {
        self.lm_head_mmap.is_some() && self.vocab_size > 0
    }

    /// KNN against lm_head via a ComputeBackend — GPU Q4 or CPU BLAS.
    ///
    /// If Q4 lm_head data and a Q4-capable backend are provided, uses Q4 matvec (~1ms).
    /// Otherwise falls back to CPU BLAS f32 (~10ms).
    pub fn lm_head_knn_backend(
        &self,
        query: &ndarray::Array1<f32>,
        top_k: usize,
        backend: &dyn larql_compute::ComputeBackend,
    ) -> Vec<(u32, f32)> {
        // Try Q4 path first
        if backend.has_q4() {
            if let Some(ref q4_mmap) = self.lm_head_q4_mmap {
                let vocab = self.vocab_size;
                let hidden = self.hidden_size;
                if vocab > 0 {
                    let x = query.as_slice().unwrap();
                    let (q8_x, q8_scales) = larql_compute::cpu::q4::quantize_to_q8(x);
                    if let Some(scores_vec) = backend.q4_matvec(
                        q4_mmap.as_ref(), &q8_x, &q8_scales, vocab, hidden,
                    ) {
                        let mut indexed: Vec<(u32, f32)> = scores_vec.iter().copied().enumerate()
                            .map(|(i, s)| (i as u32, s))
                            .collect();
                        let k = top_k.min(indexed.len());
                        if k > 0 && k < indexed.len() {
                            indexed.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
                            indexed.truncate(k);
                        }
                        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                        return indexed;
                    }
                }
            }
        }
        // Fallback to f32 BLAS
        self.lm_head_knn(query, top_k)
    }

    /// KNN against lm_head: find top-K tokens by dot product with query vector.
    /// Single BLAS gemv: query[1, hidden] @ lm_head[vocab, hidden]^T → [1, vocab].
    /// Then top-K selection. Returns (token_id, score) sorted by score descending.
    pub fn lm_head_knn(&self, query: &ndarray::Array1<f32>, top_k: usize) -> Vec<(u32, f32)> {
        let mmap = match self.lm_head_mmap.as_ref() {
            Some(m) => m,
            None => return vec![],
        };
        let vocab = self.vocab_size;
        let hidden = self.hidden_size;
        if vocab == 0 { return vec![]; }

        let expected = vocab * hidden * 4;
        if mmap.len() < expected { return vec![]; }

        // Zero-copy: reinterpret mmap as [vocab, hidden] f32 matrix
        let data = unsafe {
            let ptr = mmap.as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, vocab * hidden)
        };
        let lm_view = ndarray::ArrayView2::from_shape((vocab, hidden), data).unwrap();

        // gemv via larql-compute: scores = query @ lm_head^T → [1, vocab]
        let hidden = self.hidden_size;
        let x = query.view().into_shape_with_order((1, hidden)).unwrap();
        let cpu = larql_compute::default_backend();
        use larql_compute::ComputeBackend;
        let result = cpu.matmul_transb(x, lm_view); // [1, hidden] @ [vocab, hidden]^T → [1, vocab]
        let scores = ndarray::Array1::from_vec(result.into_raw_vec_and_offset().0);

        // Top-K selection
        let mut indexed: Vec<(u32, f32)> = scores.iter().copied().enumerate()
            .map(|(i, s)| (i as u32, s))
            .collect();
        let k = top_k.min(indexed.len());
        if k > 0 && k < indexed.len() {
            indexed.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
            indexed.truncate(k);
        }
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed
    }
}
