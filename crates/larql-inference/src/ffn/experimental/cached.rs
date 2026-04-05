use std::collections::HashMap;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use ndarray::Array2;

use crate::error::InferenceError;
use crate::ffn::FfnBackend;
use crate::model::ModelWeights;

// ── Cached FFN: precomputed FFN outputs, zero matmuls at runtime ──

/// Cached FFN backend: stores precomputed FFN output matrices per layer.
/// Built by running a calibration forward pass for each entity.
/// Runtime: ArcArray clone = refcount bump (no memcpy), no matrix multiplications.
#[deprecated(note = "Research artifact — not scalable. Use WalkFfn.")]
pub struct CachedFfn {
    /// layer → shared FFN output matrix. Clone is O(1) refcount bump.
    cache: HashMap<usize, ndarray::ArcArray2<f32>>,
    hidden_size: usize,
}

impl CachedFfn {
    /// Build cache by running a dense forward pass, capturing FFN outputs at each layer.
    pub fn calibrate(
        weights: &ModelWeights,
        token_ids: &[u32],
    ) -> Self {
        use crate::ffn::WeightFfn;
        use crate::forward::trace_forward_with_ffn;

        let num_layers = weights.num_layers;
        let hidden = weights.hidden_size;
        let all_layers: Vec<usize> = (0..num_layers).collect();

        // Run forward pass capturing activations (to get FFN outputs)
        let ffn = WeightFfn { weights };
        let _trace = trace_forward_with_ffn(
            weights, token_ids, &all_layers, true, 1, &ffn,
        );

        // For each layer, compute the FFN delta:
        // FFN delta = post-FFN residual - post-attention residual
        // But we don't have those separately from trace. Instead, we can
        // re-derive: run attention to get post-attn, then the FFN output is
        // what the dense backend would produce.
        //
        // Simpler approach: run each layer's FFN on the captured residual.
        // The residual at layer L is the POST-layer-L state (after attn+FFN).
        // We need the PRE-FFN state (post-attention). We can get the FFN output
        // by running FFN on the normed residual.
        //
        // Actually the cleanest: run a second pass capturing FFN outputs directly.

        // Approach: run layer-by-layer, capture FFN output at each layer.
        let seq_len = token_ids.len();
        let embed_scale = weights.arch.embed_scale();
        let mut h = ndarray::Array2::<f32>::zeros((seq_len, hidden));
        for (i, &tok_id) in token_ids.iter().enumerate() {
            let row = weights.embed.row(tok_id as usize);
            for j in 0..hidden { h[[i, j]] = row[j] * embed_scale; }
        }

        let mut cache = HashMap::new();
        let norm_offset = weights.arch.norm_weight_offset();

        for layer in 0..num_layers {
            // Run attention
            let h_post_attn = match crate::forward::run_attention_public(weights, &h, layer) {
                Some(ha) => ha,
                None => { h = h.clone(); continue; }
            };

            // Compute FFN output on the post-attention residual
            let arch = &*weights.arch;
            let pre_ffn_key = if arch.has_post_norms() {
                arch.pre_feedforward_layernorm_key(layer)
            } else {
                Some(arch.post_attention_layernorm_key(layer))
            };
            let h_ffn = crate::residual::rms_norm(
                &h_post_attn,
                pre_ffn_key.and_then(|k| weights.vectors.get(&k)),
                norm_offset,
            );

            let ffn_out = ffn.forward(layer, &h_ffn);

            // Cache the full FFN output matrix (all positions)
            cache.insert(layer, ffn_out.clone().into_shared());

            // Apply FFN to get post-layer residual (for next layer)
            h = if arch.has_post_norms() {
                let normed = crate::residual::rms_norm(
                    &ffn_out,
                    arch.post_feedforward_layernorm_key(layer)
                        .and_then(|k| weights.vectors.get(&k)),
                    norm_offset,
                );
                &h_post_attn + &normed
            } else {
                &h_post_attn + &ffn_out
            };
        }

        CachedFfn { cache, hidden_size: hidden }
    }
}

impl CachedFfn {
    /// Direct access to cached output matrices (for zero-copy throughput paths).
    pub fn get_cache_vecs(&self) -> &HashMap<usize, ndarray::ArcArray2<f32>> {
        &self.cache
    }

    /// Save cache to a binary file. Format: JSON header line + raw f32 per layer.
    pub fn save(&self, path: &Path) -> Result<(), InferenceError> {
        let file = std::fs::File::create(path)?;
        let mut w = BufWriter::new(file);

        // Determine seq_len from first cached layer
        let seq_len = self.cache.values().next().map(|a| a.shape()[0]).unwrap_or(0);

        let mut sorted_layers: Vec<usize> = self.cache.keys().copied().collect();
        sorted_layers.sort();

        let header = serde_json::json!({
            "_type": "ffn_cache",
            "hidden_size": self.hidden_size,
            "seq_len": seq_len,
            "num_layers": self.cache.len(),
            "layers": sorted_layers,
        });
        serde_json::to_writer(&mut w, &header)
            .map_err(|e| InferenceError::Parse(e.to_string()))?;
        w.write_all(b"\n")?;

        // Write each layer's data as raw f32 in layer order
        let mut layers: Vec<usize> = self.cache.keys().copied().collect();
        layers.sort();
        for layer in layers {
            let arr = &self.cache[&layer];
            let slice = arr.as_slice().unwrap();
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * 4)
            };
            w.write_all(bytes)?;
        }
        w.flush()?;
        Ok(())
    }

    /// Load cache from a binary file.
    pub fn load(path: &Path) -> Result<Self, InferenceError> {
        let mut file = std::fs::File::open(path)?;
        let mut reader = BufReader::new(&mut file);

        // Read header line
        let mut header_line = String::new();
        reader.read_line(&mut header_line)?;
        let header: serde_json::Value = serde_json::from_str(&header_line)
            .map_err(|e| InferenceError::Parse(e.to_string()))?;

        let hidden_size = header["hidden_size"].as_u64().unwrap() as usize;
        let seq_len = header["seq_len"].as_u64().unwrap() as usize;
        let layers: Vec<usize> = header["layers"].as_array().unwrap()
            .iter().map(|v| v.as_u64().unwrap() as usize).collect();

        let floats_per_layer = seq_len * hidden_size;
        let bytes_per_layer = floats_per_layer * 4;
        let mut cache = HashMap::new();

        for layer in layers {
            let mut buf = vec![0u8; bytes_per_layer];
            std::io::Read::read_exact(&mut reader, &mut buf)?;
            let floats: Vec<f32> = buf.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let arr = ndarray::Array2::from_shape_vec((seq_len, hidden_size), floats)
                .map_err(|e| InferenceError::Parse(e.to_string()))?;
            cache.insert(layer, arr.into_shared());
        }

        Ok(CachedFfn { cache, hidden_size })
    }

    /// Number of cached layers.
    pub fn num_layers(&self) -> usize {
        self.cache.len()
    }
}

impl FfnBackend for CachedFfn {
    fn forward(&self, layer: usize, _x: &Array2<f32>) -> Array2<f32> {
        match self.cache.get(&layer) {
            // ArcArray clone = refcount bump (O(1)), then .into_owned() only copies
            // if there are other references. Since we hold the only Arc, this is
            // typically a no-op move. But even if it copies, it's just memcpy.
            Some(cached) => cached.clone().into_owned(),
            None => Array2::<f32>::zeros((_x.shape()[0], self.hidden_size)),
        }
    }

    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        (self.forward(layer, x), Array2::<f32>::zeros((x.shape()[0], 1)))
    }

    fn name(&self) -> &str {
        "cached"
    }
}
