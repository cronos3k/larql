use ndarray::Array2;

use crate::ffn::{sigmoid, FfnBackend};
use crate::model::ModelWeights;

// ── Precomputed feature lists: calibrate once, sparse FFN at query time ──

/// Stores precomputed feature lists per layer from a calibration forward pass.
/// At query time: attention runs live, FFN uses these feature lists for sparse
/// gate/up/down — no gate matmul scan.
#[deprecated(note = "Research artifact — not scalable. Use WalkFfn.")]
pub struct FeatureListFfn<'a> {
    pub weights: &'a ModelWeights,
    /// layer → sorted feature indices (the ~50 features the gate matmul would select)
    feature_lists: Vec<Vec<usize>>,
}

impl<'a> FeatureListFfn<'a> {
    /// Calibrate: run a dense forward pass, capture which features the gate selects at each layer.
    pub fn calibrate(
        weights: &'a ModelWeights,
        token_ids: &[u32],
        top_k: usize,
    ) -> Self {
        use crate::ffn::WeightFfn;

        let num_layers = weights.num_layers;
        let hidden = weights.hidden_size;
        let seq_len = token_ids.len();
        let embed_scale = weights.arch.embed_scale();

        let mut h = ndarray::Array2::<f32>::zeros((seq_len, hidden));
        for (i, &tok_id) in token_ids.iter().enumerate() {
            let row = weights.embed.row(tok_id as usize);
            for j in 0..hidden { h[[i, j]] = row[j] * embed_scale; }
        }

        let ffn = WeightFfn { weights };
        let norm_offset = weights.arch.norm_weight_offset();
        let mut feature_lists = vec![Vec::new(); num_layers];

        for layer in 0..num_layers {
            // Run attention
            let h_post_attn = match crate::forward::run_attention_public(weights, &h, layer) {
                Some(ha) => ha,
                None => { continue; }
            };

            // Get the pre-FFN normed residual (what the gate matmul sees)
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

            // Gate matmul on last position → find top-K features
            let w_gate = weights.tensors.get(&arch.ffn_gate_key(layer)).unwrap();
            let last_row = h_ffn.row(seq_len - 1);
            let scores = w_gate.dot(&last_row);
            let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate()
                .map(|(i, v)| (i, v * sigmoid(v)))
                .collect();
            let k = top_k.min(indexed.len());
            indexed.select_nth_unstable_by(k, |a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
            indexed.truncate(k);
            let mut feats: Vec<usize> = indexed.iter().map(|&(id, _)| id).collect();
            feats.sort_unstable();
            feature_lists[layer] = feats;

            // Run dense FFN to get correct residual for next layer
            let ffn_out = ffn.forward(layer, &h_ffn);
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

        FeatureListFfn { weights, feature_lists }
    }

    /// Save feature lists to a compact binary file.
    /// Format: JSON header + one line per layer with feature IDs.
    pub fn save(&self, path: &std::path::Path) -> Result<(), crate::error::InferenceError> {
        use std::io::Write;
        let file = std::fs::File::create(path)?;
        let mut w = std::io::BufWriter::new(file);

        let header = serde_json::json!({
            "_type": "feature_lists",
            "num_layers": self.feature_lists.len(),
        });
        serde_json::to_writer(&mut w, &header)
            .map_err(|e| crate::error::InferenceError::Parse(e.to_string()))?;
        w.write_all(b"\n")?;

        for (layer, feats) in self.feature_lists.iter().enumerate() {
            let record = serde_json::json!({ "l": layer, "f": feats });
            serde_json::to_writer(&mut w, &record)
                .map_err(|e| crate::error::InferenceError::Parse(e.to_string()))?;
            w.write_all(b"\n")?;
        }
        w.flush()?;
        Ok(())
    }

    /// Load feature lists from file.
    pub fn load(
        weights: &'a ModelWeights,
        path: &std::path::Path,
    ) -> Result<Self, crate::error::InferenceError> {
        use std::io::BufRead;
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);

        let num_layers = weights.num_layers;
        let mut feature_lists = vec![Vec::new(); num_layers];

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() { continue; }
            let obj: serde_json::Value = serde_json::from_str(line)
                .map_err(|e| crate::error::InferenceError::Parse(e.to_string()))?;
            if obj.get("_type").is_some() { continue; }

            let layer = obj["l"].as_u64().unwrap_or(0) as usize;
            let feats: Vec<usize> = obj["f"].as_array().unwrap()
                .iter().map(|v| v.as_u64().unwrap() as usize).collect();
            if layer < num_layers {
                feature_lists[layer] = feats;
            }
        }

        Ok(FeatureListFfn { weights, feature_lists })
    }

    pub fn total_features(&self) -> usize {
        self.feature_lists.iter().map(|f| f.len()).sum()
    }

    pub fn avg_features_per_layer(&self) -> f64 {
        let active: Vec<_> = self.feature_lists.iter().filter(|f| !f.is_empty()).collect();
        if active.is_empty() { 0.0 } else {
            active.iter().map(|f| f.len()).sum::<usize>() as f64 / active.len() as f64
        }
    }
}

impl<'a> FfnBackend for FeatureListFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        self.forward_inner(layer, x).0
    }
    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        self.forward_inner(layer, x)
    }
    fn name(&self) -> &str { "feature-list" }
}

impl<'a> FeatureListFfn<'a> {
    fn forward_inner(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let features = &self.feature_lists[layer];
        crate::ffn::sparse_compute::sparse_ffn_forward(self.weights, layer, x, features)
    }
}
