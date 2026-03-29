//! Walk attention OV circuits — extract routing edges from attention heads.
//!
//! Each attention head is a routing rule. The OV circuit tells you:
//! when this head fires on input X, it copies output Y.
//! That's an edge: X → Y via this head.
//!
//! Confidence scoring:
//! - c_in: input amplification norm (how much this head amplifies the input)
//! - c_out: output logit magnitude (how strongly it projects to the output token)
//! - c: normalized (c_in × c_out) / max per layer
//!
//! Zero forward passes. Pure matrix multiplication.

use larql_core::core::edge::Edge;
use larql_core::core::enums::SourceType;
use larql_core::core::graph::Graph;

use super::utils::{count_threshold, decode_token, partial_top_k, top_entities};
use super::weight_walker::{LayerResult, LayerStats, WalkCallbacks, WalkConfig};
use crate::error::InferenceError;
use crate::model::{resolve_model_path, ModelWeights};

/// Result of walking attention heads at a single layer.
#[derive(Debug, Clone)]
pub struct AttentionLayerResult {
    pub layer: usize,
    pub heads_walked: usize,
    pub edges_found: usize,
    pub elapsed_ms: f64,
    pub stats: LayerStats,
}

/// A raw attention edge before per-layer normalization.
struct RawEdge {
    subject: String,
    relation: String,
    object: String,
    c_in: f32,
    c_out: f32,
    layer: usize,
    head: usize,
}

/// A loaded model ready for attention walking.
pub struct AttentionWalker {
    weights: ModelWeights,
    tokenizer: tokenizers::Tokenizer,
    head_dim: usize,
}

impl AttentionWalker {
    pub fn load(model: &str) -> Result<Self, InferenceError> {
        let model_path = resolve_model_path(model)?;
        let weights = crate::model::load_model_dir(&model_path)?;

        let tokenizer_path = model_path.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(InferenceError::MissingTensor(
                "tokenizer.json not found".into(),
            ));
        }
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| InferenceError::Parse(e.to_string()))?;

        let head_dim = weights.head_dim;

        Ok(Self {
            weights,
            tokenizer,
            head_dim,
        })
    }

    pub fn num_layers(&self) -> usize {
        self.weights.num_layers
    }

    /// Walk all attention heads at a single layer.
    pub fn walk_layer(
        &self,
        layer: usize,
        config: &WalkConfig,
        graph: &mut Graph,
        callbacks: &mut dyn WalkCallbacks,
    ) -> Result<AttentionLayerResult, InferenceError> {
        let start = std::time::Instant::now();

        let prefix = format!("layers.{layer}.self_attn.");
        let w_v = self
            .weights
            .tensors
            .get(&format!("{prefix}v_proj.weight"))
            .ok_or_else(|| InferenceError::MissingTensor(format!("{prefix}v_proj.weight")))?;
        let w_o = self
            .weights
            .tensors
            .get(&format!("{prefix}o_proj.weight"))
            .ok_or_else(|| InferenceError::MissingTensor(format!("{prefix}o_proj.weight")))?;

        let num_kv_heads = w_v.shape()[0] / self.head_dim;
        callbacks.on_layer_start(layer, num_kv_heads);

        let k_inputs = (config.top_k * 10).min(self.weights.vocab_size);

        // Phase 1: collect raw edges with c_in / c_out
        let mut raw_edges: Vec<RawEdge> = Vec::new();

        for h in 0..num_kv_heads {
            callbacks.on_progress(layer, h, num_kv_heads);

            let hd = self.head_dim;
            let v_h = w_v.slice(ndarray::s![h * hd..(h + 1) * hd, ..]);
            let o_h = w_o.slice(ndarray::s![.., h * hd..(h + 1) * hd]);

            // OV circuit: O_h @ V_h -> (hidden, hidden)
            let ov = o_h.dot(&v_h);

            // transformed = embed @ OV.T -> (vocab, hidden)
            let transformed = self.weights.embed.dot(&ov.t());

            // Norms: how much this head amplifies each input (c_in candidate)
            let norms: Vec<f32> = (0..self.weights.vocab_size)
                .map(|i| {
                    let row = transformed.row(i);
                    row.iter().map(|x| x * x).sum::<f32>().sqrt()
                })
                .collect();

            // Top-k inputs by amplification
            let top_inputs = partial_top_k(&norms, k_inputs);

            let relation = format!("L{layer}-H{h}");
            for &(inp_idx, c_in) in &top_inputs {
                let inp_vec = transformed.row(inp_idx);

                // out_logits = embed @ inp_vec -> (vocab,)
                let out_logits: Vec<f32> = (0..self.weights.vocab_size)
                    .map(|j| {
                        self.weights
                            .embed
                            .row(j)
                            .iter()
                            .zip(inp_vec.iter())
                            .map(|(a, b)| a * b)
                            .sum()
                    })
                    .collect();

                let top_out = partial_top_k(&out_logits, config.top_k);

                let inp_token = match decode_token(&self.tokenizer, inp_idx as u32) {
                    Some(t) if !t.is_empty() => t,
                    _ => continue,
                };

                for &(out_idx, c_out) in &top_out {
                    if c_out < config.min_score {
                        continue;
                    }
                    let out_token = match decode_token(&self.tokenizer, out_idx as u32) {
                        Some(t) if !t.is_empty() => t,
                        _ => continue,
                    };

                    raw_edges.push(RawEdge {
                        subject: inp_token.clone(),
                        relation: relation.clone(),
                        object: out_token,
                        c_in,
                        c_out,
                        layer,
                        head: h,
                    });
                }
            }
        }

        // Phase 2: per-layer normalization
        let max_product = raw_edges
            .iter()
            .map(|e| e.c_in * e.c_out)
            .fold(f32::MIN, f32::max)
            .max(f32::EPSILON);

        let max_cin = raw_edges
            .iter()
            .map(|e| e.c_in)
            .fold(f32::MIN, f32::max)
            .max(f32::EPSILON);

        let mut sum_conf = 0.0f64;
        let mut sum_sel = 0.0f64;
        let mut sum_cin = 0.0f64;
        let mut sum_cout = 0.0f64;
        let mut max_conf = 0.0f64;
        let mut min_conf = 1.0f64;
        let mut max_sel = 0.0f64;
        let mut self_loops = 0usize;

        let mut conf_thresholds = super::weight_walker::ThresholdCounts::default();
        let mut sel_thresholds = super::weight_walker::ThresholdCounts::default();

        let mut subj_counts: std::collections::HashMap<String, (usize, f64)> =
            std::collections::HashMap::new();
        let mut obj_counts: std::collections::HashMap<String, (usize, f64)> =
            std::collections::HashMap::new();

        for raw in &raw_edges {
            let product = raw.c_in * raw.c_out;
            let confidence = (product / max_product) as f64;
            let selectivity = (raw.c_in / max_cin) as f64;

            sum_conf += confidence;
            sum_sel += selectivity;
            sum_cin += raw.c_in as f64;
            sum_cout += raw.c_out as f64;
            if confidence > max_conf {
                max_conf = confidence;
            }
            if confidence < min_conf {
                min_conf = confidence;
            }
            if selectivity > max_sel {
                max_sel = selectivity;
            }
            if raw.subject == raw.object {
                self_loops += 1;
            }

            count_threshold(&mut conf_thresholds, confidence);
            count_threshold(&mut sel_thresholds, selectivity);

            let se = subj_counts.entry(raw.subject.clone()).or_insert((0, 0.0));
            se.0 += 1;
            se.1 += confidence;
            let oe = obj_counts.entry(raw.object.clone()).or_insert((0, 0.0));
            oe.0 += 1;
            oe.1 += confidence;

            let edge = Edge::new(&raw.subject, &raw.relation, &raw.object)
                .with_confidence(confidence)
                .with_source(SourceType::Parametric)
                .with_metadata("layer", serde_json::Value::from(raw.layer as u64))
                .with_metadata("head", serde_json::Value::from(raw.head as u64))
                .with_metadata("circuit", serde_json::Value::String("OV".to_string()))
                .with_metadata("c_in", serde_json::Value::from(raw.c_in as f64))
                .with_metadata("c_out", serde_json::Value::from(raw.c_out as f64))
                .with_metadata("selectivity", serde_json::Value::from(selectivity));
            graph.add_edge(edge);
        }

        let n = raw_edges.len();
        let top_subjects = top_entities(&subj_counts, 10);
        let top_objects = top_entities(&obj_counts, 10);

        let stats = if n > 0 {
            LayerStats {
                mean_confidence: sum_conf / n as f64,
                max_confidence: max_conf,
                min_confidence: min_conf,
                mean_c_in: sum_cin / n as f64,
                mean_c_out: sum_cout / n as f64,
                mean_selectivity: sum_sel / n as f64,
                max_selectivity: max_sel,
                self_loop_count: self_loops,
                self_loop_pct: if n > 0 {
                    (self_loops as f64 / n as f64) * 100.0
                } else {
                    0.0
                },
                top_subjects,
                top_objects,
                threshold_counts: conf_thresholds,
                selectivity_threshold_counts: sel_thresholds,
            }
        } else {
            LayerStats::default()
        };

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        callbacks.on_layer_done(&LayerResult {
            layer,
            features_scanned: num_kv_heads,
            edges_found: n,
            elapsed_ms: elapsed,
            stats: stats.clone(),
        });
        callbacks.on_checkpoint(graph);

        Ok(AttentionLayerResult {
            layer,
            heads_walked: num_kv_heads,
            edges_found: n,
            elapsed_ms: elapsed,
            stats,
        })
    }
}
