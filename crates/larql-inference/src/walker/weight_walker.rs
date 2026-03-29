//! Walk FFN weight matrices — extract edges directly from model parameters.
//!
//! Every FFN feature is a potential edge. For each feature in a layer:
//! - What tokens activate it (gate projection) → c_in
//! - What tokens it produces (down projection) → c_out
//! - Confidence = c_in × c_out, normalized per-layer to [0, 1]
//!
//! Zero forward passes. Pure matrix multiplication.

use larql_core::core::edge::Edge;
use larql_core::core::enums::SourceType;
use larql_core::core::graph::Graph;

use super::utils::{count_threshold, decode_token, partial_top_k_column, top_entities};
use crate::error::InferenceError;
use crate::model::{load_model_dir, resolve_model_path, ModelWeights};

/// Result of walking a single layer.
#[derive(Debug, Clone)]
pub struct LayerResult {
    pub layer: usize,
    pub features_scanned: usize,
    pub edges_found: usize,
    pub elapsed_ms: f64,
    pub stats: LayerStats,
}

/// Per-layer statistics for validation.
#[derive(Debug, Clone, Default)]
pub struct LayerStats {
    pub mean_confidence: f64,
    pub max_confidence: f64,
    pub min_confidence: f64,
    pub mean_c_in: f64,
    pub mean_c_out: f64,
    pub mean_selectivity: f64,
    pub max_selectivity: f64,
    pub self_loop_count: usize,
    pub self_loop_pct: f64,
    pub top_subjects: Vec<(String, usize, f64)>,
    pub top_objects: Vec<(String, usize, f64)>,
    /// Edges surviving at each confidence threshold.
    pub threshold_counts: ThresholdCounts,
    /// Edges surviving at each selectivity threshold.
    pub selectivity_threshold_counts: ThresholdCounts,
}

/// Edge counts at standard thresholds.
#[derive(Debug, Clone, Default)]
pub struct ThresholdCounts {
    pub t_01: usize,
    pub t_05: usize,
    pub t_10: usize,
    pub t_25: usize,
    pub t_50: usize,
    pub t_75: usize,
    pub t_90: usize,
}

/// Configuration for the weight walker.
pub struct WalkConfig {
    pub top_k: usize,
    pub min_score: f32,
}

impl Default for WalkConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            min_score: 0.02,
        }
    }
}

/// Callbacks for walk progress.
pub trait WalkCallbacks {
    fn on_layer_start(&mut self, _layer: usize, _num_features: usize) {}
    fn on_layer_done(&mut self, _result: &LayerResult) {}
    fn on_checkpoint(&mut self, _graph: &Graph) {}
    fn on_progress(&mut self, _layer: usize, _features_done: usize, _total: usize) {}
}

pub struct SilentWalkCallbacks;
impl WalkCallbacks for SilentWalkCallbacks {}

// resolve_model_path is re-exported from crate::model via the import above.

/// A loaded model ready for weight walking.
pub struct WeightWalker {
    weights: ModelWeights,
    tokenizer: tokenizers::Tokenizer,
}

/// A raw edge before per-layer normalization.
struct RawEdge {
    subject: String,
    relation: String,
    object: String,
    c_in: f32,
    c_out: f32,
    layer: usize,
    feature: usize,
}

impl WeightWalker {
    pub fn load(model: &str) -> Result<Self, InferenceError> {
        let model_path = resolve_model_path(model)?;
        let weights = load_model_dir(&model_path)?;

        let tokenizer_path = model_path.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(InferenceError::MissingTensor(
                "tokenizer.json not found".into(),
            ));
        }
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| InferenceError::Parse(e.to_string()))?;

        Ok(Self { weights, tokenizer })
    }

    pub fn num_layers(&self) -> usize {
        self.weights.num_layers
    }

    /// Walk a single layer's FFN weights into the graph.
    ///
    /// Confidence scoring:
    /// - `c_in`: raw gate projection score (input selectivity)
    /// - `c_out`: raw down projection score (output strength)
    /// - `c`: normalized `c_in × c_out`, scaled to [0,1] per layer
    pub fn walk_layer(
        &self,
        layer: usize,
        config: &WalkConfig,
        graph: &mut Graph,
        callbacks: &mut dyn WalkCallbacks,
    ) -> Result<LayerResult, InferenceError> {
        let start = std::time::Instant::now();

        let prefix = format!("layers.{layer}.mlp.");
        let w_gate = self
            .weights
            .tensors
            .get(&format!("{prefix}gate_proj.weight"))
            .ok_or_else(|| InferenceError::MissingTensor(format!("{prefix}gate_proj.weight")))?;
        let w_down = self
            .weights
            .tensors
            .get(&format!("{prefix}down_proj.weight"))
            .ok_or_else(|| InferenceError::MissingTensor(format!("{prefix}down_proj.weight")))?;

        let n_features = w_down.shape()[1];
        callbacks.on_layer_start(layer, n_features);

        // BLAS-accelerated matmuls
        let all_output = self.weights.embed.dot(w_down);
        let all_input = self.weights.embed.dot(&w_gate.t());

        let k = config.top_k.min(self.weights.vocab_size);
        let progress_interval = (n_features / 20).max(1);

        // Phase 1: collect raw edges with c_in / c_out
        let mut raw_edges: Vec<RawEdge> = Vec::new();

        for feat_idx in 0..n_features {
            if feat_idx % progress_interval == 0 {
                callbacks.on_progress(layer, feat_idx, n_features);
            }

            let top_in = partial_top_k_column(&all_input, feat_idx, k);
            let top_out = partial_top_k_column(&all_output, feat_idx, k);

            let mut subjects: Vec<(String, f32)> = Vec::new();
            for (idx, score) in &top_in {
                if *score >= config.min_score {
                    if let Some(tok) = decode_token(&self.tokenizer, *idx as u32) {
                        if !tok.is_empty() {
                            subjects.push((tok, *score));
                        }
                    }
                }
            }

            let mut objects: Vec<(String, f32)> = Vec::new();
            for (idx, score) in &top_out {
                if *score >= config.min_score {
                    if let Some(tok) = decode_token(&self.tokenizer, *idx as u32) {
                        if !tok.is_empty() {
                            objects.push((tok, *score));
                        }
                    }
                }
            }

            if subjects.is_empty() || objects.is_empty() {
                continue;
            }

            let relation = format!("L{layer}-F{feat_idx}");
            for (subj, c_in) in &subjects {
                for (obj, c_out) in &objects {
                    raw_edges.push(RawEdge {
                        subject: subj.clone(),
                        relation: relation.clone(),
                        object: obj.clone(),
                        c_in: *c_in,
                        c_out: *c_out,
                        layer,
                        feature: feat_idx,
                    });
                }
            }
        }

        // Phase 2: per-layer normalization
        // confidence = (c_in × c_out) / max(c_in × c_out)
        // selectivity = c_in / max(c_in) — factual signal
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

        let mut conf_thresholds = ThresholdCounts::default();
        let mut sel_thresholds = ThresholdCounts::default();

        let mut subj_counts: std::collections::HashMap<String, (usize, f64)> =
            std::collections::HashMap::new();
        let mut obj_counts: std::collections::HashMap<String, (usize, f64)> =
            std::collections::HashMap::new();

        // Phase 3: add normalized edges to graph
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

            // Track entity frequencies
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
                .with_metadata("feature", serde_json::Value::from(raw.feature as u64))
                .with_metadata("c_in", serde_json::Value::from(raw.c_in as f64))
                .with_metadata("c_out", serde_json::Value::from(raw.c_out as f64))
                .with_metadata("selectivity", serde_json::Value::from(selectivity));
            graph.add_edge(edge);
        }

        let n = raw_edges.len();

        // Build top entities by count
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
        let result = LayerResult {
            layer,
            features_scanned: n_features,
            edges_found: n,
            elapsed_ms: elapsed,
            stats,
        };
        callbacks.on_layer_done(&result);
        callbacks.on_checkpoint(graph);
        Ok(result)
    }
}

/// Convenience: load model and walk all (or selected) layers.
pub fn walk_model(
    model: &str,
    layers: Option<&[usize]>,
    config: &WalkConfig,
    graph: &mut Graph,
    callbacks: &mut dyn WalkCallbacks,
) -> Result<Vec<LayerResult>, InferenceError> {
    let walker = WeightWalker::load(model)?;

    let layer_indices: Vec<usize> = match layers {
        Some(ls) => ls.to_vec(),
        None => (0..walker.num_layers()).collect(),
    };

    let mut results = Vec::new();
    for &layer in &layer_indices {
        let result = walker.walk_layer(layer, config, graph, callbacks)?;
        results.push(result);
    }

    Ok(results)
}
