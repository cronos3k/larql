//! Capture residual stream vectors and sparse activations for entities.
//!
//! High-level API: load a model, tokenize entities, run forward passes,
//! write NDJSON output files compatible with vector-load/SurrealDB.

use std::io::{BufWriter, Write};
use std::path::Path;

use crate::error::InferenceError;
use crate::forward::trace_forward;
use crate::model::{load_model_dir, resolve_model_path, ModelWeights};
use crate::tokenizer::load_tokenizer;

/// Configuration for residual/activation capture.
pub struct CaptureConfig {
    pub layers: Vec<usize>,
    pub prompt_template: Option<String>,
    pub capture_activations: bool,
    pub activation_top_k: usize,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            layers: vec![25],
            prompt_template: None,
            capture_activations: false,
            activation_top_k: 50,
        }
    }
}

/// Callbacks for capture progress.
pub trait CaptureCallbacks {
    fn on_entity_start(&mut self, _entity: &str, _index: usize, _total: usize) {}
    fn on_entity_done(&mut self, _entity: &str, _layers_captured: usize, _elapsed_ms: f64) {}
}

pub struct SilentCallbacks;
impl CaptureCallbacks for SilentCallbacks {}

/// Loaded model ready for inference and capture.
pub struct InferenceModel {
    weights: ModelWeights,
    tokenizer: tokenizers::Tokenizer,
    model_name: String,
}

// Re-export shared vector types from larql-models.
pub use larql_models::{TopKEntry, VectorFileHeader, VectorRecord};

/// Sparse activation record for NDJSON output.
#[derive(serde::Serialize)]
struct ActivationRecord {
    id: String,
    entity: String,
    layer: usize,
    activations: Vec<FeatureActivation>,
}

#[derive(serde::Serialize)]
struct FeatureActivation {
    feature: usize,
    magnitude: f32,
}

impl InferenceModel {
    /// Load a model from a path or HuggingFace model ID.
    pub fn load(model: &str) -> Result<Self, InferenceError> {
        let model_path = resolve_model_path(model)?;
        let weights = load_model_dir(&model_path)?;
        let tokenizer = load_tokenizer(&model_path)?;

        Ok(Self {
            weights,
            tokenizer,
            model_name: model.to_string(),
        })
    }

    pub fn num_layers(&self) -> usize {
        self.weights.num_layers
    }

    pub fn hidden_size(&self) -> usize {
        self.weights.hidden_size
    }

    pub fn weights(&self) -> &ModelWeights {
        &self.weights
    }

    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    /// Capture residuals and optionally activations for a list of entities.
    pub fn capture(
        &self,
        entities: &[String],
        config: &CaptureConfig,
        output_dir: &Path,
        callbacks: &mut dyn CaptureCallbacks,
    ) -> Result<(usize, usize), InferenceError> {
        std::fs::create_dir_all(output_dir)?;

        // Residuals writer
        let residual_path = output_dir.join("residuals.vectors.jsonl");
        let res_file = std::fs::File::create(&residual_path)?;
        let mut res_writer = std::io::BufWriter::new(res_file);

        let header = VectorFileHeader {
            _header: true,
            component: "residuals".to_string(),
            model: self.model_name.clone(),
            dimension: self.weights.hidden_size,
            extraction_date: current_date(),
        };
        serde_json::to_writer(&mut res_writer, &header)
            .map_err(|e| InferenceError::Parse(e.to_string()))?;
        res_writer.write_all(b"\n")?;

        // Activations writer (optional)
        let act_path = output_dir.join("activations.jsonl");
        let mut act_writer = if config.capture_activations {
            let file = std::fs::File::create(&act_path)?;
            Some(BufWriter::new(file))
        } else {
            None
        };

        let total = entities.len();
        let mut res_count = 0;
        let mut act_count = 0;

        for (i, entity) in entities.iter().enumerate() {
            let start = std::time::Instant::now();
            callbacks.on_entity_start(entity, i, total);

            let prompt = match &config.prompt_template {
                Some(tmpl) => tmpl.replace("{entity}", entity),
                None => entity.clone(),
            };

            let encoding = self
                .tokenizer
                .encode(prompt.as_str(), false)
                .map_err(|e| InferenceError::Parse(format!("tokenize error: {e}")))?;
            let token_ids: Vec<u32> = encoding.get_ids().to_vec();

            if token_ids.is_empty() {
                continue;
            }

            let trace = trace_forward(
                &self.weights,
                &token_ids,
                &config.layers,
                config.capture_activations,
                config.activation_top_k,
            );

            // Write residuals
            for (layer, vector) in &trace.residuals {
                let top_k = project_to_vocab(&self.weights.embed, vector, 10, &self.tokenizer);

                let (top_token, top_token_id, c_score) = if let Some(first) = top_k.first() {
                    (first.token.clone(), first.token_id, first.logit)
                } else {
                    (String::new(), 0, 0.0)
                };

                let record = VectorRecord {
                    id: format!("{entity}_L{layer}"),
                    layer: *layer,
                    feature: 0,
                    dim: vector.len(),
                    vector: vector.clone(),
                    top_token,
                    top_token_id,
                    c_score,
                    top_k,
                };
                serde_json::to_writer(&mut res_writer, &record)
                    .map_err(|e| InferenceError::Parse(e.to_string()))?;
                res_writer.write_all(b"\n")?;
                res_count += 1;
            }

            // Write activations
            if let Some(ref mut writer) = act_writer {
                for (layer, features) in &trace.activations {
                    let record = ActivationRecord {
                        id: format!("{entity}_L{layer}"),
                        entity: entity.clone(),
                        layer: *layer,
                        activations: features
                            .iter()
                            .map(|&(feat, mag)| FeatureActivation {
                                feature: feat,
                                magnitude: mag,
                            })
                            .collect(),
                    };
                    serde_json::to_writer(&mut *writer, &record)
                        .map_err(|e| InferenceError::Parse(e.to_string()))?;
                    writer.write_all(b"\n")?;
                    act_count += 1;
                }
            }

            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            callbacks.on_entity_done(entity, trace.residuals.len(), elapsed_ms);
        }

        res_writer.flush()?;
        if let Some(ref mut writer) = act_writer {
            writer.flush()?;
        }

        Ok((res_count, act_count))
    }
}

/// Project a residual vector onto the embedding matrix to find top-k tokens.
fn project_to_vocab(
    embed: &ndarray::Array2<f32>,
    residual: &[f32],
    k: usize,
    tokenizer: &tokenizers::Tokenizer,
) -> Vec<TopKEntry> {
    let vocab_size = embed.shape()[0];
    let mut scores: Vec<(usize, f32)> = Vec::with_capacity(vocab_size);

    for i in 0..vocab_size {
        let row = embed.row(i);
        let dot: f32 = row.iter().zip(residual.iter()).map(|(a, b)| a * b).sum();
        scores.push((i, dot));
    }

    let k = k.min(scores.len());
    if k > 0 && k < scores.len() {
        scores.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
    }
    scores.truncate(k);
    scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    scores
        .into_iter()
        .filter_map(|(idx, logit)| {
            tokenizer
                .decode(&[idx as u32], true)
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .map(|token| TopKEntry {
                    token,
                    token_id: idx as u32,
                    logit,
                })
        })
        .collect()
}

fn current_date() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let days = now / 86400;
    let year = 1970 + (days / 365);
    let remaining = days % 365;
    let month = remaining / 30 + 1;
    let day = remaining % 30 + 1;
    format!("{year}-{month:02}-{day:02}")
}
