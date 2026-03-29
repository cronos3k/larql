//! Extract full vectors from model weight matrices to intermediate NDJSON files.
//!
//! Same safetensors loading and BLAS matmuls as weight_walker, but captures the
//! full weight vector (hidden-dim) alongside top-k token metadata. Output is one
//! `.vectors.jsonl` file per component type (ffn_down, ffn_gate, etc.).
//!
//! The stored vector is the raw weight direction (dim = hidden_size), NOT the
//! vocab-projected logits. The vocab projection is computed only to derive
//! top-k token metadata (same matmul as weight_walker).
//!
//! Zero forward passes. Pure matrix multiplication.

use std::collections::HashSet;
use std::io::{BufRead, BufWriter, Write};
use std::path::{Path, PathBuf};

use super::utils::{current_date, decode_token, partial_top_k, partial_top_k_column};
use crate::error::InferenceError;
use crate::model::{load_model_dir, resolve_model_path, ModelWeights};

// Re-export shared vector types from larql-models.
pub use larql_models::{
    TopKEntry, VectorFileHeader, VectorRecord, ALL_COMPONENTS, COMPONENT_ATTN_OV,
    COMPONENT_ATTN_QK, COMPONENT_EMBEDDINGS, COMPONENT_FFN_DOWN, COMPONENT_FFN_GATE,
    COMPONENT_FFN_UP,
};

/// Configuration for vector extraction.
pub struct ExtractConfig {
    pub components: Vec<String>,
    pub layers: Option<Vec<usize>>,
    pub top_k: usize,
}

impl Default for ExtractConfig {
    fn default() -> Self {
        Self {
            components: ALL_COMPONENTS.iter().map(|s| s.to_string()).collect(),
            layers: None,
            top_k: 10,
        }
    }
}

/// Callbacks for extraction progress.
pub trait ExtractCallbacks {
    fn on_component_start(&mut self, _component: &str, _total_layers: usize) {}
    fn on_layer_start(&mut self, _component: &str, _layer: usize, _num_vectors: usize) {}
    fn on_progress(&mut self, _component: &str, _layer: usize, _done: usize, _total: usize) {}
    fn on_layer_done(
        &mut self,
        _component: &str,
        _layer: usize,
        _vectors_written: usize,
        _elapsed_ms: f64,
    ) {
    }
    fn on_component_done(&mut self, _component: &str, _total_written: usize) {}
}

pub struct SilentExtractCallbacks;
impl ExtractCallbacks for SilentExtractCallbacks {}

/// Summary of a full extraction run.
pub struct ExtractSummary {
    pub components: Vec<ComponentSummary>,
    pub total_vectors: usize,
    pub elapsed_secs: f64,
}

/// Summary for a single component.
pub struct ComponentSummary {
    pub component: String,
    pub vectors_written: usize,
    pub output_path: PathBuf,
    pub elapsed_secs: f64,
}

/// Streaming NDJSON writer for vector records.
pub struct VectorWriter {
    writer: BufWriter<std::fs::File>,
    count: usize,
}

impl VectorWriter {
    /// Create a new writer, truncating any existing file.
    pub fn create(path: &Path) -> Result<Self, InferenceError> {
        let file = std::fs::File::create(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
            count: 0,
        })
    }

    /// Open an existing file for appending and count existing records.
    pub fn append(path: &Path) -> Result<(Self, usize), InferenceError> {
        // Count existing lines (excluding header)
        let existing = if path.exists() {
            let file = std::fs::File::open(path)?;
            let reader = std::io::BufReader::new(file);
            let total_lines = reader.lines().count();
            if total_lines > 0 {
                total_lines - 1 // subtract header
            } else {
                0
            }
        } else {
            0
        };

        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok((
            Self {
                writer: BufWriter::new(file),
                count: existing,
            },
            existing,
        ))
    }

    /// Write the metadata header as the first line.
    pub fn write_header(&mut self, header: &VectorFileHeader) -> Result<(), InferenceError> {
        serde_json::to_writer(&mut self.writer, header)
            .map_err(|e| InferenceError::Parse(e.to_string()))?;
        self.writer.write_all(b"\n")?;
        Ok(())
    }

    /// Write a single vector record as one NDJSON line.
    pub fn write_record(&mut self, record: &VectorRecord) -> Result<(), InferenceError> {
        serde_json::to_writer(&mut self.writer, record)
            .map_err(|e| InferenceError::Parse(e.to_string()))?;
        self.writer.write_all(b"\n")?;
        self.count += 1;
        Ok(())
    }

    pub fn flush(&mut self) -> Result<(), InferenceError> {
        self.writer.flush()?;
        Ok(())
    }

    pub fn count(&self) -> usize {
        self.count
    }
}

/// Scan an existing NDJSON file for completed layer numbers.
pub fn scan_completed_layers(path: &Path) -> Result<HashSet<usize>, InferenceError> {
    let mut layers = HashSet::new();
    if !path.exists() {
        return Ok(layers);
    }

    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);

    for line in reader.lines() {
        let line = line?;
        // Quick parse: extract "layer" field without full deserialization
        if let Some(pos) = line.find("\"layer\":") {
            let rest = &line[pos + 8..];
            let num_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(layer) = num_str.parse::<usize>() {
                layers.insert(layer);
            }
        }
    }

    Ok(layers)
}

/// A loaded model ready for vector extraction.
pub struct VectorExtractor {
    weights: ModelWeights,
    tokenizer: tokenizers::Tokenizer,
    model_name: String,
}

impl VectorExtractor {
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

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Extract FFN down vectors for a single layer.
    ///
    /// The stored vector is `w_down.column(feat)` — the raw weight direction in
    /// hidden space (dim = hidden_size). The vocab projection (`embed @ w_down`)
    /// is computed only to derive top-k token metadata.
    pub fn extract_ffn_down(
        &self,
        layer: usize,
        config: &ExtractConfig,
        writer: &mut VectorWriter,
        callbacks: &mut dyn ExtractCallbacks,
    ) -> Result<usize, InferenceError> {
        let prefix = format!("layers.{layer}.mlp.");
        let w_down = self
            .weights
            .tensors
            .get(&format!("{prefix}down_proj.weight"))
            .ok_or_else(|| InferenceError::MissingTensor(format!("{prefix}down_proj.weight")))?;

        // w_down shape: (hidden, intermediate)
        let n_features = w_down.shape()[1];
        callbacks.on_layer_start(COMPONENT_FFN_DOWN, layer, n_features);

        // Vocab projection for top-k metadata: (vocab, hidden) @ (hidden, intermediate) → (vocab, intermediate)
        let logits = self.weights.embed.dot(w_down);

        let progress_interval = (n_features / 20).max(1);
        let mut count = 0;

        for feat_idx in 0..n_features {
            if feat_idx % progress_interval == 0 {
                callbacks.on_progress(COMPONENT_FFN_DOWN, layer, feat_idx, n_features);
            }

            // Raw weight vector in hidden space (dim = hidden_size)
            let vector: Vec<f32> = w_down.column(feat_idx).to_vec();

            // Top-k tokens from vocab projection
            let top_k_pairs = partial_top_k_column(&logits, feat_idx, config.top_k);
            let top_k: Vec<TopKEntry> = top_k_pairs
                .iter()
                .filter_map(|&(idx, logit)| {
                    decode_token(&self.tokenizer, idx as u32).map(|token| TopKEntry {
                        token,
                        token_id: idx as u32,
                        logit,
                    })
                })
                .collect();

            let (top_token, top_token_id, c_score) = if let Some(first) = top_k.first() {
                (first.token.clone(), first.token_id, first.logit)
            } else {
                (String::new(), 0, 0.0)
            };

            writer.write_record(&VectorRecord {
                id: format!("L{layer}_F{feat_idx}"),
                layer,
                feature: feat_idx,
                dim: vector.len(),
                vector,
                top_token,
                top_token_id,
                c_score,
                top_k,
            })?;
            count += 1;
        }

        writer.flush()?;
        Ok(count)
    }

    /// Extract FFN gate vectors for a single layer.
    ///
    /// The stored vector is `w_gate.row(feat)` — the raw weight direction in
    /// hidden space (dim = hidden_size). The vocab projection (`embed @ w_gate.T`)
    /// is computed only to derive top-k token metadata.
    pub fn extract_ffn_gate(
        &self,
        layer: usize,
        config: &ExtractConfig,
        writer: &mut VectorWriter,
        callbacks: &mut dyn ExtractCallbacks,
    ) -> Result<usize, InferenceError> {
        let prefix = format!("layers.{layer}.mlp.");
        let w_gate = self
            .weights
            .tensors
            .get(&format!("{prefix}gate_proj.weight"))
            .ok_or_else(|| InferenceError::MissingTensor(format!("{prefix}gate_proj.weight")))?;

        // w_gate shape: (intermediate, hidden)
        let n_features = w_gate.shape()[0];
        callbacks.on_layer_start(COMPONENT_FFN_GATE, layer, n_features);

        // Vocab projection for top-k metadata: (vocab, hidden) @ (intermediate, hidden).T → (vocab, intermediate)
        let logits = self.weights.embed.dot(&w_gate.t());

        let progress_interval = (n_features / 20).max(1);
        let mut count = 0;

        for feat_idx in 0..n_features {
            if feat_idx % progress_interval == 0 {
                callbacks.on_progress(COMPONENT_FFN_GATE, layer, feat_idx, n_features);
            }

            // Raw weight vector in hidden space (dim = hidden_size)
            let vector: Vec<f32> = w_gate.row(feat_idx).to_vec();

            // Top-k tokens from vocab projection
            let top_k_pairs = partial_top_k_column(&logits, feat_idx, config.top_k);
            let top_k: Vec<TopKEntry> = top_k_pairs
                .iter()
                .filter_map(|&(idx, logit)| {
                    decode_token(&self.tokenizer, idx as u32).map(|token| TopKEntry {
                        token,
                        token_id: idx as u32,
                        logit,
                    })
                })
                .collect();

            let (top_token, top_token_id, c_score) = if let Some(first) = top_k.first() {
                (first.token.clone(), first.token_id, first.logit)
            } else {
                (String::new(), 0, 0.0)
            };

            writer.write_record(&VectorRecord {
                id: format!("L{layer}_F{feat_idx}"),
                layer,
                feature: feat_idx,
                dim: vector.len(),
                vector,
                top_token,
                top_token_id,
                c_score,
                top_k,
            })?;
            count += 1;
        }

        writer.flush()?;
        Ok(count)
    }

    /// Extract FFN up vectors for a single layer.
    /// Same pattern as gate — `up_proj.weight` row per feature.
    pub fn extract_ffn_up(
        &self,
        layer: usize,
        config: &ExtractConfig,
        writer: &mut VectorWriter,
        callbacks: &mut dyn ExtractCallbacks,
    ) -> Result<usize, InferenceError> {
        let prefix = format!("layers.{layer}.mlp.");
        let w_up = self
            .weights
            .tensors
            .get(&format!("{prefix}up_proj.weight"))
            .ok_or_else(|| InferenceError::MissingTensor(format!("{prefix}up_proj.weight")))?;

        let n_features = w_up.shape()[0];
        callbacks.on_layer_start(COMPONENT_FFN_UP, layer, n_features);

        let logits = self.weights.embed.dot(&w_up.t());
        let progress_interval = (n_features / 20).max(1);
        let mut count = 0;

        for feat_idx in 0..n_features {
            if feat_idx % progress_interval == 0 {
                callbacks.on_progress(COMPONENT_FFN_UP, layer, feat_idx, n_features);
            }

            let vector: Vec<f32> = w_up.row(feat_idx).to_vec();
            let top_k_pairs = partial_top_k_column(&logits, feat_idx, config.top_k);
            let top_k: Vec<TopKEntry> = top_k_pairs
                .iter()
                .filter_map(|&(idx, logit)| {
                    decode_token(&self.tokenizer, idx as u32).map(|token| TopKEntry {
                        token,
                        token_id: idx as u32,
                        logit,
                    })
                })
                .collect();

            let (top_token, top_token_id, c_score) = if let Some(first) = top_k.first() {
                (first.token.clone(), first.token_id, first.logit)
            } else {
                (String::new(), 0, 0.0)
            };

            writer.write_record(&VectorRecord {
                id: format!("L{layer}_F{feat_idx}"),
                layer,
                feature: feat_idx,
                dim: vector.len(),
                vector,
                top_token,
                top_token_id,
                c_score,
                top_k,
            })?;
            count += 1;
        }

        writer.flush()?;
        Ok(count)
    }

    /// Extract attention OV circuit vectors for a single layer.
    ///
    /// For each KV head, computes OV = O_h @ V_h and stores the mean output
    /// direction (hidden-dim) — the average column of the OV matrix, which
    /// represents the head's typical write direction. Same dimensionality as
    /// FFN vectors, so HNSW indexes work uniformly.
    pub fn extract_attn_ov(
        &self,
        layer: usize,
        config: &ExtractConfig,
        writer: &mut VectorWriter,
        callbacks: &mut dyn ExtractCallbacks,
    ) -> Result<usize, InferenceError> {
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

        let head_dim = self.weights.head_dim;
        let hidden = self.weights.hidden_size;
        let num_kv_heads = w_v.shape()[0] / head_dim;
        callbacks.on_layer_start(COMPONENT_ATTN_OV, layer, num_kv_heads);

        let mut count = 0;

        for h in 0..num_kv_heads {
            callbacks.on_progress(COMPONENT_ATTN_OV, layer, h, num_kv_heads);

            let v_h = w_v.slice(ndarray::s![h * head_dim..(h + 1) * head_dim, ..]);
            let o_h = w_o.slice(ndarray::s![.., h * head_dim..(h + 1) * head_dim]);

            // OV circuit: O_h @ V_h → (hidden, hidden)
            let ov = o_h.dot(&v_h);

            // Mean output direction: average column of OV → (hidden,)
            let mut vector = vec![0.0f32; hidden];
            for col in 0..hidden {
                let mut sum = 0.0f32;
                for row in 0..hidden {
                    sum += ov[[row, col]];
                }
                vector[col] = sum / hidden as f32;
            }

            // Top-k: project vocab through OV, find most amplified
            let transformed = self.weights.embed.dot(&ov.t());
            let norms: Vec<f32> = (0..self.weights.vocab_size)
                .map(|i| {
                    let row = transformed.row(i);
                    row.iter().map(|x| x * x).sum::<f32>().sqrt()
                })
                .collect();

            let top_k_pairs = partial_top_k(&norms, config.top_k);
            let top_k: Vec<TopKEntry> = top_k_pairs
                .iter()
                .filter_map(|&(idx, logit)| {
                    decode_token(&self.tokenizer, idx as u32).map(|token| TopKEntry {
                        token,
                        token_id: idx as u32,
                        logit,
                    })
                })
                .collect();

            let (top_token, top_token_id, c_score) = if let Some(first) = top_k.first() {
                (first.token.clone(), first.token_id, first.logit)
            } else {
                (String::new(), 0, 0.0)
            };

            writer.write_record(&VectorRecord {
                id: format!("L{layer}_H{h}"),
                layer,
                feature: h,
                dim: vector.len(),
                vector,
                top_token,
                top_token_id,
                c_score,
                top_k,
            })?;
            count += 1;
        }

        writer.flush()?;
        Ok(count)
    }

    /// Extract attention Q/K mean direction vectors per head for a single layer.
    ///
    /// Each Q/K head's projection is (head_dim, hidden). We store the mean row
    /// as a hidden-dim vector — the average direction this head queries/keys in
    /// hidden space. Same dimensionality as FFN vectors for uniform HNSW indexing.
    pub fn extract_attn_qk(
        &self,
        layer: usize,
        _config: &ExtractConfig,
        writer: &mut VectorWriter,
        callbacks: &mut dyn ExtractCallbacks,
    ) -> Result<usize, InferenceError> {
        let prefix = format!("layers.{layer}.self_attn.");
        let w_q = self
            .weights
            .tensors
            .get(&format!("{prefix}q_proj.weight"))
            .ok_or_else(|| InferenceError::MissingTensor(format!("{prefix}q_proj.weight")))?;
        let w_k = self
            .weights
            .tensors
            .get(&format!("{prefix}k_proj.weight"))
            .ok_or_else(|| InferenceError::MissingTensor(format!("{prefix}k_proj.weight")))?;

        let head_dim = self.weights.head_dim;
        let hidden = self.weights.hidden_size;
        let num_q_heads = w_q.shape()[0] / head_dim;
        let num_kv_heads = w_k.shape()[0] / head_dim;
        let total = num_q_heads + num_kv_heads;
        callbacks.on_layer_start(COMPONENT_ATTN_QK, layer, total);

        let mut count = 0;

        // Q heads — mean row of each head's (head_dim, hidden) slice → (hidden,)
        for h in 0..num_q_heads {
            callbacks.on_progress(COMPONENT_ATTN_QK, layer, h, total);
            let head_slice = w_q.slice(ndarray::s![h * head_dim..(h + 1) * head_dim, ..]);
            let mut vector = vec![0.0f32; hidden];
            for col in 0..hidden {
                let mut sum = 0.0f32;
                for row in 0..head_dim {
                    sum += head_slice[[row, col]];
                }
                vector[col] = sum / head_dim as f32;
            }

            writer.write_record(&VectorRecord {
                id: format!("L{layer}_Q{h}"),
                layer,
                feature: h,
                dim: vector.len(),
                vector,
                top_token: String::new(),
                top_token_id: 0,
                c_score: 0.0,
                top_k: vec![],
            })?;
            count += 1;
        }

        // K heads — same approach
        for h in 0..num_kv_heads {
            callbacks.on_progress(COMPONENT_ATTN_QK, layer, num_q_heads + h, total);
            let head_slice = w_k.slice(ndarray::s![h * head_dim..(h + 1) * head_dim, ..]);
            let mut vector = vec![0.0f32; hidden];
            for col in 0..hidden {
                let mut sum = 0.0f32;
                for row in 0..head_dim {
                    sum += head_slice[[row, col]];
                }
                vector[col] = sum / head_dim as f32;
            }

            writer.write_record(&VectorRecord {
                id: format!("L{layer}_K{h}"),
                layer,
                feature: h,
                dim: vector.len(),
                vector,
                top_token: String::new(),
                top_token_id: 0,
                c_score: 0.0,
                top_k: vec![],
            })?;
            count += 1;
        }

        writer.flush()?;
        Ok(count)
    }

    /// Extract embedding vectors — one per vocab token.
    pub fn extract_embeddings(
        &self,
        _config: &ExtractConfig,
        writer: &mut VectorWriter,
        callbacks: &mut dyn ExtractCallbacks,
    ) -> Result<usize, InferenceError> {
        let vocab_size = self.weights.vocab_size;
        callbacks.on_layer_start(COMPONENT_EMBEDDINGS, 0, vocab_size);

        let progress_interval = (vocab_size / 20).max(1);
        let mut count = 0;

        for tok_id in 0..vocab_size {
            if tok_id % progress_interval == 0 {
                callbacks.on_progress(COMPONENT_EMBEDDINGS, 0, tok_id, vocab_size);
            }

            let vector: Vec<f32> = self.weights.embed.row(tok_id).to_vec();
            let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();

            let token = decode_token(&self.tokenizer, tok_id as u32).unwrap_or_default();

            writer.write_record(&VectorRecord {
                id: format!("T{tok_id}"),
                layer: 0,
                feature: tok_id,
                dim: vector.len(),
                vector,
                top_token: token,
                top_token_id: tok_id as u32,
                c_score: norm,
                top_k: vec![],
            })?;
            count += 1;
        }

        writer.flush()?;
        Ok(count)
    }

    /// Orchestrate extraction of all requested components across requested layers.
    ///
    /// Returns `None` for unimplemented components so the caller can decide
    /// how to report them (keeps eprintln out of core, same as weight_walker).
    pub fn extract_all(
        &self,
        config: &ExtractConfig,
        output_dir: &Path,
        resume: bool,
        callbacks: &mut dyn ExtractCallbacks,
    ) -> Result<ExtractSummary, InferenceError> {
        std::fs::create_dir_all(output_dir)?;
        let overall_start = std::time::Instant::now();
        let mut summaries = Vec::new();

        let layers: Vec<usize> = match &config.layers {
            Some(ls) => ls.clone(),
            None => (0..self.weights.num_layers).collect(),
        };

        for component in &config.components {
            // Embeddings are layer-independent — handle separately
            if component == COMPONENT_EMBEDDINGS {
                let file_path = output_dir.join(format!("{component}.vectors.jsonl"));
                if resume && file_path.exists() {
                    summaries.push(ComponentSummary {
                        component: component.clone(),
                        vectors_written: 0,
                        output_path: file_path,
                        elapsed_secs: 0.0,
                    });
                    continue;
                }
                let comp_start = std::time::Instant::now();
                callbacks.on_component_start(component, 1);
                let mut w = VectorWriter::create(&file_path)?;
                w.write_header(&VectorFileHeader {
                    _header: true,
                    component: component.clone(),
                    model: self.model_name.clone(),
                    dimension: self.weights.hidden_size,
                    extraction_date: current_date(),
                })?;
                let count = self.extract_embeddings(config, &mut w, callbacks)?;
                let elapsed_ms = comp_start.elapsed().as_secs_f64() * 1000.0;
                callbacks.on_layer_done(component, 0, count, elapsed_ms);
                callbacks.on_component_done(component, count);
                summaries.push(ComponentSummary {
                    component: component.clone(),
                    vectors_written: count,
                    output_path: file_path,
                    elapsed_secs: comp_start.elapsed().as_secs_f64(),
                });
                continue;
            }

            let file_path = output_dir.join(format!("{component}.vectors.jsonl"));
            let comp_start = std::time::Instant::now();

            // Determine completed layers for resume
            let completed = if resume {
                scan_completed_layers(&file_path)?
            } else {
                HashSet::new()
            };

            let pending: Vec<usize> = layers
                .iter()
                .filter(|l| !completed.contains(l))
                .copied()
                .collect();

            if pending.is_empty() {
                summaries.push(ComponentSummary {
                    component: component.clone(),
                    vectors_written: 0,
                    output_path: file_path,
                    elapsed_secs: 0.0,
                });
                continue;
            }

            callbacks.on_component_start(component, pending.len());

            // Open writer (append for resume, create for fresh)
            let (mut writer, _existing) = if resume && file_path.exists() {
                VectorWriter::append(&file_path)?
            } else {
                let mut w = VectorWriter::create(&file_path)?;
                w.write_header(&VectorFileHeader {
                    _header: true,
                    component: component.clone(),
                    model: self.model_name.clone(),
                    dimension: self.weights.hidden_size,
                    extraction_date: current_date(),
                })?;
                (w, 0)
            };

            let mut total_written = 0;

            for &layer in &pending {
                let layer_start = std::time::Instant::now();

                let count = match component.as_str() {
                    COMPONENT_FFN_DOWN => {
                        self.extract_ffn_down(layer, config, &mut writer, callbacks)?
                    }
                    COMPONENT_FFN_GATE => {
                        self.extract_ffn_gate(layer, config, &mut writer, callbacks)?
                    }
                    COMPONENT_FFN_UP => {
                        self.extract_ffn_up(layer, config, &mut writer, callbacks)?
                    }
                    COMPONENT_ATTN_OV => {
                        self.extract_attn_ov(layer, config, &mut writer, callbacks)?
                    }
                    COMPONENT_ATTN_QK => {
                        self.extract_attn_qk(layer, config, &mut writer, callbacks)?
                    }
                    _ => 0,
                };

                let elapsed_ms = layer_start.elapsed().as_secs_f64() * 1000.0;
                callbacks.on_layer_done(component, layer, count, elapsed_ms);
                total_written += count;
            }

            callbacks.on_component_done(component, total_written);

            summaries.push(ComponentSummary {
                component: component.clone(),
                vectors_written: total_written,
                output_path: file_path,
                elapsed_secs: comp_start.elapsed().as_secs_f64(),
            });
        }

        let total_vectors = summaries.iter().map(|s| s.vectors_written).sum();
        Ok(ExtractSummary {
            components: summaries,
            total_vectors,
            elapsed_secs: overall_start.elapsed().as_secs_f64(),
        })
    }
}
