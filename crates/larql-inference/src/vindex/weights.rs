//! Model weights serialization to/from .vindex directories.

use std::collections::HashMap;
use std::io::{BufWriter, Write};
use std::path::Path;

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::error::InferenceError;
use crate::model::ModelWeights;

use super::build::IndexBuildCallbacks;
use larql_vindex::config::{VindexConfig, VindexModelConfig};
use larql_vindex::{IndexLoadCallbacks, load_vindex_config};

/// Write all model weights (attention + FFN + norms) to a vindex directory.
///
/// Creates `model_weights.bin` containing all 2D tensors and 1D vectors
/// serialized contiguously, with a `weight_manifest.json` mapping keys to offsets.
/// Updates `index.json` to mark the vindex as inference-capable.
pub fn write_model_weights(
    weights: &ModelWeights,
    dir: &Path,
    callbacks: &mut dyn IndexBuildCallbacks,
) -> Result<(), InferenceError> {
    callbacks.on_stage("model_weights");
    let start = std::time::Instant::now();

    let bin_path = dir.join("model_weights.bin");
    let mut bin_file = BufWriter::new(std::fs::File::create(&bin_path)?);

    #[derive(Serialize)]
    struct WeightEntry {
        key: String,
        kind: String, // "tensor" (2D) or "vector" (1D)
        shape: Vec<usize>,
        offset: u64,
        length: u64,
    }

    let mut entries: Vec<WeightEntry> = Vec::new();
    let mut offset: u64 = 0;

    // Write 2D tensors (attention Q/K/V/O, FFN up/down — gate already in gate_vectors.bin)
    let arch = &*weights.arch;
    let num_layers = weights.num_layers;

    for layer in 0..num_layers {
        callbacks.on_layer_start("weights", layer, num_layers);

        // Attention weights
        for (suffix, key_fn) in &[
            ("q", arch.attn_q_key(layer)),
            ("k", arch.attn_k_key(layer)),
            ("v", arch.attn_v_key(layer)),
            ("o", arch.attn_o_key(layer)),
        ] {
            if let Some(tensor) = weights.tensors.get(key_fn) {
                let data = tensor.as_slice().unwrap();
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
                };
                bin_file.write_all(bytes)?;
                entries.push(WeightEntry {
                    key: key_fn.clone(),
                    kind: "tensor".into(),
                    shape: vec![tensor.shape()[0], tensor.shape()[1]],
                    offset,
                    length: bytes.len() as u64,
                });
                offset += bytes.len() as u64;
            }
            let _ = suffix;
        }

        // FFN up and down (gate is in gate_vectors.bin, but we need it here too for WalkFfn)
        for key in &[
            arch.ffn_gate_key(layer),
            arch.ffn_up_key(layer),
            arch.ffn_down_key(layer),
        ] {
            if let Some(tensor) = weights.tensors.get(key) {
                let data = tensor.as_slice().unwrap();
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
                };
                bin_file.write_all(bytes)?;
                entries.push(WeightEntry {
                    key: key.clone(),
                    kind: "tensor".into(),
                    shape: vec![tensor.shape()[0], tensor.shape()[1]],
                    offset,
                    length: bytes.len() as u64,
                });
                offset += bytes.len() as u64;
            }
        }

        callbacks.on_layer_done("weights", layer, 0.0);
    }

    // Write 1D vectors (all norms)
    for (key, vec) in &weights.vectors {
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(vec.as_ptr() as *const u8, vec.len() * 4)
        };
        bin_file.write_all(bytes)?;
        entries.push(WeightEntry {
            key: key.clone(),
            kind: "vector".into(),
            shape: vec![vec.len()],
            offset,
            length: bytes.len() as u64,
        });
        offset += bytes.len() as u64;
    }

    bin_file.flush()?;

    // Write manifest
    let manifest_json = serde_json::to_string_pretty(&entries)
        .map_err(|e| InferenceError::Parse(e.to_string()))?;
    std::fs::write(dir.join("weight_manifest.json"), manifest_json)?;

    // Update index.json
    let config_path = dir.join("index.json");
    let config_text = std::fs::read_to_string(&config_path)?;
    let mut config: VindexConfig = serde_json::from_str(&config_text)
        .map_err(|e| InferenceError::Parse(e.to_string()))?;

    config.has_model_weights = true;
    config.model_config = Some(VindexModelConfig {
        model_type: weights.arch.config().model_type.clone(),
        head_dim: weights.head_dim,
        num_q_heads: weights.num_q_heads,
        num_kv_heads: weights.num_kv_heads,
        rope_base: weights.rope_base,
        sliding_window: weights.arch.config().sliding_window,
    });

    let config_json = serde_json::to_string_pretty(&config)
        .map_err(|e| InferenceError::Parse(e.to_string()))?;
    std::fs::write(&config_path, config_json)?;

    callbacks.on_stage_done("model_weights", start.elapsed().as_secs_f64() * 1000.0);
    Ok(())
}

/// Load a full ModelWeights from a vindex directory.
///
/// Reads model_weights.bin + embeddings.bin + weight_manifest.json,
/// reconstructs the architecture from index.json config.
/// Returns a ModelWeights that can be used with the existing forward pass.
pub fn load_model_weights_from_vindex(
    dir: &Path,
    callbacks: &mut dyn IndexLoadCallbacks,
) -> Result<ModelWeights, InferenceError> {
    let config = load_vindex_config(dir)?;

    if !config.has_model_weights {
        return Err(InferenceError::Parse(
            "vindex does not contain model weights. Rebuild with: larql extract-index <model> -o <vindex> --include-weights".into(),
        ));
    }

    let model_cfg = config.model_config.as_ref().ok_or_else(|| {
        InferenceError::Parse("vindex missing model_config in index.json".into())
    })?;

    // Reconstruct architecture
    let _arch_config = larql_models::ModelConfig {
        model_type: model_cfg.model_type.clone(),
        num_layers: config.num_layers,
        hidden_size: config.hidden_size,
        intermediate_size: config.intermediate_size,
        head_dim: model_cfg.head_dim,
        num_q_heads: model_cfg.num_q_heads,
        num_kv_heads: model_cfg.num_kv_heads,
        vocab_size: Some(config.vocab_size),
        rope_base: model_cfg.rope_base,
        sliding_window: model_cfg.sliding_window,
        num_experts: None,
        num_experts_per_token: None,
        num_shared_experts: None,
        kv_lora_rank: None,
        q_lora_rank: None,
        rope_scaling: None,
        rope_local_base: None,
        embedding_multiplier: None,
        residual_multiplier: None,
        attention_multiplier: None,
        logits_scaling: None,
        attn_logit_softcapping: None,
        final_logit_softcapping: None,
        query_pre_attn_scalar: None,
    };

    let arch_json = serde_json::json!({
        "model_type": model_cfg.model_type,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_layers,
        "intermediate_size": config.intermediate_size,
        "head_dim": model_cfg.head_dim,
        "num_attention_heads": model_cfg.num_q_heads,
        "num_key_value_heads": model_cfg.num_kv_heads,
        "rope_theta": model_cfg.rope_base,
        "sliding_window": model_cfg.sliding_window,
        "vocab_size": config.vocab_size,
    });
    let arch = larql_models::detect_from_json(&arch_json);

    // Load embeddings
    callbacks.on_file_start("embeddings", &dir.join("embeddings.bin").display().to_string());
    let embed_bytes = std::fs::read(dir.join("embeddings.bin"))?;
    let embed_floats: Vec<f32> = unsafe {
        std::slice::from_raw_parts(
            embed_bytes.as_ptr() as *const f32,
            embed_bytes.len() / 4,
        )
    }
    .to_vec();
    let embed = Array2::from_shape_vec((config.vocab_size, config.hidden_size), embed_floats)
        .map_err(|e| InferenceError::Parse(e.to_string()))?;
    callbacks.on_file_done("embeddings", config.vocab_size, 0.0);

    // Load weight manifest
    callbacks.on_file_start("model_weights", &dir.join("model_weights.bin").display().to_string());
    let manifest_text = std::fs::read_to_string(dir.join("weight_manifest.json"))?;

    #[derive(Deserialize)]
    struct WeightEntry {
        key: String,
        kind: String,
        shape: Vec<usize>,
        offset: u64,
        length: u64,
    }

    let entries: Vec<WeightEntry> = serde_json::from_str(&manifest_text)
        .map_err(|e| InferenceError::Parse(e.to_string()))?;

    // Read binary weight data
    let bin_data = std::fs::read(dir.join("model_weights.bin"))?;
    let all_floats: &[f32] = unsafe {
        std::slice::from_raw_parts(
            bin_data.as_ptr() as *const f32,
            bin_data.len() / 4,
        )
    };

    let mut tensors: HashMap<String, Array2<f32>> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();

    for entry in &entries {
        let float_offset = entry.offset as usize / 4;
        let float_count = entry.length as usize / 4;
        let data = &all_floats[float_offset..float_offset + float_count];

        match entry.kind.as_str() {
            "tensor" => {
                let arr = Array2::from_shape_vec(
                    (entry.shape[0], entry.shape[1]),
                    data.to_vec(),
                )
                .map_err(|e| InferenceError::Parse(e.to_string()))?;
                tensors.insert(entry.key.clone(), arr);
            }
            "vector" => {
                vectors.insert(entry.key.clone(), data.to_vec());
            }
            _ => {}
        }
    }

    callbacks.on_file_done("model_weights", entries.len(), 0.0);

    let cfg = arch.config();
    let lm_head = embed.clone();
    Ok(ModelWeights {
        tensors,
        vectors,
        embed,
        lm_head,
        num_layers: cfg.num_layers,
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        vocab_size: config.vocab_size,
        head_dim: cfg.head_dim,
        num_q_heads: cfg.num_q_heads,
        num_kv_heads: cfg.num_kv_heads,
        rope_base: cfg.rope_base,
        arch,
    })
}

/// Find the tokenizer path near a model or vindex directory.
pub fn find_tokenizer_path(dir: &Path) -> Option<std::path::PathBuf> {
    let p = dir.join("tokenizer.json");
    if p.exists() {
        return Some(p);
    }
    if let Some(parent) = dir.parent() {
        let p = parent.join("tokenizer.json");
        if p.exists() {
            return Some(p);
        }
    }
    None
}
