//! Auto-detect model architecture from config.json.

use std::path::Path;

use crate::config::{ModelArchitecture, ModelConfig};
use crate::gemma3::Gemma3Arch;
use crate::generic::GenericArch;

/// Error from model detection/config parsing.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Read config.json from a model directory and return the architecture.
pub fn detect_architecture(model_dir: &Path) -> Result<Box<dyn ModelArchitecture>, ModelError> {
    let config_path = model_dir.join("config.json");
    let config_json = if config_path.exists() {
        let text = std::fs::read_to_string(&config_path)?;
        serde_json::from_str::<serde_json::Value>(&text)?
    } else {
        serde_json::json!({})
    };

    Ok(detect_from_json(&config_json))
}

/// Detect architecture from an already-parsed config.json value.
pub fn detect_from_json(config: &serde_json::Value) -> Box<dyn ModelArchitecture> {
    let model_config = parse_model_config(config);
    let model_type = model_config.model_type.as_str();

    match model_type {
        t if t.starts_with("gemma3") => Box::new(Gemma3Arch::from_config(model_config)),
        t if t.starts_with("gemma2") => Box::new(Gemma3Arch::from_config(model_config)),
        _ => Box::new(GenericArch::from_config(model_config)),
    }
}

/// Parse ModelConfig from a config.json value.
/// Handles both top-level and nested text_config (multimodal models).
fn parse_model_config(config: &serde_json::Value) -> ModelConfig {
    let text_config = config.get("text_config").unwrap_or(config);

    // Detect model_type from text_config or top level.
    let model_type = text_config["model_type"]
        .as_str()
        .or_else(|| config["model_type"].as_str())
        .unwrap_or("")
        .to_string();

    // Pick defaults based on model type.
    let is_gemma = model_type.starts_with("gemma3") || model_type.starts_with("gemma2");
    let rope_default = if is_gemma { 1_000_000.0 } else { 10_000.0 };

    let num_layers = text_config["num_hidden_layers"].as_u64().unwrap_or(32) as usize;
    let hidden_size = text_config["hidden_size"].as_u64().unwrap_or(2048) as usize;
    let intermediate_size = text_config["intermediate_size"].as_u64().unwrap_or(8192) as usize;
    let head_dim = text_config["head_dim"].as_u64().unwrap_or(256) as usize;
    let num_q_heads = text_config["num_attention_heads"].as_u64().unwrap_or(8) as usize;
    let num_kv_heads = text_config["num_key_value_heads"].as_u64().unwrap_or(4) as usize;
    let rope_base = text_config["rope_theta"].as_f64().unwrap_or(rope_default);
    let vocab_size = text_config["vocab_size"].as_u64().map(|v| v as usize);
    let sliding_window = text_config["sliding_window"].as_u64().map(|v| v as usize);

    ModelConfig {
        model_type,
        num_layers,
        hidden_size,
        intermediate_size,
        head_dim,
        num_q_heads,
        num_kv_heads,
        vocab_size,
        rope_base,
        sliding_window,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_gemma3() {
        let config = serde_json::json!({
            "model_type": "gemma3",
            "text_config": {
                "model_type": "gemma3_text",
                "hidden_size": 2560,
                "num_hidden_layers": 34,
                "intermediate_size": 10240,
                "sliding_window": 1024
            }
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "gemma3");
        assert_eq!(arch.config().num_layers, 34);
        assert_eq!(arch.config().hidden_size, 2560);
        assert_eq!(arch.config().rope_base, 1_000_000.0);
        assert_eq!(arch.norm_weight_offset(), 1.0);
        assert_eq!(arch.embed_scale(), (2560.0f32).sqrt());
        assert!(arch.has_post_norms());
        assert!(arch.attn_q_norm_key(0).is_some());

        // Sliding window: layer 4 is sliding, layer 5 is full
        assert!(arch.is_sliding_window_layer(4));
        assert!(!arch.is_sliding_window_layer(5));
    }

    #[test]
    fn test_detect_unknown_defaults_to_generic() {
        let config = serde_json::json!({
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32
        });

        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "generic");
        assert_eq!(arch.config().hidden_size, 4096);
        assert_eq!(arch.config().rope_base, 10_000.0);
        assert_eq!(arch.norm_weight_offset(), 0.0);
        assert_eq!(arch.embed_scale(), 1.0);
        assert!(!arch.has_post_norms());
        assert!(arch.attn_q_norm_key(0).is_none());
    }

    #[test]
    fn test_tensor_keys() {
        let config = serde_json::json!({"model_type": "gemma3_text"});
        let arch = detect_from_json(&config);

        assert_eq!(arch.attn_q_key(5), "layers.5.self_attn.q_proj.weight");
        assert_eq!(arch.ffn_gate_key(10), "layers.10.mlp.gate_proj.weight");
        assert_eq!(
            arch.input_layernorm_key(0),
            "layers.0.input_layernorm.weight"
        );
        assert_eq!(arch.final_norm_key(), "norm.weight");
        assert_eq!(arch.embed_key(), "embed_tokens.weight");

        assert_eq!(
            arch.attn_q_norm_key(3),
            Some("layers.3.self_attn.q_norm.weight".to_string())
        );
    }

    #[test]
    fn test_empty_config() {
        let config = serde_json::json!({});
        let arch = detect_from_json(&config);
        assert_eq!(arch.family(), "generic");
        assert_eq!(arch.config().num_layers, 32);
    }
}
