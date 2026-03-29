//! Gemma 3 architecture — Google's multimodal model family.
//!
//! Key differences from standard Llama:
//! - Embedding scaled by sqrt(hidden_size)
//! - RMSNorm has +1 weight offset: out = (x / rms) * (1 + weight)
//! - QK normalization per-head (q_norm, k_norm weights)
//! - 4 norms per layer (pre/post attention, pre/post FFN)
//! - Sliding window attention on most layers (every Nth layer is full)
//! - rope_theta defaults to 1,000,000 (not in config.json, HF class default)

use crate::config::{ModelArchitecture, ModelConfig};

/// Gemma 3 sliding window pattern: every 6th layer (0-indexed: 5, 11, 17, ...)
/// uses full attention, the rest use sliding window.
const GEMMA3_SLIDING_WINDOW_PATTERN: usize = 6;

pub struct Gemma3Arch {
    config: ModelConfig,
}

impl Gemma3Arch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for Gemma3Arch {
    fn family(&self) -> &str {
        "gemma3"
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    // ── Gemma 3 has QK norm ──

    fn attn_q_norm_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}self_attn.q_norm.weight",
            self.layer_prefix(layer)
        ))
    }

    fn attn_k_norm_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}self_attn.k_norm.weight",
            self.layer_prefix(layer)
        ))
    }

    // ── Gemma-specific behavior ──

    fn norm_weight_offset(&self) -> f32 {
        1.0
    }

    fn embed_scale(&self) -> f32 {
        (self.config.hidden_size as f32).sqrt()
    }

    fn has_post_norms(&self) -> bool {
        true
    }

    fn is_sliding_window_layer(&self, layer: usize) -> bool {
        // Full attention on every Nth layer, sliding window on the rest.
        // Layer indices 5, 11, 17, 23, 29 are full attention (0-indexed).
        !(layer + 1).is_multiple_of(GEMMA3_SLIDING_WINDOW_PATTERN)
    }
}
