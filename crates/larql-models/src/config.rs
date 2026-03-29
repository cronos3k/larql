//! Model architecture trait and shared types.
//!
//! Every model architecture implements `ModelArchitecture`. This trait
//! describes *what the model is* — tensor key patterns, norm behavior,
//! activation functions, scaling — without any compute dependencies.

/// Normalization type used by the model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormType {
    /// RMSNorm (Gemma, Llama)
    RmsNorm,
    /// Standard LayerNorm (GPT-2, BERT)
    LayerNorm,
}

/// Activation function used in the FFN.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    /// SiLU / Swish (Gemma, Llama)
    Silu,
    /// GELU (GPT-2, BERT)
    Gelu,
    /// GELU with tanh approximation
    GeluTanh,
    /// ReLU
    Relu,
}

/// Whether the FFN uses a gated architecture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FfnType {
    /// Gated: SiLU(x @ gate.T) * (x @ up.T) @ down.T (Gemma, Llama)
    Gated,
    /// Standard: activation(x @ up.T) @ down.T (GPT-2)
    Standard,
}

/// Model dimensions and architecture parameters, parsed from config.json.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_type: String,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub head_dim: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub vocab_size: Option<usize>,
    pub rope_base: f64,
    pub sliding_window: Option<usize>,
}

/// Architecture-specific behavior. Describes how a model is structured
/// without performing any computation.
pub trait ModelArchitecture: Send + Sync {
    /// Model family name (e.g., "gemma3", "llama").
    fn family(&self) -> &str;

    /// Parsed model configuration.
    fn config(&self) -> &ModelConfig;

    // ── Tensor key patterns ──

    /// Key prefix for a layer's tensors (e.g., "layers.5.").
    fn layer_prefix(&self, layer: usize) -> String {
        format!("layers.{layer}.")
    }

    /// Prefixes to strip from raw safetensors keys.
    /// Tried in order; first match wins.
    fn key_prefixes_to_strip(&self) -> &[&str] {
        &["language_model.model.", "model."]
    }

    /// Embedding tensor key (after prefix stripping).
    fn embed_key(&self) -> &str {
        "embed_tokens.weight"
    }

    /// Final norm weight key.
    fn final_norm_key(&self) -> &str {
        "norm.weight"
    }

    /// Attention weight keys for a layer.
    fn attn_q_key(&self, layer: usize) -> String {
        format!("{}self_attn.q_proj.weight", self.layer_prefix(layer))
    }
    fn attn_k_key(&self, layer: usize) -> String {
        format!("{}self_attn.k_proj.weight", self.layer_prefix(layer))
    }
    fn attn_v_key(&self, layer: usize) -> String {
        format!("{}self_attn.v_proj.weight", self.layer_prefix(layer))
    }
    fn attn_o_key(&self, layer: usize) -> String {
        format!("{}self_attn.o_proj.weight", self.layer_prefix(layer))
    }

    /// QK norm weight keys (None if model doesn't use QK norm).
    fn attn_q_norm_key(&self, layer: usize) -> Option<String> {
        let _ = layer;
        None
    }
    fn attn_k_norm_key(&self, layer: usize) -> Option<String> {
        let _ = layer;
        None
    }

    /// FFN weight keys for a layer.
    fn ffn_gate_key(&self, layer: usize) -> String {
        format!("{}mlp.gate_proj.weight", self.layer_prefix(layer))
    }
    fn ffn_up_key(&self, layer: usize) -> String {
        format!("{}mlp.up_proj.weight", self.layer_prefix(layer))
    }
    fn ffn_down_key(&self, layer: usize) -> String {
        format!("{}mlp.down_proj.weight", self.layer_prefix(layer))
    }

    /// Layer norm weight keys.
    fn input_layernorm_key(&self, layer: usize) -> String {
        format!("{}input_layernorm.weight", self.layer_prefix(layer))
    }
    fn post_attention_layernorm_key(&self, layer: usize) -> String {
        format!(
            "{}post_attention_layernorm.weight",
            self.layer_prefix(layer)
        )
    }
    fn pre_feedforward_layernorm_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}pre_feedforward_layernorm.weight",
            self.layer_prefix(layer)
        ))
    }
    fn post_feedforward_layernorm_key(&self, layer: usize) -> Option<String> {
        Some(format!(
            "{}post_feedforward_layernorm.weight",
            self.layer_prefix(layer)
        ))
    }

    // ── Behavior ──

    /// Norm type (RMSNorm vs LayerNorm).
    fn norm_type(&self) -> NormType {
        NormType::RmsNorm
    }

    /// Weight offset added during normalization.
    /// Gemma: 1.0 (weight = 1 + learned_weight), Llama: 0.0 (weight = learned_weight).
    fn norm_weight_offset(&self) -> f32 {
        0.0
    }

    /// Embedding scaling factor applied after lookup.
    /// Gemma: sqrt(hidden_size), Llama: 1.0.
    fn embed_scale(&self) -> f32 {
        1.0
    }

    /// Activation function for the FFN.
    fn activation(&self) -> Activation {
        Activation::Silu
    }

    /// FFN type (gated vs standard).
    fn ffn_type(&self) -> FfnType {
        FfnType::Gated
    }

    /// Whether this model has separate pre/post norms around attention and FFN
    /// (Gemma 2/3 style with 4 norms per layer) vs standard pre-norm only.
    fn has_post_norms(&self) -> bool {
        false
    }

    /// Whether this layer uses sliding window attention.
    fn is_sliding_window_layer(&self, _layer: usize) -> bool {
        false
    }

    /// Sliding window size (None = full attention).
    fn sliding_window_size(&self) -> Option<usize> {
        self.config().sliding_window
    }
}
