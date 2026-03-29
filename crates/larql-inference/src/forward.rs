//! Full transformer forward pass.
//!
//! Runs tokens through embedding → layers → final norm → logits.
//! Uses the ModelArchitecture trait for all model-specific behavior.

use ndarray::Array2;

use crate::attention::{apply_rope, gqa_attention};
use crate::ffn::{ffn_forward, ffn_forward_with_activation};
use crate::model::ModelWeights;
use crate::residual::{rms_norm, rms_norm_heads};

/// Result of a forward trace — residuals and optional sparse activations.
pub struct TraceResult {
    /// (layer, residual_vector) for each capture layer.
    pub residuals: Vec<(usize, Vec<f32>)>,
    /// (layer, top-K (feature_index, activation_magnitude)) for each capture layer.
    /// Only populated if capture_activations=true.
    pub activations: Vec<(usize, Vec<(usize, f32)>)>,
}

/// Prediction result from a full forward pass.
pub struct PredictResult {
    /// Top-k predicted tokens as (token_string, probability).
    pub predictions: Vec<(String, f64)>,
}

/// Embed token IDs with architecture-specific scaling.
fn embed_tokens(weights: &ModelWeights, token_ids: &[u32]) -> Array2<f32> {
    let seq_len = token_ids.len();
    let hidden = weights.hidden_size;
    let scale = weights.arch.embed_scale();

    let mut h = Array2::<f32>::zeros((seq_len, hidden));
    for (i, &tok_id) in token_ids.iter().enumerate() {
        let row = weights.embed.row(tok_id as usize);
        for j in 0..hidden {
            h[[i, j]] = row[j] * scale;
        }
    }
    h
}

/// Run a single transformer layer, returning the updated hidden state.
fn run_layer(
    weights: &ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    capture_activation: bool,
) -> Option<(Array2<f32>, Option<Array2<f32>>)> {
    let head_dim = weights.head_dim;
    let num_q = weights.num_q_heads;
    let num_kv = weights.num_kv_heads;
    let reps = num_q / num_kv;
    let scale = (head_dim as f64).powf(-0.5);
    let seq_len = h.shape()[0];
    let norm_offset = weights.arch.norm_weight_offset();
    let arch = &*weights.arch;

    // ── Attention ──
    let h_norm = rms_norm(
        h,
        weights.vectors.get(&arch.input_layernorm_key(layer)),
        norm_offset,
    );

    let w_q = weights.tensors.get(&arch.attn_q_key(layer))?;
    let w_k = weights.tensors.get(&arch.attn_k_key(layer)).unwrap();
    let w_v = weights.tensors.get(&arch.attn_v_key(layer)).unwrap();
    let w_o = weights.tensors.get(&arch.attn_o_key(layer)).unwrap();

    let q_full = h_norm.dot(&w_q.t());
    let k_full = h_norm.dot(&w_k.t());
    let v_full = h_norm.dot(&w_v.t());

    // QK norm (Gemma 3) — architecture tells us whether it exists.
    let q_normed = match arch
        .attn_q_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        Some(norm_w) => rms_norm_heads(&q_full, norm_w, num_q, head_dim, norm_offset),
        None => q_full,
    };
    let k_normed = match arch
        .attn_k_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        Some(norm_w) => rms_norm_heads(&k_full, norm_w, num_kv, head_dim, norm_offset),
        None => k_full,
    };

    let q_rope = apply_rope(&q_normed, num_q, head_dim, weights.rope_base);
    let k_rope = apply_rope(&k_normed, num_kv, head_dim, weights.rope_base);

    let attn_out = gqa_attention(
        &q_rope, &k_rope, &v_full, num_q, head_dim, reps, scale, seq_len,
    );
    let attn_projected = attn_out.dot(&w_o.t());

    // Post-attention: architecture decides whether there's a post-attention norm.
    let h_post_attn = if arch.has_post_norms() {
        let normed = rms_norm(
            &attn_projected,
            weights
                .vectors
                .get(&arch.post_attention_layernorm_key(layer)),
            norm_offset,
        );
        h + &normed
    } else {
        h + &attn_projected
    };

    // ── FFN ──
    // Pre-FFN norm: Gemma has pre_feedforward_layernorm, Llama uses post_attention_layernorm.
    let pre_ffn_key = if arch.has_post_norms() {
        arch.pre_feedforward_layernorm_key(layer)
    } else {
        Some(arch.post_attention_layernorm_key(layer))
    };
    let h_ffn = rms_norm(
        &h_post_attn,
        pre_ffn_key.and_then(|k| weights.vectors.get(&k)),
        norm_offset,
    );

    let w_gate = weights.tensors.get(&arch.ffn_gate_key(layer)).unwrap();
    let w_up = weights.tensors.get(&arch.ffn_up_key(layer)).unwrap();
    let w_down = weights.tensors.get(&arch.ffn_down_key(layer)).unwrap();

    let (ffn_out, activation) = if capture_activation {
        let (out, act) = ffn_forward_with_activation(&h_ffn, w_gate, w_up, w_down);
        (out, Some(act))
    } else {
        (ffn_forward(&h_ffn, w_gate, w_up, w_down), None)
    };

    // Post-FFN norm (Gemma has it, Llama doesn't).
    let h_out = if arch.has_post_norms() {
        let normed = rms_norm(
            &ffn_out,
            arch.post_feedforward_layernorm_key(layer)
                .and_then(|k| weights.vectors.get(&k)),
            norm_offset,
        );
        &h_post_attn + &normed
    } else {
        &h_post_attn + &ffn_out
    };

    Some((h_out, activation))
}

/// Run a forward pass through layers 0..=max_layer and return the
/// last-token residual at each requested capture layer.
pub fn capture_residuals(
    weights: &ModelWeights,
    token_ids: &[u32],
    capture_layers: &[usize],
) -> Vec<(usize, Vec<f32>)> {
    let trace = trace_forward(weights, token_ids, capture_layers, false, 0);
    trace.residuals
}

/// Run a forward pass and capture both residuals and sparse activations.
pub fn trace_forward(
    weights: &ModelWeights,
    token_ids: &[u32],
    capture_layers: &[usize],
    capture_activations: bool,
    activation_top_k: usize,
) -> TraceResult {
    let seq_len = token_ids.len();
    let max_layer = *capture_layers.iter().max().unwrap_or(&0);

    let mut h = embed_tokens(weights, token_ids);
    let mut results = Vec::new();
    let mut activations: Vec<(usize, Vec<(usize, f32)>)> = Vec::new();

    for layer in 0..=max_layer {
        let need_activation = capture_activations && capture_layers.contains(&layer);

        let (h_new, activation) = match run_layer(weights, &h, layer, need_activation) {
            Some(result) => result,
            None => continue,
        };
        h = h_new;

        if capture_layers.contains(&layer) {
            let last_row = h.row(seq_len - 1);
            results.push((layer, last_row.to_vec()));

            if let Some(act) = activation {
                let act_row = act.row(seq_len - 1);
                let mut indexed: Vec<(usize, f32)> = act_row.iter().copied().enumerate().collect();
                indexed.sort_unstable_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
                indexed.truncate(activation_top_k);
                activations.push((layer, indexed));
            }
        }
    }

    TraceResult {
        residuals: results,
        activations,
    }
}

/// Run a full forward pass and return the top-k next token predictions.
pub fn predict(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
) -> PredictResult {
    let seq_len = token_ids.len();
    let num_layers = weights.num_layers;
    let norm_offset = weights.arch.norm_weight_offset();

    let mut h = embed_tokens(weights, token_ids);

    for layer in 0..num_layers {
        h = match run_layer(weights, &h, layer, false) {
            Some((h_new, _)) => h_new,
            None => continue,
        };
    }

    // Final norm
    let h_final = rms_norm(
        &h,
        weights.vectors.get(weights.arch.final_norm_key()),
        norm_offset,
    );

    // Project last token against embeddings: logits = h_final[-1] @ embed.T
    let last = h_final.row(seq_len - 1);
    let mut logits: Vec<f32> = Vec::with_capacity(weights.vocab_size);
    for tok_id in 0..weights.vocab_size {
        let emb_row = weights.embed.row(tok_id);
        let dot: f32 = last.iter().zip(emb_row.iter()).map(|(a, b)| a * b).sum();
        logits.push(dot);
    }

    // Softmax
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|l| (l - max_logit).exp()).sum();
    let probs: Vec<f32> = logits
        .iter()
        .map(|l| (l - max_logit).exp() / exp_sum)
        .collect();

    // Top-k
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    let k = top_k.min(indexed.len());
    indexed.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.truncate(k);
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let predictions = indexed
        .into_iter()
        .filter_map(|(idx, prob)| {
            tokenizer
                .decode(&[idx as u32], true)
                .ok()
                .map(|s| (s.trim().to_string(), prob as f64))
        })
        .collect();

    PredictResult { predictions }
}
