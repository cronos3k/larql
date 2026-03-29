//! Attention computation — GQA, RoPE, causal masking.

use ndarray::Array2;

/// Apply Rotary Position Embeddings to Q or K.
/// x: (seq_len, num_heads * head_dim)
pub fn apply_rope(
    x: &Array2<f32>,
    num_heads: usize,
    head_dim: usize,
    rope_base: f64,
) -> Array2<f32> {
    let seq_len = x.shape()[0];
    let mut out = x.clone();

    let half_dim = head_dim / 2;
    let inv_freq: Vec<f64> = (0..half_dim)
        .map(|i| 1.0 / rope_base.powf(2.0 * i as f64 / head_dim as f64))
        .collect();

    for pos in 0..seq_len {
        for h in 0..num_heads {
            let offset = h * head_dim;
            for i in 0..half_dim {
                let theta = pos as f64 * inv_freq[i];
                let cos_t = theta.cos() as f32;
                let sin_t = theta.sin() as f32;

                let x0 = x[[pos, offset + i]];
                let x1 = x[[pos, offset + half_dim + i]];

                out[[pos, offset + i]] = x0 * cos_t - x1 * sin_t;
                out[[pos, offset + half_dim + i]] = x0 * sin_t + x1 * cos_t;
            }
        }
    }
    out
}

/// Grouped-query attention with causal masking.
///
/// q: (seq, num_q * head_dim), k: (seq, num_kv * head_dim), v: same as k
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    num_q: usize,
    head_dim: usize,
    reps: usize,
    scale: f64,
    seq_len: usize,
) -> Array2<f32> {
    let mut out = Array2::<f32>::zeros((seq_len, num_q * head_dim));

    for h in 0..num_q {
        let kv_h = h / reps;
        let q_off = h * head_dim;
        let kv_off = kv_h * head_dim;

        let mut scores = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..=i {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[[i, q_off + d]] * k[[j, kv_off + d]];
                }
                scores[i * seq_len + j] = dot * scale as f32;
            }
            for j in (i + 1)..seq_len {
                scores[i * seq_len + j] = -1e9;
            }
        }

        // Softmax per row
        for i in 0..seq_len {
            let row_start = i * seq_len;
            let max_val = scores[row_start..row_start + seq_len]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for j in 0..seq_len {
                scores[row_start + j] = (scores[row_start + j] - max_val).exp();
                sum += scores[row_start + j];
            }
            for j in 0..seq_len {
                scores[row_start + j] /= sum;
            }
        }

        // Weighted sum of V
        for i in 0..seq_len {
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for j in 0..seq_len {
                    val += scores[i * seq_len + j] * v[[j, kv_off + d]];
                }
                out[[i, q_off + d]] = val;
            }
        }
    }

    out
}
