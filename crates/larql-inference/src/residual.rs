//! Layer normalization and residual stream operations.

use ndarray::Array2;

/// RMS norm with configurable weight offset.
/// offset=1.0 for Gemma (weight = 1 + learned), offset=0.0 for Llama (weight = learned).
pub fn rms_norm(x: &Array2<f32>, weight: Option<&Vec<f32>>, offset: f32) -> Array2<f32> {
    let eps = 1e-5f32;
    let (rows, cols) = (x.shape()[0], x.shape()[1]);
    let mut out = Array2::zeros((rows, cols));

    for i in 0..rows {
        let row = x.row(i);
        let rms = (row.iter().map(|v| v * v).sum::<f32>() / cols as f32 + eps).sqrt();
        for j in 0..cols {
            let w = match weight {
                Some(wt) => offset + wt[j],
                None => 1.0,
            };
            out[[i, j]] = row[j] / rms * w;
        }
    }
    out
}

/// Per-head RMS norm for Q/K projections with configurable weight offset.
pub fn rms_norm_heads(
    x: &Array2<f32>,
    weight: &[f32],
    num_heads: usize,
    head_dim: usize,
    offset: f32,
) -> Array2<f32> {
    let eps = 1e-5f32;
    let seq_len = x.shape()[0];
    let mut out = x.clone();

    for s in 0..seq_len {
        for h in 0..num_heads {
            let off = h * head_dim;
            let mut sq_sum = 0.0f32;
            for d in 0..head_dim {
                let v = x[[s, off + d]];
                sq_sum += v * v;
            }
            let rms = (sq_sum / head_dim as f32 + eps).sqrt();
            for d in 0..head_dim {
                out[[s, off + d]] = x[[s, off + d]] / rms * (offset + weight[d]);
            }
        }
    }
    out
}
