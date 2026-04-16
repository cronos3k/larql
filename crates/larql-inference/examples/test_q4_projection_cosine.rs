//! Quick cosine similarity test: Q4 vs f32 attention projections on real weights.
//! If cosine > 0.99, attention routing survives Q4. If < 0.95, need Q8.
//!
//! Usage:
//!   cargo run --release -p larql-inference --example test_q4_projection_cosine

#[cfg(unix)]
extern crate blas_src;

use larql_inference::{InferenceModel, forward::forward_to_layer};
use larql_models::quant::ggml::{quantize_q4_0, dequantize_q4_0};

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-12 || nb < 1e-12 { return 0.0; }
    dot / (na * nb)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = InferenceModel::load("google/gemma-3-4b-it")?;
    let weights = model.weights();
    let tokenizer = model.tokenizer();

    println!("=== Q4 vs f32 Projection Cosine Similarity ===\n");

    let prompt = "The capital of France is";
    let encoding = tokenizer.encode(prompt, true).map_err(|e| format!("{e}"))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    println!("Prompt: \"{prompt}\" ({} tokens)\n", token_ids.len());
    println!("{:>5} {:>8} {:>8} {:>8} {:>8}", "Layer", "Q cos", "K cos", "V cos", "O cos");

    for &layer in &[0, 5, 10, 13, 15, 20, 25, 30, 33] {
        if layer >= weights.num_layers { continue; }

        // Get the hidden state at this layer
        let h = forward_to_layer(weights, &token_ids, layer);
        let last_row = h.row(h.shape()[0] - 1);
        let x = last_row.as_slice().unwrap();

        let arch = &*weights.arch;
        let mut cosines = Vec::new();

        for key in &[
            arch.attn_q_key(layer),
            arch.attn_k_key(layer),
            arch.attn_v_key(layer),
            arch.attn_o_key(layer),
        ] {
            if let Some(w) = weights.tensors.get(key) {
                let w_data = w.as_slice().unwrap();
                let rows = w.shape()[0];
                let cols = w.shape()[1];

                // f32 projection: x @ w^T
                let mut f32_result = vec![0.0f32; rows];
                for r in 0..rows {
                    let mut dot = 0.0f32;
                    for c in 0..cols {
                        dot += x[c] * w_data[r * cols + c];
                    }
                    f32_result[r] = dot;
                }

                // Q4 projection: quantize w, dequant, project
                if w_data.len() % 32 == 0 {
                    let q4 = quantize_q4_0(w_data);
                    let w_recon = dequantize_q4_0(&q4, w_data.len()).unwrap();

                    let mut q4_result = vec![0.0f32; rows];
                    for r in 0..rows {
                        let mut dot = 0.0f32;
                        for c in 0..cols {
                            dot += x[c] * w_recon[r * cols + c];
                        }
                        q4_result[r] = dot;
                    }

                    let cos = cosine(&f32_result, &q4_result);
                    cosines.push(cos);
                } else {
                    cosines.push(0.0);
                }
            } else {
                cosines.push(0.0);
            }
        }

        println!("  L{layer:2}  {:.4}   {:.4}   {:.4}   {:.4}",
            cosines[0], cosines[1], cosines[2], cosines[3]);
    }

    println!("\n  > 0.99 = safe for Q4,  < 0.95 = need Q8\n");

    // Q8 V projection — should fix the low V cosines
    println!("  Q8 V projection (should be > 0.999):");
    for &layer in &[0, 10, 13, 15, 20, 33] {
        if layer >= weights.num_layers { continue; }
        let h = forward_to_layer(weights, &token_ids, layer);
        let last_row = h.row(h.shape()[0] - 1);
        let x = last_row.as_slice().unwrap();

        let key = weights.arch.attn_v_key(layer);
        if let Some(w) = weights.tensors.get(&key) {
            let w_data = w.as_slice().unwrap();
            let rows = w.shape()[0];
            let cols = w.shape()[1];

            // f32 reference
            let mut f32_result = vec![0.0f32; rows];
            for r in 0..rows {
                for c in 0..cols { f32_result[r] += x[c] * w_data[r * cols + c]; }
            }

            // Q8
            let (w_q8, w_scales) = larql_compute::cpu::ops::q8_matvec::quantize_weights_q8(w_data, rows, cols);
            let (x_q8, x_scales) = larql_compute::cpu::ops::q4_common::quantize_to_q8(x);
            let q8_result = larql_compute::cpu::ops::q8_matvec::dispatch(
                &w_q8, &w_scales, &x_q8, &x_scales, rows, cols,
            );

            let cos = cosine(&f32_result, &q8_result);
            let status = if cos > 0.999 { "✓" } else if cos > 0.99 { "~" } else { "✗" };
            println!("    L{layer:2} V (Q8): {cos:.4} {status}");
        }
    }

    println!("\n=== Done ===");
    Ok(())
}
