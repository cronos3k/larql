//! Accuracy test: Q4 attention weight quantization error on real Gemma 3 4B.
//!
//! Measures: Q4 quantization RMSE on attention weights, weight statistics.
//! The Q4 pipeline benchmark uses synthetic weights — this validates real weights.
//!
//! Usage:
//!   cargo run --release -p larql-inference --example test_q4_accuracy

extern crate blas_src;

use larql_inference::{InferenceModel, predict};
use larql_models::quant::ggml::{quantize_q4_0, dequantize_q4_0};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = InferenceModel::load("google/gemma-3-4b-it")?;
    let weights = model.weights();
    let tokenizer = model.tokenizer();
    let num_layers = weights.num_layers;

    println!("=== Q4 Attention Weight Analysis ===\n");

    // ── 1. Quantization error across all attention weights ──
    let mut total_max_error = 0.0f32;
    let mut total_rmse = 0.0f64;
    let mut count = 0;
    let mut total_elements = 0u64;

    for layer in 0..num_layers {
        let arch = &*weights.arch;
        let keys = [
            arch.attn_q_key(layer),
            arch.attn_k_key(layer),
            arch.attn_v_key(layer),
            arch.attn_o_key(layer),
        ];

        for key in &keys {
            if let Some(w) = weights.tensors.get(key) {
                let data = w.as_slice().unwrap();
                if data.len() % 32 != 0 { continue; }

                let q4 = quantize_q4_0(data);
                let recon = dequantize_q4_0(&q4, data.len()).unwrap();

                let mut layer_rmse = 0.0f64;
                for i in 0..data.len() {
                    let err = (data[i] - recon[i]).abs();
                    if err > total_max_error { total_max_error = err; }
                    layer_rmse += (err as f64) * (err as f64);
                }
                layer_rmse = (layer_rmse / data.len() as f64).sqrt();
                total_rmse += layer_rmse;
                total_elements += data.len() as u64;
                count += 1;
            }
        }
    }

    let avg_rmse = total_rmse / count.max(1) as f64;
    println!("  Attention weights ({count} matrices, {total_elements} total elements):");
    println!("    Max absolute error: {total_max_error:.6}");
    println!("    Average RMSE:       {avg_rmse:.6}");

    // ── 2. Weight statistics ──
    println!("\n  Weight statistics (sample layers):");
    for &layer in &[0, 13, 33] {
        if layer >= num_layers { continue; }
        let key = weights.arch.attn_q_key(layer);
        if let Some(w) = weights.tensors.get(&key) {
            let data = w.as_slice().unwrap();
            let min_v = data.iter().copied().fold(f32::INFINITY, f32::min);
            let max_v = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
            let std: f32 = (data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / data.len() as f32).sqrt();
            println!("    L{layer} Q proj {:?}: range=[{min_v:.4},{max_v:.4}] mean={mean:.6} std={std:.6} err/std={:.4}",
                w.shape(), total_max_error / std);
        }
    }

    // ── 3. Dense baseline predictions ──
    println!("\n  Dense f32 predictions:");
    let prompts = [
        "The capital of France is",
        "The language spoken in Japan is",
        "Albert Einstein was born in",
        "Python is a programming",
    ];
    for prompt in &prompts {
        let encoding = tokenizer.encode(*prompt, true).map_err(|e| format!("{e}"))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();
        let result = predict(weights, tokenizer, &token_ids, 3);
        let preds: Vec<String> = result.predictions.iter()
            .map(|(t, p)| format!("{t} ({:.1}%)", p * 100.0))
            .collect();
        println!("    \"{prompt}\" → {}", preds.join(", "));
    }

    // ── 4. Summary ──
    println!("\n  Q4 impact assessment:");
    let _q4_snr = avg_rmse / total_max_error as f64;
    println!("    At RMSE {avg_rmse:.6}, the Q4 error is {:.1}% of weight max",
        total_max_error as f64 / weights.tensors.get(&weights.arch.attn_q_key(0))
            .map(|w| w.as_slice().unwrap().iter().map(|v| v.abs()).fold(0.0f32, f32::max))
            .unwrap_or(1.0) as f64 * 100.0);
    println!("    llama.cpp uses Q4_K_M (per-group scaling) which has ~2× lower RMSE");
    println!("    For factual queries (strong top-1 signal), Q4_0 should be sufficient");
    println!("    For nuanced queries, Q8 attention may be needed as fallback");

    println!("\n=== Done ===");
    Ok(())
}
