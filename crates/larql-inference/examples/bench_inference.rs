//! Benchmark the inference pipeline — component by component.
//!
//! Measures: embed, RMS norm, RoPE, attention, FFN, full layer, full forward pass.
//! Uses the actual model weights for realistic numbers.
//!
//! Run: cargo run --release -p larql-inference --example bench_inference

use std::time::Instant;

use larql_inference::attention::{apply_rope, gqa_attention};
use larql_inference::ffn::{ffn_forward, ffn_forward_with_activation};
use larql_inference::model::{load_model_dir, resolve_model_path};
use larql_inference::residual::{rms_norm, rms_norm_heads};
use larql_inference::{capture_residuals, predict, InferenceModel};
use ndarray::Array2;

fn bench<F: FnMut()>(name: &str, iters: usize, mut f: F) {
    // Warmup
    f();
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = start.elapsed();
    let per_iter_us = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;
    let per_iter_ms = per_iter_us / 1000.0;
    if per_iter_ms > 1.0 {
        println!("  {:<35} {:>8.2} ms  ({} iters)", name, per_iter_ms, iters);
    } else {
        println!("  {:<35} {:>8.1} us  ({} iters)", name, per_iter_us, iters);
    }
}

fn main() {
    let model_name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "google/gemma-3-4b-it".to_string());

    println!("=== larql-inference: Benchmark ===\n");

    // Load model
    println!("Loading model: {model_name}");
    let start = Instant::now();
    let model_path = resolve_model_path(&model_name).expect("model not found");
    let weights = load_model_dir(&model_path).expect("failed to load");
    println!("  Loaded in {:.1}s\n", start.elapsed().as_secs_f64());

    let hidden = weights.hidden_size;
    let head_dim = weights.head_dim;
    let num_q = weights.num_q_heads;
    let num_kv = weights.num_kv_heads;
    let reps = num_q / num_kv;
    let norm_offset = weights.arch.norm_weight_offset();
    let rope_base = weights.rope_base;

    // ── Component benchmarks with synthetic data ──
    println!("--- Component Benchmarks (seq_len=6, synthetic data) ---\n");

    let seq_len = 6;
    let x = Array2::<f32>::ones((seq_len, hidden)) * 0.01;

    // RMS Norm
    let norm_weight = weights
        .vectors
        .get(&weights.arch.input_layernorm_key(0))
        .cloned();
    bench("RMS norm (seq=6)", 1000, || {
        let _ = rms_norm(&x, norm_weight.as_ref(), norm_offset);
    });

    // RMS Norm Heads (QK norm)
    let qk_data = Array2::<f32>::ones((seq_len, num_q * head_dim)) * 0.01;
    if let Some(qk_weight) = weights
        .arch
        .attn_q_norm_key(0)
        .and_then(|k| weights.vectors.get(&k))
    {
        bench("QK head norm (seq=6)", 1000, || {
            let _ = rms_norm_heads(&qk_data, qk_weight, num_q, head_dim, norm_offset);
        });
    }

    // RoPE
    bench("RoPE (seq=6)", 1000, || {
        let _ = apply_rope(&qk_data, num_q, head_dim, rope_base);
    });

    // Attention
    let q = Array2::<f32>::ones((seq_len, num_q * head_dim)) * 0.01;
    let k = Array2::<f32>::ones((seq_len, num_kv * head_dim)) * 0.01;
    let v = k.clone();
    let scale = (head_dim as f64).powf(-0.5);
    bench("GQA attention (seq=6)", 100, || {
        let _ = gqa_attention(&q, &k, &v, num_q, head_dim, reps, scale, seq_len);
    });

    // FFN
    let w_gate = weights.tensors.get(&weights.arch.ffn_gate_key(0)).unwrap();
    let w_up = weights.tensors.get(&weights.arch.ffn_up_key(0)).unwrap();
    let w_down = weights.tensors.get(&weights.arch.ffn_down_key(0)).unwrap();
    bench("FFN forward (seq=6)", 100, || {
        let _ = ffn_forward(&x, w_gate, w_up, w_down);
    });

    bench("FFN forward + activation (seq=6)", 100, || {
        let _ = ffn_forward_with_activation(&x, w_gate, w_up, w_down);
    });

    // ── Full forward pass benchmarks ──
    println!("\n--- Full Forward Pass (real weights) ---\n");

    let model = InferenceModel::load(&model_name).expect("failed to load model");
    let prompt = "The capital of France is";
    let encoding = model
        .tokenizer()
        .encode(prompt, true)
        .expect("tokenize failed");
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    println!("  Prompt: \"{prompt}\" ({} tokens)\n", token_ids.len());

    // Full predict
    bench("predict (top-10)", 3, || {
        let _ = predict(model.weights(), model.tokenizer(), &token_ids, 10);
    });

    // Residual capture at select layers
    bench("capture_residuals (4 layers)", 3, || {
        let _ = capture_residuals(model.weights(), &token_ids, &[0, 16, 25, 33]);
    });

    // Capture all layers
    let all_layers: Vec<usize> = (0..weights.num_layers).collect();
    bench("capture_residuals (all 34 layers)", 3, || {
        let _ = capture_residuals(model.weights(), &token_ids, &all_layers);
    });

    // ── Scaling: different sequence lengths ──
    println!("\n--- Sequence Length Scaling ---\n");
    for seq in [1, 6, 12, 24] {
        let ids: Vec<u32> = token_ids.iter().copied().cycle().take(seq).collect();
        let label = format!("predict (seq={})", seq);
        bench(&label, 3, || {
            let _ = predict(model.weights(), model.tokenizer(), &ids, 5);
        });
    }

    // ── Throughput summary ──
    println!("\n--- Throughput ---\n");
    let start = Instant::now();
    let n_runs = 10;
    for _ in 0..n_runs {
        let _ = predict(model.weights(), model.tokenizer(), &token_ids, 1);
    }
    let elapsed = start.elapsed();
    let per_query = elapsed.as_secs_f64() / n_runs as f64;
    println!(
        "  {:.1} queries/sec ({:.0}ms/query, {} layers, {} tokens)",
        1.0 / per_query,
        per_query * 1000.0,
        weights.num_layers,
        token_ids.len()
    );
}
