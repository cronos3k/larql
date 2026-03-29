//! Demonstrate the inference engine — forward pass from safetensors weights.
//!
//! Loads a model, tokenizes a prompt, runs the full transformer forward pass,
//! and shows top-k next-token predictions. Also demonstrates residual capture.
//!
//! Requires a model in the HuggingFace cache (e.g. google/gemma-3-4b-it).
//!
//! Run: cargo run --release -p larql-inference --example inference_demo

use std::time::Instant;

use larql_inference::{capture_residuals, predict, InferenceModel};

fn main() {
    let model_name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "google/gemma-3-4b-it".to_string());

    println!("=== larql-inference: Inference Demo ===\n");

    // ── Load model ──
    println!("Loading model: {model_name}");
    let start = Instant::now();
    let model = InferenceModel::load(&model_name).expect("failed to load model");
    println!(
        "  {} layers, hidden_size={}, vocab_size={} ({:.1}s)\n",
        model.num_layers(),
        model.hidden_size(),
        model.weights().vocab_size,
        start.elapsed().as_secs_f64()
    );

    // ── Architecture info ──
    let arch = &model.weights().arch;
    println!("Architecture: {}", arch.family());
    println!("  Norm offset:    {}", arch.norm_weight_offset());
    println!("  Embed scale:    {:.2}", arch.embed_scale());
    println!("  RoPE base:      {:.0}", arch.config().rope_base);
    println!("  Has post norms: {}", arch.has_post_norms());
    println!("  Has QK norm:    {}\n", arch.attn_q_norm_key(0).is_some());

    // ── Prediction demo ──
    let prompts = [
        "The capital of France is",
        "The capital of Japan is",
        "Water freezes at",
        "The largest planet in our solar system is",
    ];

    println!("=== Predictions ===\n");
    for prompt in &prompts {
        let encoding = model
            .tokenizer()
            .encode(*prompt, true)
            .expect("tokenize failed");
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        let start = Instant::now();
        let result = predict(model.weights(), model.tokenizer(), &token_ids, 5);
        let elapsed = start.elapsed();

        println!("  \"{prompt}\"");
        println!(
            "    {} tokens, {:.2}s",
            token_ids.len(),
            elapsed.as_secs_f64()
        );
        for (i, (token, prob)) in result.predictions.iter().enumerate() {
            println!("    {:2}. {:15} ({:.2}%)", i + 1, token, prob * 100.0);
        }
        println!();
    }

    // ── Residual capture demo ──
    println!("=== Residual Capture ===\n");
    let prompt = "The capital of France is";
    let encoding = model
        .tokenizer()
        .encode(prompt, true)
        .expect("tokenize failed");
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    let capture_layers = vec![0, 16, 25, 33];
    let start = Instant::now();
    let residuals = capture_residuals(model.weights(), &token_ids, &capture_layers);
    let elapsed = start.elapsed();

    println!("  Prompt: \"{prompt}\"");
    println!(
        "  Captured {} layers in {:.2}s\n",
        residuals.len(),
        elapsed.as_secs_f64()
    );

    for (layer, residual) in &residuals {
        let norm: f32 = residual.iter().map(|v| v * v).sum::<f32>().sqrt();
        let max_val = residual.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min_val = residual.iter().copied().fold(f32::INFINITY, f32::min);
        println!(
            "  Layer {:2}: dim={}, norm={:.2}, range=[{:.4}, {:.4}]",
            layer,
            residual.len(),
            norm,
            min_val,
            max_val,
        );
    }
}
