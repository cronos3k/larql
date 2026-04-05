//! Profile the forward pass overhead — measures each phase separately
//! to find where the 155ms of unexplained overhead lives.
//!
//! Usage:
//!   cargo run --release -p larql-inference --example profile_overhead

use std::time::Instant;
use larql_inference::{predict, InferenceModel, WeightFfn, FfnBackend};
use larql_inference::forward::{dot_proj, apply_norm, forward_to_layer};
use ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_name = std::env::args().nth(1)
        .unwrap_or_else(|| "google/gemma-3-4b-it".to_string());

    println!("=== Forward Pass Overhead Profiler ===\n");

    let model = InferenceModel::load(&model_name)?;
    let weights = model.weights();
    let tokenizer = model.tokenizer();
    let num_layers = weights.num_layers;
    let hidden = weights.hidden_size;

    let prompt = "The capital of France is";
    let encoding = tokenizer.encode(prompt, true).map_err(|e| format!("{e}"))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    let seq_len = token_ids.len();
    println!("Prompt: \"{prompt}\" ({seq_len} tokens, hidden={hidden})\n");

    // Warmup
    let _ = predict(weights, tokenizer, &token_ids, 5);

    // ── Total forward pass ──
    let t0 = Instant::now();
    let n = 3;
    for _ in 0..n {
        let _ = predict(weights, tokenizer, &token_ids, 5);
    }
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
    println!("Total predict:    {total_ms:.1}ms\n");

    // ── Embedding ──
    let t0 = Instant::now();
    for _ in 0..100 {
        let scale = weights.arch.embed_scale();
        let mut h = Array2::<f32>::zeros((seq_len, hidden));
        for (i, &tok_id) in token_ids.iter().enumerate() {
            let row = weights.embed.row(tok_id as usize);
            for j in 0..hidden { h[[i, j]] = row[j] * scale; }
        }
        std::hint::black_box(&h);
    }
    let embed_ms = t0.elapsed().as_secs_f64() * 1000.0 / 100.0;
    println!("Embedding:        {embed_ms:.3}ms");

    // ── Input norm (one layer) ──
    let h = forward_to_layer(weights, &token_ids, 0);
    let norm_offset = weights.arch.norm_weight_offset();

    let t0 = Instant::now();
    for _ in 0..1000 {
        let _ = apply_norm(weights, &h, &weights.arch.input_layernorm_key(0), norm_offset);
    }
    let norm_ms = t0.elapsed().as_secs_f64() * 1000.0 / 1000.0;
    println!("RMS norm:         {norm_ms:.3}ms");

    // ── Q/K/V projection (one layer, one proj) ──
    let h_norm = apply_norm(weights, &h, &weights.arch.input_layernorm_key(0), norm_offset);
    let w_q = weights.tensors.get(&weights.arch.attn_q_key(0)).unwrap();

    let t0 = Instant::now();
    for _ in 0..100 {
        let _ = dot_proj(&h_norm, w_q);
    }
    let proj_ms = t0.elapsed().as_secs_f64() * 1000.0 / 100.0;
    println!("One QKV proj:     {proj_ms:.3}ms  (×4 per layer = {:.1}ms)", proj_ms * 4.0);

    // ── Residual add (h + &attn_projected) ──
    let other = Array2::<f32>::ones((seq_len, hidden));
    let t0 = Instant::now();
    for _ in 0..1000 {
        let _ = &h + &other;
    }
    let add_ms = t0.elapsed().as_secs_f64() * 1000.0 / 1000.0;
    println!("Residual add:     {add_ms:.3}ms  (×2 per layer = {:.3}ms)", add_ms * 2.0);

    // ── Array2 allocation ──
    let t0 = Instant::now();
    for _ in 0..1000 {
        let a = Array2::<f32>::zeros((seq_len, hidden));
        std::hint::black_box(&a);
    }
    let alloc_ms = t0.elapsed().as_secs_f64() * 1000.0 / 1000.0;
    println!("Array2 alloc:     {alloc_ms:.3}ms  (~14 per layer = {:.2}ms)", alloc_ms * 14.0);

    // ── FFN forward (one layer) ──
    let weight_ffn = WeightFfn { weights };
    let t0 = Instant::now();
    for _ in 0..10 {
        let _ = weight_ffn.forward(0, &h_norm);
    }
    let ffn_ms = t0.elapsed().as_secs_f64() * 1000.0 / 10.0;
    println!("FFN forward:      {ffn_ms:.1}ms");

    // ── GQA attention (one layer) ──
    let t0 = Instant::now();
    for _ in 0..10 {
        let _ = larql_inference::attention::run_attention_block(weights, &h, 0, false);
    }
    let attn_ms = t0.elapsed().as_secs_f64() * 1000.0 / 10.0;
    println!("Attention block:  {attn_ms:.1}ms  (proj + norm + RoPE + fused attn + residual)");

    // ── Full layer (attention + FFN) ──
    let t0 = Instant::now();
    for _ in 0..10 {
        let (h_post_attn, _, _) = larql_inference::attention::run_attention_block(weights, &h, 0, false).unwrap();
        let h_ffn = apply_norm(weights, &h_post_attn, &weights.arch.post_attention_layernorm_key(0), norm_offset);
        let _ = weight_ffn.forward(0, &h_ffn);
    }
    let layer_ms = t0.elapsed().as_secs_f64() * 1000.0 / 10.0;
    println!("Full layer:       {layer_ms:.1}ms  (attn block + norm + FFN, no residual bookkeeping)");

    // ── Logits projection ──
    let h_final = apply_norm(weights, &h, weights.arch.final_norm_key(), norm_offset);
    let t0 = Instant::now();
    for _ in 0..10 {
        let _ = dot_proj(
            &h_final.slice(ndarray::s![seq_len-1..seq_len, ..]),
            &weights.lm_head,
        );
    }
    let logits_ms = t0.elapsed().as_secs_f64() * 1000.0 / 10.0;
    println!("Logits proj:      {logits_ms:.1}ms");

    // ── Softmax + top-k ──
    let logits_raw = dot_proj(
        &h_final.slice(ndarray::s![seq_len-1..seq_len, ..]),
        &weights.lm_head,
    );
    let logits_row = logits_raw.row(0);
    let t0 = Instant::now();
    for _ in 0..100 {
        let max_logit = logits_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f64 = logits_row.iter().map(|l| ((l - max_logit) as f64).exp()).sum();
        let mut indexed: Vec<(usize, f32)> = logits_row.iter().copied().enumerate()
            .map(|(i, l)| (i, (((l - max_logit) as f64).exp() / exp_sum) as f32)).collect();
        indexed.select_nth_unstable_by(10, |a, b| b.1.partial_cmp(&a.1).unwrap());
        std::hint::black_box(&indexed);
    }
    let softmax_ms = t0.elapsed().as_secs_f64() * 1000.0 / 100.0;
    println!("Softmax+topk:     {softmax_ms:.1}ms  (262K vocab)");

    // ── All 34 FFN layers sequential (cache pressure test) ──
    let ffn_norms: Vec<Array2<f32>> = (0..num_layers).map(|layer| {
        let h_l = forward_to_layer(weights, &token_ids, layer);
        apply_norm(weights, &h_l, &weights.arch.post_attention_layernorm_key(layer), norm_offset)
    }).collect();

    // Warm
    for layer in 0..num_layers {
        let _ = weight_ffn.forward(layer, &ffn_norms[layer]);
    }

    let t0 = Instant::now();
    for _ in 0..3 {
        for layer in 0..num_layers {
            let _ = weight_ffn.forward(layer, &ffn_norms[layer]);
        }
    }
    let ffn_all_ms = t0.elapsed().as_secs_f64() * 1000.0 / 3.0;
    println!("\nFFN all 34 sequential: {ffn_all_ms:.1}ms  ({:.1}ms/layer)", ffn_all_ms / num_layers as f64);
    println!("FFN single (repeated): {ffn_ms:.1}ms  (cache-hot, same layer)");
    println!("Cache pressure ratio:  {:.1}x", (ffn_all_ms / num_layers as f64) / ffn_ms);

    // ── Summary ──
    let computed = embed_ms
        + (attn_ms + norm_ms + ffn_ms) * num_layers as f64
        + logits_ms + softmax_ms;
    let overhead = total_ms - computed;

    println!("\n--- Budget ---\n");
    println!("  Embedding:                      {embed_ms:.1}ms");
    println!("  Attention block × {num_layers}:       {:.1}ms  ({attn_ms:.1}ms/layer)", attn_ms * num_layers as f64);
    println!("  FFN norm × {num_layers}:              {:.1}ms  ({norm_ms:.3}ms/layer)", norm_ms * num_layers as f64);
    println!("  FFN forward × {num_layers}:           {:.1}ms  ({ffn_ms:.1}ms/layer)", ffn_ms * num_layers as f64);
    println!("  Logits:                         {logits_ms:.1}ms");
    println!("  Softmax+topk:                   {softmax_ms:.1}ms");
    println!("  ─────────────────────────────");
    println!("  Computed total:                 {computed:.1}ms");
    println!("  Measured total:                 {total_ms:.1}ms");
    println!("  Overhead:                       {overhead:.1}ms ({:.0}%)", overhead / total_ms * 100.0);

    let alloc_total = alloc_ms * 14.0 * num_layers as f64;
    let add_total = add_ms * 2.0 * num_layers as f64;
    println!("\n  Estimated allocation cost:      {alloc_total:.1}ms ({alloc_ms:.3}ms × 14 × {num_layers})");
    println!("  Estimated residual add cost:    {add_total:.1}ms ({add_ms:.3}ms × 2 × {num_layers})");

    println!("\n=== Done ===");
    Ok(())
}
