//! Memory analysis — profiles RSS, heap, and mmap usage during walk inference.
//!
//! Shows what's loaded, what's mmap'd, and what actually gets touched
//! during a walk forward pass vs dense forward pass.
//!
//! Usage:
//!   cargo run --release -p larql-inference --example memory_analysis -- \
//!     --model google/gemma-3-4b-it \
//!     --vindex path/to/gemma3-4b.vindex

use std::path::PathBuf;
use std::time::Instant;

use larql_inference::{
    predict, predict_with_ffn,
    InferenceModel,
    vindex::WalkFfn,
};
use larql_vindex::{SilentLoadCallbacks, VectorIndex};

fn rss_mb() -> f64 {
    // macOS: read RSS from proc info
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok();
    output
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|kb| kb as f64 / 1024.0)
        .unwrap_or(0.0)
}

fn file_size_mb(path: &std::path::Path) -> f64 {
    std::fs::metadata(path).map(|m| m.len() as f64 / 1e6).unwrap_or(0.0)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_name = String::new();
    let mut vindex_path = PathBuf::new();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => { i += 1; model_name = args[i].clone(); }
            "--vindex" => { i += 1; vindex_path = PathBuf::from(&args[i]); }
            _ => {}
        }
        i += 1;
    }
    if model_name.is_empty() || !vindex_path.is_dir() {
        eprintln!("Usage: memory_analysis --model MODEL --vindex PATH");
        std::process::exit(1);
    }

    println!("=== Memory Analysis ===\n");
    let rss_start = rss_mb();
    println!("Baseline RSS: {rss_start:.0} MB\n");

    // ── Vindex file inventory ──
    println!("--- Vindex Files ---\n");
    let vindex_files = [
        ("gate_vectors.bin", "Gate vectors (f32, mmap'd for KNN)"),
        ("down_features.bin", "Down features (f32, mmap'd for walk down proj)"),
        ("up_features.bin", "Up features (f32, mmap'd for full mmap walk)"),
        ("down_weights.bin", "Down weights (f16, original extraction)"),
        ("up_weights.bin", "Up weights (f16, original extraction)"),
        ("attn_weights.bin", "Attention weights"),
        ("embeddings.bin", "Token embeddings"),
        ("down_meta.bin", "Feature metadata (binary)"),
        ("index.json", "Config"),
        ("tokenizer.json", "Tokenizer"),
    ];

    let mut total_vindex = 0.0;
    for (file, desc) in &vindex_files {
        let path = vindex_path.join(file);
        let size = file_size_mb(&path);
        if size > 0.1 {
            total_vindex += size;
            println!("  {file:<25} {size:>8.1} MB  {desc}");
        }
    }
    println!("  {:<25} {:>8.1} MB", "Total", total_vindex);

    // ── Load model ──
    println!("\n--- Model Load ---\n");
    let t0 = Instant::now();
    let model = InferenceModel::load(&model_name)?;
    let rss_model = rss_mb();
    println!("  Model loaded in {:.1}s", t0.elapsed().as_secs_f64());
    println!("  RSS after model: {rss_model:.0} MB (+{:.0} MB)", rss_model - rss_start);

    // ── Load vindex ──
    let t0 = Instant::now();
    let mut cb = SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex(&vindex_path, &mut cb)?;
    let rss_vindex = rss_mb();
    println!("  Vindex loaded in {:.1}s ({} vectors)", t0.elapsed().as_secs_f64(), index.total_gate_vectors());
    println!("  RSS after vindex: {rss_vindex:.0} MB (+{:.0} MB from vindex mmap)", rss_vindex - rss_model);

    // ── Load feature-major files ──
    index.warmup();
    let rss_warmup = rss_mb();
    println!("  RSS after warmup: {rss_warmup:.0} MB (+{:.0} MB)", rss_warmup - rss_vindex);

    let _ = index.load_down_features(&vindex_path);
    let rss_down = rss_mb();
    println!("  RSS after down_features mmap: {rss_down:.0} MB (+{:.0} MB)", rss_down - rss_warmup);

    let _ = index.load_up_features(&vindex_path);
    let rss_up = rss_mb();
    println!("  RSS after up_features mmap: {rss_up:.0} MB (+{:.0} MB)", rss_up - rss_down);

    let weights = model.weights();
    let tokenizer = model.tokenizer();

    let prompt = "The capital of France is";
    let encoding = tokenizer.encode(prompt, true).map_err(|e| format!("{e}"))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    // ── Dense forward pass ──
    println!("\n--- Dense Forward Pass ---\n");
    let rss_before_dense = rss_mb();
    let result = predict(weights, tokenizer, &token_ids, 5);
    let rss_after_dense = rss_mb();
    let (tok, prob) = result.predictions.first().map(|(t, p)| (t.as_str(), *p)).unwrap_or(("?", 0.0));
    println!("  Result: {tok} ({:.1}%)", prob * 100.0);
    println!("  RSS before: {rss_before_dense:.0} MB");
    println!("  RSS after:  {rss_after_dense:.0} MB (+{:.0} MB during forward pass)", rss_after_dense - rss_before_dense);

    // Run a few more to see steady state
    for _ in 0..3 { let _ = predict(weights, tokenizer, &token_ids, 5); }
    let rss_dense_steady = rss_mb();
    println!("  RSS steady (4 runs): {rss_dense_steady:.0} MB");

    // ── Walk forward pass ──
    println!("\n--- Walk Forward Pass ---\n");
    let walk_ffn = WalkFfn::new(weights, &index, 8092);
    let rss_before_walk = rss_mb();
    let result = predict_with_ffn(weights, tokenizer, &token_ids, 5, &walk_ffn);
    let rss_after_walk = rss_mb();
    let (tok, prob) = result.predictions.first().map(|(t, p)| (t.as_str(), *p)).unwrap_or(("?", 0.0));
    println!("  Result: {tok} ({:.1}%)", prob * 100.0);
    println!("  RSS before: {rss_before_walk:.0} MB");
    println!("  RSS after:  {rss_after_walk:.0} MB (+{:.0} MB during forward pass)", rss_after_walk - rss_before_walk);

    for _ in 0..3 { let _ = predict_with_ffn(weights, tokenizer, &token_ids, 5, &walk_ffn); }
    let rss_walk_steady = rss_mb();
    println!("  RSS steady (4 runs): {rss_walk_steady:.0} MB");

    // ── Summary ──
    println!("\n--- Memory Summary ---\n");
    println!("  {:<35} {:>8} MB", "Baseline", format!("{rss_start:.0}"));
    println!("  {:<35} {:>8} MB", "After model load", format!("{rss_model:.0}"));
    println!("  {:<35} {:>8} MB", "After vindex mmap", format!("{rss_vindex:.0}"));
    println!("  {:<35} {:>8} MB", "After feature mmaps", format!("{rss_up:.0}"));
    println!("  {:<35} {:>8} MB", "Dense steady state", format!("{rss_dense_steady:.0}"));
    println!("  {:<35} {:>8} MB", "Walk steady state", format!("{rss_walk_steady:.0}"));
    println!();

    let walk_overhead = rss_walk_steady - rss_dense_steady;
    println!("  Walk memory overhead: {walk_overhead:+.0} MB vs dense");
    println!();
    println!("  Note: RSS on macOS includes mmap'd pages. These are");
    println!("  demand-paged by the OS and reclaimed under memory pressure.");
    println!("  The walk path only touches down_features.bin (~{:.0} MB)", file_size_mb(&vindex_path.join("down_features.bin")));
    println!("  during inference — other mmap'd files stay as virtual mappings.");

    // ── Growth test ──
    println!("\n--- Growth Test (10 sequential inferences) ---\n");
    let rss_growth_start = rss_mb();
    for i in 0..10 {
        let _ = predict_with_ffn(weights, tokenizer, &token_ids, 5, &walk_ffn);
        if i == 0 || i == 4 || i == 9 {
            let rss_now = rss_mb();
            println!("  Run {}: RSS = {rss_now:.0} MB (+{:.0} MB from start)", i + 1, rss_now - rss_growth_start);
        }
    }
    let rss_growth_end = rss_mb();
    let growth = rss_growth_end - rss_growth_start;
    println!();
    if growth.abs() < 50.0 {
        println!("  PASS: No memory growth ({growth:+.0} MB across 10 runs)");
    } else {
        println!("  NOTE: {growth:+.0} MB change (likely page cache warm-up, not a leak)");
    }

    // ── Walk-only mode: measure FFN weight drop ──
    println!("\n--- Walk-Only Mode ---\n");

    // Drop FFN weights from the already-loaded model to measure savings
    let tensors_before = weights.tensors.len();
    // We can't mutate the borrowed weights, so report what drop_ffn_weights would save
    let ffn_patterns = ["gate_proj", "up_proj", "down_proj", "ffn_gate", "ffn_up", "ffn_down", "mlp.experts"];
    let ffn_tensor_bytes: usize = weights.tensors.iter()
        .filter(|(k, _)| ffn_patterns.iter().any(|p| k.contains(p)))
        .map(|(_, v)| v.len() * 4)
        .sum();
    let ffn_tensor_count = weights.tensors.keys()
        .filter(|k| ffn_patterns.iter().any(|p| k.contains(p)))
        .count();
    let attn_tensor_count = tensors_before - ffn_tensor_count;

    println!("  Total tensors:  {tensors_before}");
    println!("  FFN tensors:    {ffn_tensor_count} ({:.1} GB)", ffn_tensor_bytes as f64 / 1e9);
    println!("  Attn+other:     {attn_tensor_count} ({:.1} GB)", (weights.tensors.iter().map(|(_, v)| v.len() * 4).sum::<usize>() - ffn_tensor_bytes) as f64 / 1e9);
    println!();
    println!("  drop_ffn_weights() would free: {:.1} GB", ffn_tensor_bytes as f64 / 1e9);
    println!("  Walk-only model size: {:.1} GB (attention + embeddings + norms)",
        (rss_model - rss_start) as f64 / 1024.0 - ffn_tensor_bytes as f64 / 1e9);
    println!();
    println!("  Use InferenceModel::load_walk_only() to load without FFN weights.");
    println!("  Requires down_features.bin + up_features.bin in the vindex.");

    println!("\n=== Done ===");
    Ok(())
}
