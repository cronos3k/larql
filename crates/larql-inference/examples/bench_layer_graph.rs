//! Production pipeline benchmark: cache L0-12 + mmap walk L13-33 via LayerGraph.
//!
//! Tests:
//!   1. Dense baseline (predict)
//!   2. LayerGraph dense (verify match)
//!   3. Cache+Dense (skip L0-12)
//!   4. Cache+Walk (skip L0-12, mmap FFN L13-33)
//!   5. Walk only (mmap FFN all layers)
//!
//! Usage:
//!   cargo run --release -p larql-inference --example bench_layer_graph -- \
//!     --vindex output/gemma3-4b-v2.vindex

use std::time::Instant;

use larql_inference::{
    predict, predict_with_graph, predict_with_graph_vindex_logits,
    InferenceModel, WeightFfn, WalkLayerGraph,
    CachedLayerGraph, build_adaptive_graph, default_backend,
};
use larql_inference::vindex::WalkFfn;
use larql_vindex::{SilentLoadCallbacks, VectorIndex};

fn bench(
    weights: &larql_inference::ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    graph: &dyn larql_inference::LayerGraph,
    n: usize,
) -> (String, f64, f64) {
    let _ = predict_with_graph(weights, tokenizer, token_ids, 5, graph);
    let t0 = Instant::now();
    for _ in 0..n { let _ = predict_with_graph(weights, tokenizer, token_ids, 5, graph); }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let r = predict_with_graph(weights, tokenizer, token_ids, 5, graph);
    let (tok, prob) = r.predictions.first()
        .map(|(t, p)| (t.clone(), *p)).unwrap_or_default();
    (tok, prob, ms)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut vindex_path = std::path::PathBuf::from("output/gemma3-4b-v2.vindex");
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--vindex" { i += 1; vindex_path = std::path::PathBuf::from(&args[i]); }
        i += 1;
    }

    let model = InferenceModel::load("google/gemma-3-4b-it")?;
    let weights = model.weights();
    let tokenizer = model.tokenizer();
    let num_layers = weights.num_layers;

    let mut cb = SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex(&vindex_path, &mut cb)?;
    index.load_down_features(&vindex_path)?;
    index.load_up_features(&vindex_path)?;
    index.load_lm_head(&vindex_path)?;
    match index.load_interleaved(&vindex_path) {
        Ok(()) => print!("interleaved "),
        Err(_) => {}
    }
    match index.load_interleaved_q4(&vindex_path) {
        Ok(()) => print!("Q4 "),
        Err(_) => {}
    }
    println!("lm_head (vocab={})\n", index.vocab_size);

    let dense_ffn = WeightFfn { weights };
    let walk_ffn_cpu = WalkFfn::new(weights, &index, 8092);
    let gpu_be = default_backend();
    let walk_ffn_gpu = WalkFfn::new_with_backend(weights, &index, 8092, &*gpu_be);

    let n = 3;

    let prompt = "The capital of France is";
    let encoding = tokenizer.encode(prompt, true).map_err(|e| format!("{e}"))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    println!("=== Production Pipeline Benchmark ===");
    println!("Prompt: \"{prompt}\" ({} tokens)", token_ids.len());
    println!("Backend: {}\n", gpu_be.name());

    // 1. Dense baseline (no LayerGraph)
    let _ = predict(weights, tokenizer, &token_ids, 5);
    let t0 = Instant::now();
    for _ in 0..n { let _ = predict(weights, tokenizer, &token_ids, 5); }
    let dense_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let dense_r = predict(weights, tokenizer, &token_ids, 5);
    let (dense_tok, dense_prob) = dense_r.predictions.first()
        .map(|(t, p)| (t.clone(), *p)).unwrap_or_default();

    // 2. Cache+Walk (CPU) — FFN through CPU BLAS
    let walk_cpu_graph = WalkLayerGraph { ffn: &walk_ffn_cpu, backend: None };
    let cached_layers: Vec<usize> = (0..=12).collect();
    let cache = CachedLayerGraph::build(weights, &token_ids, &cached_layers, &dense_ffn);
    let cw_cpu = build_adaptive_graph(&cache, &walk_cpu_graph, num_layers, &(0..=12));
    let (cw_cpu_tok, _, cw_cpu_ms) = bench(weights, tokenizer, &token_ids, &cw_cpu, n);

    // 3. Cache+Walk (Metal Q4 FFN, CPU attention)
    let walk_gpu_graph = WalkLayerGraph { ffn: &walk_ffn_gpu, backend: None };
    let cw_gpu = build_adaptive_graph(&cache, &walk_gpu_graph, num_layers, &(0..=12));
    let (cw_gpu_tok, _, cw_gpu_ms) = bench(weights, tokenizer, &token_ids, &cw_gpu, n);

    // 4. Full pipeline (CPU): Cache+Walk(CPU)+Vindex logits
    let _ = predict_with_graph_vindex_logits(weights, tokenizer, &token_ids, 5, &cw_cpu, &index);
    let t0 = Instant::now();
    for _ in 0..n { let _ = predict_with_graph_vindex_logits(weights, tokenizer, &token_ids, 5, &cw_cpu, &index); }
    let full_cpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let full_cpu_r = predict_with_graph_vindex_logits(weights, tokenizer, &token_ids, 5, &cw_cpu, &index);
    let (full_cpu_tok, _) = full_cpu_r.predictions.first()
        .map(|(t, p)| (t.clone(), *p)).unwrap_or_default();

    // 5. Full pipeline (Metal Q4 FFN, CPU attention, vindex logits)
    let _ = predict_with_graph_vindex_logits(weights, tokenizer, &token_ids, 5, &cw_gpu, &index);
    let t0 = Instant::now();
    for _ in 0..n { let _ = predict_with_graph_vindex_logits(weights, tokenizer, &token_ids, 5, &cw_gpu, &index); }
    let full_gpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let full_gpu_r = predict_with_graph_vindex_logits(weights, tokenizer, &token_ids, 5, &cw_gpu, &index);
    let (full_gpu_tok, full_gpu_prob) = full_gpu_r.predictions.first()
        .map(|(t, p)| (t.clone(), *p)).unwrap_or_default();

    println!("  Dense (baseline):    {dense_tok:>10} ({:.2}%)  {dense_ms:>6.0}ms  ({:.1} tok/s)", dense_prob * 100.0, 1000.0/dense_ms);
    println!("  Cache+Walk (CPU):    {cw_cpu_tok:>10}           {cw_cpu_ms:>6.0}ms  ({:.1} tok/s)", 1000.0/cw_cpu_ms);
    println!("  Cache+Walk (GPU):    {cw_gpu_tok:>10}           {cw_gpu_ms:>6.0}ms  ({:.1} tok/s)", 1000.0/cw_gpu_ms);
    println!("  Full pipe (CPU):     {full_cpu_tok:>10}           {full_cpu_ms:>6.0}ms  ({:.1} tok/s)", 1000.0/full_cpu_ms);
    println!("  Full pipe (GPU):     {full_gpu_tok:>10} ({:.2}%)  {full_gpu_ms:>6.0}ms  ({:.1} tok/s)", full_gpu_prob * 100.0, 1000.0/full_gpu_ms);
    println!();
    println!("  GPU vs CPU attn+FFN: {:.2}x ({:.0}ms)", cw_cpu_ms / cw_gpu_ms, cw_cpu_ms - cw_gpu_ms);
    println!("  GPU vs CPU full:     {:.2}x ({:.0}ms)", full_cpu_ms / full_gpu_ms, full_cpu_ms - full_gpu_ms);
    println!("  Full(GPU) vs Dense:  {:.2}x ({:.0}ms saved)", dense_ms / full_gpu_ms, dense_ms - full_gpu_ms);

    println!("=== Done ===");
    Ok(())
}
