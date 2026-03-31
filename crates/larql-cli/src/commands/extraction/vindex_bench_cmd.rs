use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use larql_vindex::{load_vindex_tokenizer, IndexLoadCallbacks, VectorIndex};
use larql_inference::{
    predict, predict_with_ffn, DownClusteredFfn, DownClusteredIndex, InferenceModel,
    vindex::WalkFfn,
};

#[derive(Args)]
pub struct VindexBenchArgs {
    /// Path to .vindex directory.
    #[arg(long)]
    index: PathBuf,

    /// Model path (required for attention + vindex FFN).
    #[arg(short, long)]
    model: String,

    /// Comma-separated prompts.
    #[arg(long)]
    prompts: String,

    /// Comma-separated K values to sweep.
    #[arg(short = 'k', long, default_value = "10,50,100,500,1000,2000,4000,8092")]
    top_k_values: String,
}

struct QuietCallbacks;
impl IndexLoadCallbacks for QuietCallbacks {
    fn on_file_start(&mut self, _c: &str, _p: &str) {}
    fn on_progress(&mut self, _r: usize) {}
    fn on_file_done(&mut self, _c: &str, _r: usize, _ms: f64) {}
}

pub fn run(args: VindexBenchArgs) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading vindex: {}", args.index.display());
    let mut cb = QuietCallbacks;
    let index = VectorIndex::load_vindex(&args.index, &mut cb)?;
    let _tokenizer = load_vindex_tokenizer(&args.index)?;

    eprintln!("Loading model: {}", args.model);
    let model = InferenceModel::load(&args.model)?;
    let weights = model.weights();
    eprintln!("  {} layers, hidden_size={}", weights.num_layers, weights.hidden_size);

    let prompts: Vec<&str> = args.prompts.split(',').map(|s| s.trim()).collect();
    let k_values: Vec<usize> = args.top_k_values.split(',')
        .map(|s| s.trim().parse().unwrap()).collect();

    // Get dense ground truth for all prompts first
    eprintln!("Running dense ground truth...");
    let mut dense_results: Vec<(String, f64, f64)> = Vec::new(); // (token, prob, ms)
    for prompt in &prompts {
        let encoding = model.tokenizer().encode(*prompt, true)
            .map_err(|e| format!("tokenize error: {e}"))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();
        let start = Instant::now();
        let result = predict(weights, model.tokenizer(), &token_ids, 5);
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        let (tok, prob) = result.predictions.first()
            .map(|(t, p)| (t.clone(), *p)).unwrap_or(("?".into(), 0.0));
        dense_results.push((tok, prob, ms));
    }

    // Header
    println!();
    println!("Attention + Vindex WalkFfn — accuracy vs K");
    println!("{}", "=".repeat(90));

    // For each K value, run all prompts
    for &k in &k_values {
        let walk_ffn = WalkFfn::new(weights, &index, k);

        let mut matches = 0;
        let mut total_walk_ms = 0.0;

        for (i, prompt) in prompts.iter().enumerate() {
            let encoding = model.tokenizer().encode(*prompt, true)
                .map_err(|e| format!("tokenize error: {e}"))?;
            let token_ids: Vec<u32> = encoding.get_ids().to_vec();

            let start = Instant::now();
            let result = predict_with_ffn(weights, model.tokenizer(), &token_ids, 5, &walk_ffn);
            let walk_ms = start.elapsed().as_secs_f64() * 1000.0;
            total_walk_ms += walk_ms;

            let walk_top1 = result.predictions.first()
                .map(|(t, _)| t.as_str()).unwrap_or("?");
            if walk_top1 == dense_results[i].0 {
                matches += 1;
            }
        }

        let avg_walk_ms = total_walk_ms / prompts.len() as f64;
        let avg_dense_ms: f64 = dense_results.iter().map(|r| r.2).sum::<f64>() / prompts.len() as f64;

        println!(
            "  K={:<6}  Match: {}/{} ({:>3.0}%)  Walk: {:>7.0}ms  Dense: {:>7.0}ms  Speedup: {:.2}x",
            k, matches, prompts.len(),
            matches as f64 / prompts.len() as f64 * 100.0,
            avg_walk_ms, avg_dense_ms,
            avg_dense_ms / avg_walk_ms,
        );
    }

    // Down-clustered: features selected by output direction
    let all_layers: Vec<usize> = (0..weights.num_layers).collect();
    for &nc in &[64, 128, 256] {
        for &tc in &[1, 2, 4, 8] {
            eprint!("\r  Building down-clusters: {} clusters, top_c={}...", nc, tc);
            let dc_index = DownClusteredIndex::build(
                weights, &all_layers, nc, tc, 10, |_, _| {},
            );

            let dc_ffn = DownClusteredFfn { weights, down_index: &dc_index };
            let mut matches = 0;
            let mut total_ms = 0.0;

            for (i, prompt) in prompts.iter().enumerate() {
                let encoding = model.tokenizer().encode(*prompt, true)
                    .map_err(|e| format!("tokenize error: {e}"))?;
                let token_ids: Vec<u32> = encoding.get_ids().to_vec();
                let start = Instant::now();
                let result = predict_with_ffn(weights, model.tokenizer(), &token_ids, 5, &dc_ffn);
                total_ms += start.elapsed().as_secs_f64() * 1000.0;
                let top1 = result.predictions.first().map(|(t, _)| t.as_str()).unwrap_or("?");
                if top1 == dense_results[i].0 { matches += 1; }
            }

            let avg_ms = total_ms / prompts.len() as f64;
            let avg_dense_ms: f64 = dense_results.iter().map(|r| r.2).sum::<f64>() / prompts.len() as f64;
            let avg_feats = dc_index.avg_cluster_size() * tc as f64;
            eprintln!("\r  dc:{}/c{}  Match: {}/{} ({:>3.0}%)  {:>7.0}ms  ~{:.0} feats  {:.2}x",
                nc, tc, matches, prompts.len(),
                matches as f64 / prompts.len() as f64 * 100.0,
                avg_ms, avg_feats, avg_dense_ms / avg_ms);
        }
    }

    // Show per-prompt detail at the best K
    let best_k = *k_values.last().unwrap();
    let walk_ffn = WalkFfn::new(weights, &index, best_k);

    println!();
    println!("Detail at K={}:", best_k);
    println!("{:40} {:>12} {:>7} {:>12} {:>7} {:>5}",
        "Prompt", "Walk Top-1", "Prob", "Dense Top-1", "Prob", "Match");
    println!("{}", "-".repeat(90));

    for (i, prompt) in prompts.iter().enumerate() {
        let encoding = model.tokenizer().encode(*prompt, true)
            .map_err(|e| format!("tokenize error: {e}"))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        let result = predict_with_ffn(weights, model.tokenizer(), &token_ids, 5, &walk_ffn);
        let (w_tok, w_prob) = result.predictions.first()
            .map(|(t, p)| (t.as_str(), *p)).unwrap_or(("?", 0.0));
        let (d_tok, d_prob) = (&dense_results[i].0, dense_results[i].1);
        let is_match = w_tok == d_tok.as_str();

        println!("{:40} {:>12} {:>6.1}% {:>12} {:>6.1}% {:>5}",
            prompt, w_tok, w_prob * 100.0, d_tok, d_prob * 100.0,
            if is_match { "yes" } else { "NO" });
    }

    Ok(())
}
