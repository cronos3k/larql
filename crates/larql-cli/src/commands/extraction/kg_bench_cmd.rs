use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use larql_vindex::load_feature_labels;
use larql_inference::{GateIndex, InferenceModel};

#[derive(Args)]
pub struct KgBenchArgs {
    /// Model path or HuggingFace model ID (for tokenizer).
    #[arg(short, long)]
    model: String,

    /// Path to gate index file.
    #[arg(long)]
    gate_index: PathBuf,

    /// Path to down_meta labels (from vindex).
    #[arg(long)]
    labels: PathBuf,

    /// Comma-separated prompts to test.
    #[arg(long)]
    prompts: String,

    /// Top-K features per layer from gate index.
    #[arg(short = 'k', long, default_value = "20")]
    top_k: usize,

    /// Layers to query (default: 26-33, the output band).
    #[arg(long, default_value = "26-33")]
    layers: String,

    /// Number of iterations for throughput test.
    #[arg(long, default_value = "100000")]
    throughput_iters: usize,
}

fn parse_range(spec: &str) -> Vec<usize> {
    let mut out = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if let Some((a, b)) = part.split_once('-') {
            let start: usize = a.parse().unwrap_or(0);
            let end: usize = b.parse().unwrap_or(0);
            out.extend(start..=end);
        } else if let Ok(l) = part.parse::<usize>() {
            out.push(l);
        }
    }
    out
}

pub fn run(args: KgBenchArgs) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading model (tokenizer only)...");
    let model = InferenceModel::load(&args.model)?;

    eprintln!("Loading gate index...");
    let gi = GateIndex::load(&args.gate_index, 10)?;

    eprintln!("Loading labels...");
    let labels = load_feature_labels(&args.labels)?;
    eprintln!("  {} labels loaded", labels.len());

    let layers = parse_range(&args.layers);
    let prompts: Vec<&str> = args.prompts.split(',').map(|s| s.trim()).collect();

    println!();
    println!("KG Retrieval — gate index lookup, zero matmuls");
    println!("{}", "=".repeat(80));

    for prompt in &prompts {
        let encoding = model.tokenizer().encode(*prompt, true)
            .map_err(|e| format!("tokenize error: {e}"))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();
        let entity_tokens: Vec<(usize, f32)> = token_ids.iter().map(|&t| (t as usize, 1.0)).collect();

        println!("\n{:?}", prompt);

        // Aggregate answer tokens across layers
        let mut token_votes: std::collections::HashMap<String, f32> = std::collections::HashMap::new();

        for &layer in &layers {
            let features = gi.lookup_from_tokens(&entity_tokens, layer, args.top_k);

            let mut display: Vec<String> = Vec::new();
            for &feat_id in features.iter().take(5) {
                let label = labels.get(&(layer, feat_id))
                    .map(|s| s.as_str())
                    .unwrap_or("?");
                display.push(format!("F{}→{}", feat_id, label));
                if label != "?" {
                    *token_votes.entry(label.to_string()).or_insert(0.0) += 1.0;
                }
            }
            // Count all features for votes, not just display
            for &feat_id in features.iter().skip(5) {
                if let Some(label) = labels.get(&(layer, feat_id)) {
                    *token_votes.entry(label.clone()).or_insert(0.0) += 1.0;
                }
            }

            println!("  L{:2}: {:3} feats  [{}]", layer, features.len(), display.join(", "));
        }

        if !token_votes.is_empty() {
            let mut sorted: Vec<_> = token_votes.into_iter().collect();
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            print!("  → ");
            for (tok, score) in sorted.iter().take(8) {
                print!("{:?}({:.0}) ", tok, score);
            }
            println!();
        }
    }

    // Throughput benchmark
    println!("\n{}", "=".repeat(80));

    let encoding = model.tokenizer().encode(prompts[0], true)
        .map_err(|e| format!("tokenize error: {e}"))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    let entity_tokens: Vec<(usize, f32)> = token_ids.iter().map(|&t| (t as usize, 1.0)).collect();

    // Method 1: Dynamic lookup (HashMap per call)
    for &layer in &layers { let _ = gi.lookup_from_tokens(&entity_tokens, layer, args.top_k); }
    let start = Instant::now();
    for _ in 0..args.throughput_iters {
        for &layer in &layers {
            let _ = gi.lookup_from_tokens(&entity_tokens, layer, args.top_k);
        }
    }
    let dyn_elapsed = start.elapsed();
    let dyn_us = dyn_elapsed.as_micros() as f64 / args.throughput_iters as f64;
    let dyn_qps = args.throughput_iters as f64 / dyn_elapsed.as_secs_f64();

    // Method 2: Precomputed entity (flat vec, zero alloc at query time)
    let precomputed = gi.precompute_entity(&token_ids, args.top_k);
    let mut checksum = 0usize;
    let start = Instant::now();
    for _ in 0..args.throughput_iters {
        for &layer in &layers {
            checksum += precomputed[layer].len();
        }
    }
    let pre_elapsed = start.elapsed();
    let pre_us = pre_elapsed.as_micros() as f64 / args.throughput_iters as f64;
    let pre_qps = args.throughput_iters as f64 / pre_elapsed.as_secs_f64();

    // Method 3: Precomputed + label resolve
    // Precompute feature→label for this entity
    let mut entity_labels: Vec<Vec<&str>> = vec![Vec::new(); precomputed.len()];
    for &layer in &layers {
        for &feat_id in &precomputed[layer] {
            let label = labels.get(&(layer, feat_id)).map(|s| s.as_str()).unwrap_or("?");
            entity_labels[layer].push(label);
        }
    }
    let start = Instant::now();
    let mut label_checksum = 0usize;
    for _ in 0..args.throughput_iters {
        for &layer in &layers {
            label_checksum += entity_labels[layer].len();
        }
    }
    let label_elapsed = start.elapsed();
    let label_us = label_elapsed.as_micros() as f64 / args.throughput_iters as f64;
    let label_qps = args.throughput_iters as f64 / label_elapsed.as_secs_f64();

    println!("Throughput: {} iters, {} layers, K={}", args.throughput_iters, layers.len(), args.top_k);
    println!("{:>25} {:>10} {:>12}", "Method", "us/query", "queries/sec");
    println!("{}", "-".repeat(50));
    println!("{:>25} {:>10.2} {:>12.0}", "dynamic (HashMap)", dyn_us, dyn_qps);
    println!("{:>25} {:>10.2} {:>12.0}", "precomputed (vec read)", pre_us, pre_qps);
    println!("{:>25} {:>10.2} {:>12.0}", "precomputed + labels", label_us, label_qps);
    println!("  (checksums: {} {} — prevents elimination)", checksum, label_checksum);

    Ok(())
}
