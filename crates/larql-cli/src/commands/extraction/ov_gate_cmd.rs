use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use larql_inference::ndarray::{self, Array2};
use larql_inference::tokenizers;
use larql_vindex::load_feature_labels;
use larql_inference::InferenceModel;

#[derive(Args)]
pub struct OvGateArgs {
    /// Model path or HuggingFace model ID.
    model: String,

    /// Layers to analyze. Default: all.
    #[arg(short, long)]
    layers: Option<String>,

    /// Top-K gate features to show per head.
    #[arg(short = 'k', long, default_value = "10")]
    top_k: usize,

    /// Only show heads at these layers (for focused analysis).
    #[arg(long)]
    heads: Option<String>,

    /// Show verbose per-feature details.
    #[arg(short, long)]
    verbose: bool,

    /// Output format: "table" (default) or "ndjson".
    #[arg(long, default_value = "table")]
    output: String,

    /// Output file path (for ndjson output). Defaults to stdout.
    #[arg(short = 'o', long)]
    output_file: Option<PathBuf>,

    /// Path to feature labels file (down_meta.jsonl or ffn_gate.vectors.jsonl).
    /// Skips slow vocab projection — uses precomputed labels instead.
    #[arg(long)]
    labels: Option<PathBuf>,
}

pub fn run(args: OvGateArgs) -> Result<(), Box<dyn std::error::Error>> {
    let ndjson = args.output == "ndjson";

    eprintln!("Loading model: {}", args.model);
    let start = Instant::now();
    let model = InferenceModel::load(&args.model)?;
    let weights = model.weights();
    let num_layers = weights.num_layers;
    let num_q_heads = weights.num_q_heads;
    let num_kv_heads = weights.num_kv_heads;
    let head_dim = weights.head_dim;
    let hidden_size = weights.hidden_size;
    let reps = num_q_heads / num_kv_heads;
    let arch = &*weights.arch;

    eprintln!(
        "  {} layers, {} Q heads, {} KV heads, head_dim={}, hidden={} ({:.1}s)",
        num_layers, num_q_heads, num_kv_heads, head_dim, hidden_size,
        start.elapsed().as_secs_f64()
    );

    let layers: Vec<usize> = match &args.layers {
        Some(spec) => parse_layer_spec(spec)?,
        None => (0..num_layers).collect(),
    };

    // Set up output writer for NDJSON mode
    let mut ndjson_writer: Option<BufWriter<Box<dyn Write>>> = if ndjson {
        let writer: Box<dyn Write> = match &args.output_file {
            Some(path) => Box::new(std::fs::File::create(path)?),
            None => Box::new(std::io::stdout()),
        };
        let mut w = BufWriter::new(writer);
        // Write header
        let header = serde_json::json!({
            "_header": true,
            "type": "ov_gate_coupling",
            "model": args.model,
            "top_k": args.top_k,
            "num_layers": num_layers,
            "num_q_heads": num_q_heads,
            "layers": &layers,
        });
        serde_json::to_writer(&mut w, &header)?;
        w.write_all(b"\n")?;
        Some(w)
    } else {
        None
    };

    // ── For each layer, for each head: compute OV circuit → gate coupling ──

    if !ndjson {
        println!(
            "\n{:<6} {:<5} {:>8}  {:<60}  {:<60}",
            "Layer", "Head", "Coupling", "Top gate features (what head activates)", "Top gate features (what head hears)"
        );
        println!("{}", "-".repeat(150));
    }

    // Collected data: (layer, head, feature, coupling, total_coupling)
    struct HeadData {
        layer: usize,
        head: usize,
        couplings: Vec<(usize, f32)>,
        total_coupling: f32,
    }

    let compute_start = Instant::now();
    eprintln!();
    let mut all_heads: Vec<HeadData> = Vec::new();

    for (li, &layer) in layers.iter().enumerate() {
        eprint!("L{layer}... ");
        let _ = std::io::stderr().flush();
        if (li + 1) % 10 == 0 {
            eprintln!("({}/{} layers, {:.0}s)", li + 1, layers.len(), compute_start.elapsed().as_secs_f64());
            eprint!("  ");
            let _ = std::io::stderr().flush();
        }

        let w_o = match weights.tensors.get(&arch.attn_o_key(layer)) {
            Some(w) => w,
            None => continue,
        };
        let w_gate = match weights.tensors.get(&arch.ffn_gate_key(layer)) {
            Some(w) => w,
            None => continue,
        };
        let intermediate = w_gate.shape()[0];

        for q_head in 0..num_q_heads {
            let o_start = q_head * head_dim;
            let o_block = w_o.slice(ndarray::s![.., o_start..o_start + head_dim]);
            let o_block_owned = o_block.to_owned();
            let gate_o = w_gate.dot(&o_block_owned);

            let mut couplings: Vec<(usize, f32)> = Vec::with_capacity(intermediate);
            for f in 0..intermediate {
                let row = gate_o.row(f);
                let norm: f32 = row.iter().map(|&v| v * v).sum::<f32>().sqrt();
                couplings.push((f, norm));
            }

            let total_coupling: f32 = couplings.iter().map(|(_, n)| n).sum::<f32>();

            let k = args.top_k.min(couplings.len());
            couplings.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
            couplings.truncate(k);
            couplings.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            all_heads.push(HeadData {
                layer,
                head: q_head,
                couplings,
                total_coupling,
            });
        }
    }
    eprintln!("\n  {} heads computed ({:.1}s)", all_heads.len(), compute_start.elapsed().as_secs_f64());

    // Label unique features
    let label_start = Instant::now();
    let feature_labels: std::collections::HashMap<(usize, usize), String> = if let Some(ref labels_path) = args.labels {
        eprintln!("  Loading labels from {}...", labels_path.display());
        let labels = load_feature_labels(labels_path)?;
        eprintln!("  {} labels loaded ({:.1}s)", labels.len(), label_start.elapsed().as_secs_f64());
        labels
    } else {
        eprintln!("  Labeling features (slow — use --labels for instant labels)...");
        let mut labels: std::collections::HashMap<(usize, usize), String> = std::collections::HashMap::new();
        for hd in &all_heads {
            for &(f, _) in &hd.couplings {
                labels.entry((hd.layer, f)).or_default();
            }
        }
        let total_features = labels.len();
        for (i, (&(layer, feat), label)) in labels.iter_mut().enumerate() {
            let gate_key = arch.ffn_gate_key(layer);
            if let Some(w_gate) = weights.tensors.get(&gate_key) {
                let gate_row = w_gate.row(feat);
                *label = project_top_token(&weights.embed, &gate_row.to_vec(), model.tokenizer());
            }
            if (i + 1) % 500 == 0 {
                eprint!("\r  {}/{} features...", i + 1, total_features);
                let _ = std::io::stderr().flush();
            }
        }
        eprintln!("\r  {} features labeled ({:.1}s)", total_features, label_start.elapsed().as_secs_f64());
        labels
    };

    // Output
    let mut total_edges = 0usize;

    if let Some(ref mut writer) = ndjson_writer {
        for hd in &all_heads {
            for &(f, c) in &hd.couplings {
                let top_tok = feature_labels.get(&(hd.layer, f)).map(|s| s.as_str()).unwrap_or("?");
                let record = serde_json::json!({
                    "head": format!("L{}_H{}", hd.layer, hd.head),
                    "layer": hd.layer,
                    "head_idx": hd.head,
                    "feature": format!("L{}_F{}", hd.layer, f),
                    "feature_idx": f,
                    "feature_layer": hd.layer,
                    "target_token": top_tok,
                    "coupling": (c * 1000.0).round() / 1000.0,
                    "total_coupling": (hd.total_coupling * 10.0).round() / 10.0,
                });
                serde_json::to_writer(&mut *writer, &record)?;
                writer.write_all(b"\n")?;
                total_edges += 1;
            }
        }
        writer.flush()?;
        eprintln!(
            "\nWrote {} coupling edges ({} layers × {} heads × top-{})",
            total_edges, layers.len(), num_q_heads, args.top_k,
        );
    } else {
        println!(
            "\n{:<6} {:<5} {:>8}  {:<60}  {:<60}",
            "Layer", "Head", "Coupling", "Top gate features (what head activates)", "Top gate features (what head hears)"
        );
        println!("{}", "-".repeat(150));

        for hd in &all_heads {
            let top_activates: String = hd.couplings
                .iter()
                .take(5)
                .map(|(f, c)| {
                    let tok = feature_labels.get(&(hd.layer, *f)).map(|s| s.as_str()).unwrap_or("?");
                    format!("F{}→{} ({:.2})", f, tok, c)
                })
                .collect::<Vec<_>>()
                .join(", ");

            // V-block hears (still needs vocab projection — but only 5 per head for display)
            let w_v = weights.tensors.get(&arch.attn_v_key(hd.layer));
            let top_hears: String = if let Some(w_v) = w_v {
                let kv_head = hd.head / reps;
                let v_start = kv_head * head_dim;
                let v_block = w_v.slice(ndarray::s![v_start..v_start + head_dim, ..]);
                let mut v_sum = vec![0.0f32; hidden_size];
                for d in 0..head_dim {
                    let row = v_block.row(d);
                    for (j, &v) in row.iter().enumerate() {
                        v_sum[j] += v.abs();
                    }
                }
                let top_toks = project_top_n(&weights.embed, &v_sum, 5, model.tokenizer());
                top_toks.join(", ")
            } else {
                String::new()
            };

            println!(
                "L{:<4} H{:<4} {:>7.1}  {:<60}  {:<60}",
                hd.layer, hd.head, hd.total_coupling, top_activates, top_hears,
            );

            if args.verbose {
                for (f, c) in &hd.couplings {
                    let tok = feature_labels.get(&(hd.layer, *f)).map(|s| s.as_str()).unwrap_or("?");
                    println!("        F{:<6} coupling={:.3}  gate_hears={}", f, c, tok);
                }
            }
        }
    }

    Ok(())
}

fn project_top_token(
    embed: &Array2<f32>,
    vector: &[f32],
    tokenizer: &tokenizers::Tokenizer,
) -> String {
    let vocab_size = embed.shape()[0];
    let mut best_idx = 0;
    let mut best_dot = f32::NEG_INFINITY;

    for i in 0..vocab_size {
        let row = embed.row(i);
        let dot: f32 = row.iter().zip(vector.iter()).map(|(a, b)| a * b).sum();
        if dot > best_dot {
            best_dot = dot;
            best_idx = i;
        }
    }

    tokenizer
        .decode(&[best_idx as u32], true)
        .unwrap_or_else(|_| format!("T{best_idx}"))
        .trim()
        .to_string()
}

fn project_top_n(
    embed: &Array2<f32>,
    vector: &[f32],
    n: usize,
    tokenizer: &tokenizers::Tokenizer,
) -> Vec<String> {
    let vocab_size = embed.shape()[0];
    let mut scores: Vec<(usize, f32)> = Vec::with_capacity(vocab_size);

    for i in 0..vocab_size {
        let row = embed.row(i);
        let dot: f32 = row.iter().zip(vector.iter()).map(|(a, b)| a * b).sum();
        scores.push((i, dot));
    }

    let k = n.min(scores.len());
    scores.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
    scores.truncate(k);
    scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    scores
        .into_iter()
        .filter_map(|(idx, _)| {
            tokenizer
                .decode(&[idx as u32], true)
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        })
        .collect()
}

fn parse_layer_spec(spec: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let mut layers = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.contains('-') {
            let (a, b) = part
                .split_once('-')
                .ok_or_else(|| format!("invalid range: {part}"))?;
            let start: usize = a.parse()?;
            let end: usize = b.parse()?;
            layers.extend(start..=end);
        } else {
            layers.push(part.parse()?);
        }
    }
    Ok(layers)
}
