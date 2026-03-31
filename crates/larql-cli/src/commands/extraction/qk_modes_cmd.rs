use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use larql_inference::ndarray::{self, Array1, Array2};
use larql_vindex::load_feature_labels;
use larql_inference::InferenceModel;

#[derive(Args)]
pub struct QkModesArgs {
    /// Model path or HuggingFace model ID.
    model: String,

    /// Only show heads with rank <= this value (the specialized ones).
    #[arg(long, default_value = "50")]
    max_rank: usize,

    /// Singular value threshold (fraction of largest).
    #[arg(long, default_value = "0.1")]
    threshold: f32,

    /// Top gate features to show per mode.
    #[arg(short = 'k', long, default_value = "5")]
    top_k: usize,

    /// Path to feature labels (down_meta.jsonl). Skips vocab projection.
    #[arg(long)]
    labels: Option<PathBuf>,

    /// Layers to analyze. Default: all.
    #[arg(short, long)]
    layers: Option<String>,
}

pub fn run(args: QkModesArgs) -> Result<(), Box<dyn std::error::Error>> {
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
        "  {} layers, {} Q heads, head_dim={}, hidden={} ({:.1}s)",
        num_layers, num_q_heads, head_dim, hidden_size,
        start.elapsed().as_secs_f64()
    );

    let layers: Vec<usize> = match &args.layers {
        Some(spec) => parse_layer_spec(spec)?,
        None => (0..num_layers).collect(),
    };

    // Load feature labels if provided
    let feature_labels: Option<HashMap<(usize, usize), String>> =
        if let Some(ref path) = args.labels {
            eprintln!("  Loading labels from {}...", path.display());
            let labels = load_feature_labels(path)?;
            eprintln!("  {} labels loaded", labels.len());
            Some(labels)
        } else {
            None
        };

    eprintln!("\n── Extracting QK modes for specialized heads (rank <= {}) ──\n", args.max_rank);

    let mut total_specialized = 0;
    let mut total_modes = 0;

    for &layer in &layers {
        let w_q = match weights.tensors.get(&arch.attn_q_key(layer)) {
            Some(w) => w,
            None => continue,
        };
        let w_k = match weights.tensors.get(&arch.attn_k_key(layer)) {
            Some(w) => w,
            None => continue,
        };
        let w_gate = match weights.tensors.get(&arch.ffn_gate_key(layer)) {
            Some(w) => w,
            None => continue,
        };

        let _intermediate = w_gate.shape()[0];

        for q_head in 0..num_q_heads {
            let kv_head = q_head / reps;

            // Extract Q and K blocks
            let q_start = q_head * head_dim;
            let q_block = w_q.slice(ndarray::s![q_start..q_start + head_dim, ..]);
            let k_start = kv_head * head_dim;
            let k_block = w_k.slice(ndarray::s![k_start..k_start + head_dim, ..]);

            // QK = Q_block × K_block^T = (head_dim, head_dim)
            let qk = q_block.dot(&k_block.t());

            // SVD via power iteration on QK^T × QK
            let qk_sq = qk.t().dot(&qk);
            let (singular_values, singular_vectors) =
                compute_svd(&qk_sq, head_dim, args.threshold);

            let rank = singular_values.len();
            if rank > args.max_rank {
                continue;
            }

            total_specialized += 1;
            total_modes += rank;

            println!("L{}H{} — rank {} (S_max={:.1})", layer, q_head, rank,
                if !singular_values.is_empty() { singular_values[0] } else { 0.0 });

            // For each mode (significant singular vector):
            // 1. The singular vector v is in head_dim space (from QK^T × QK)
            // 2. Map it back to hidden_size space: mode_hidden = K_block^T × v
            //    This gives us "what input pattern this mode detects"
            // 3. Project against gate vectors to see which FFN features it activates

            for (mode_idx, (sv, svec)) in singular_values.iter().zip(singular_vectors.iter()).enumerate() {
                // Map from head_dim space to hidden_size space via K^T
                // mode_hidden = K_block^T × svec = (hidden, head_dim) × (head_dim,) = (hidden,)
                let mode_hidden: Array1<f32> = k_block.t().dot(svec);

                // Project against gate vectors: scores = W_gate × mode_hidden
                let gate_scores = w_gate.dot(&mode_hidden);

                // Top features by absolute score
                let mut indexed: Vec<(usize, f32)> = gate_scores
                    .iter()
                    .copied()
                    .enumerate()
                    .collect();
                let k = args.top_k.min(indexed.len());
                indexed.select_nth_unstable_by(k, |a, b| {
                    b.1.abs().partial_cmp(&a.1.abs()).unwrap()
                });
                indexed.truncate(k);
                indexed.sort_unstable_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

                // Get labels
                let features_str: String = indexed
                    .iter()
                    .map(|&(f, score)| {
                        let label = feature_labels
                            .as_ref()
                            .and_then(|labels| labels.get(&(layer, f)))
                            .map(|s| s.as_str())
                            .unwrap_or("?");
                        format!("F{}→{} ({:+.2})", f, label, score)
                    })
                    .collect::<Vec<_>>()
                    .join(", ");

                let bar = if *sv > singular_values[0] * 0.5 {
                    "█"
                } else if *sv > singular_values[0] * 0.2 {
                    "▓"
                } else {
                    "▒"
                };

                println!(
                    "  mode {:2}: S={:5.1} {} [{}]",
                    mode_idx + 1,
                    sv,
                    bar,
                    features_str
                );
            }
            println!();
        }
    }

    println!("═══ Summary ═══");
    println!("  Specialized heads (rank <= {}): {}", args.max_rank, total_specialized);
    println!("  Total modes: {}", total_modes);
    println!("  Average modes per head: {:.1}",
        if total_specialized > 0 { total_modes as f64 / total_specialized as f64 } else { 0.0 });

    Ok(())
}

/// Compute SVD of symmetric PSD matrix via power iteration with deflation.
/// Returns (singular_values, singular_vectors) for significant components.
fn compute_svd(
    ata: &Array2<f32>,
    dim: usize,
    threshold: f32,
) -> (Vec<f32>, Vec<Array1<f32>>) {
    let mut matrix = ata.clone();
    let mut singular_values: Vec<f32> = Vec::new();
    let mut singular_vectors: Vec<Array1<f32>> = Vec::new();
    let iterations = 80;
    let max_components = dim.min(100);

    let mut s_max = 0.0f32;

    for _ in 0..max_components {
        let mut v = Array1::<f32>::ones(dim);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v /= norm;

        let mut eigenvalue = 0.0f32;

        for _ in 0..iterations {
            let mv = matrix.dot(&v);
            eigenvalue = mv.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            let norm: f32 = mv.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm < 1e-10 {
                break;
            }
            v = mv / norm;
        }

        let sv = eigenvalue.max(0.0).sqrt();

        if s_max == 0.0 {
            s_max = sv;
        }

        if sv < s_max * threshold || sv < 1e-6 {
            break;
        }

        singular_values.push(sv);
        singular_vectors.push(v.clone());

        // Deflate
        let mut outer = Array2::<f32>::zeros((dim, dim));
        for i in 0..dim {
            for j in 0..dim {
                outer[[i, j]] = eigenvalue * v[i] * v[j];
            }
        }
        matrix = matrix - outer;
    }

    (singular_values, singular_vectors)
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
