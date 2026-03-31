use std::collections::HashMap;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use larql_inference::ndarray;
use larql_inference::tokenizers;
use larql_vindex::load_feature_labels;
use larql_inference::InferenceModel;

#[derive(Args)]
pub struct CircuitDiscoverArgs {
    /// Model path or HuggingFace model ID.
    model: String,

    /// Output directory for circuit edges and templates.
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Top-K gate features per head to keep.
    #[arg(short = 'k', long, default_value = "50")]
    top_k: usize,

    /// Minimum coupling score to include an edge.
    #[arg(long, default_value = "0.5")]
    min_coupling: f32,

    /// Community detection threshold (Jaccard similarity on token sets).
    #[arg(long, default_value = "0.3")]
    cluster_threshold: f32,

    /// Layers to analyze. Default: all.
    #[arg(short, long)]
    layers: Option<String>,

    /// Path to feature labels file (down_meta.jsonl or ffn_gate.vectors.jsonl).
    /// Skips slow vocab projection — uses precomputed labels instead.
    #[arg(long)]
    labels: Option<PathBuf>,
}

/// An OV→gate edge: attention head activates FFN feature.
#[derive(serde::Serialize, Clone)]
struct OvGateEdge {
    layer: usize,
    head: usize,
    feature: usize,
    coupling: f32,
    gate_top_token: String,
}

/// A template circuit: a set of attention heads that route to the same FFN features.
struct Circuit {
    id: usize,
    heads: Vec<(usize, usize)>, // (layer, head)
    features: Vec<(usize, usize, f32)>, // (layer, feature, total_coupling)
    top_tokens: Vec<String>,
}

pub fn run(args: CircuitDiscoverArgs) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading model: {}", args.model);
    let start = Instant::now();
    let model = InferenceModel::load(&args.model)?;
    let weights = model.weights();
    let num_layers = weights.num_layers;
    let num_q_heads = weights.num_q_heads;
    let num_kv_heads = weights.num_kv_heads;
    let head_dim = weights.head_dim;
    let reps = num_q_heads / num_kv_heads;
    let arch = &*weights.arch;

    eprintln!(
        "  {} layers, {} heads ({:.1}s)",
        num_layers, num_q_heads,
        start.elapsed().as_secs_f64()
    );

    let layers: Vec<usize> = match &args.layers {
        Some(spec) => parse_layer_spec(spec)?,
        None => (0..num_layers).collect(),
    };

    // ── Step 1: Compute all OV→gate edges ──
    eprintln!("\n── Computing OV→gate couplings ──");
    let start = Instant::now();

    let mut all_edges: Vec<OvGateEdge> = Vec::new();
    // head_fingerprints[head_key] = feature coupling vector for clustering
    let mut head_fingerprints: HashMap<(usize, usize), Vec<(usize, usize, f32)>> = HashMap::new();

    for &layer in &layers {
        let w_v = match weights.tensors.get(&arch.attn_v_key(layer)) {
            Some(w) => w,
            None => continue,
        };
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
            let kv_head = q_head / reps;

            let v_start = kv_head * head_dim;
            let _v_block = w_v.slice(ndarray::s![v_start..v_start + head_dim, ..]);
            let o_start = q_head * head_dim;
            let o_block = w_o.slice(ndarray::s![.., o_start..o_start + head_dim]);

            // gate_o = W_gate × W_O_block = (intermediate, head_dim)
            let o_block_owned = o_block.to_owned();
            let gate_o = w_gate.dot(&o_block_owned);

            // Per-feature coupling
            let mut couplings: Vec<(usize, f32)> = Vec::with_capacity(intermediate);
            for f in 0..intermediate {
                let row = gate_o.row(f);
                let norm: f32 = row.iter().map(|&v| v * v).sum::<f32>().sqrt();
                couplings.push((f, norm));
            }

            // Top-K
            let k = args.top_k.min(couplings.len());
            couplings.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
            couplings.truncate(k);
            couplings.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let mut fingerprint: Vec<(usize, usize, f32)> = Vec::new();

            for &(feat, coupling) in &couplings {
                if coupling < args.min_coupling {
                    continue;
                }

                // Defer token labeling — just store the edge with empty label
                all_edges.push(OvGateEdge {
                    layer,
                    head: q_head,
                    feature: feat,
                    coupling,
                    gate_top_token: String::new(),
                });

                fingerprint.push((layer, feat, coupling));
            }

            head_fingerprints.insert((layer, q_head), fingerprint);
        }

        eprint!("L{layer}... ");
        let _ = io::stderr().flush();
        if (layer + 1) % 10 == 0 {
            eprintln!("({}/{} layers, {:.0}s)", layer + 1, num_layers, start.elapsed().as_secs_f64());
            eprint!("  ");
            let _ = io::stderr().flush();
        }
    }
    eprintln!(
        "\r  {} edges from {} heads ({:.1}s)",
        all_edges.len(),
        head_fingerprints.len(),
        start.elapsed().as_secs_f64()
    );

    // ── Step 1b: Label features with top tokens ──
    {
        let label_start = Instant::now();
        if let Some(ref labels_path) = args.labels {
            // Fast path: load precomputed labels from vindex/NDJSON
            eprintln!("  Loading labels from {}...", labels_path.display());
            let label_map = load_feature_labels(labels_path)?;
            for edge in &mut all_edges {
                if let Some(label) = label_map.get(&(edge.layer, edge.feature)) {
                    edge.gate_top_token = label.clone();
                }
            }
            eprintln!("  {} labels loaded ({:.1}s)", label_map.len(), label_start.elapsed().as_secs_f64());
        } else {
            // Slow path: project each feature against vocab
            eprintln!("  Labeling features (slow — use --labels for instant labels)...");
            let mut unique_features: HashMap<(usize, usize), String> = HashMap::new();
            for edge in &all_edges {
                unique_features.entry((edge.layer, edge.feature)).or_default();
            }
            let total = unique_features.len();
            for (i, (&(layer, feat), label)) in unique_features.iter_mut().enumerate() {
                let gate_key = arch.ffn_gate_key(layer);
                if let Some(w_gate) = weights.tensors.get(&gate_key) {
                    let gate_row = w_gate.row(feat);
                    *label = project_top_token(&weights.embed, &gate_row.to_vec(), model.tokenizer());
                }
                if (i + 1) % 500 == 0 {
                    eprint!("\r  {}/{} features...", i + 1, total);
                    let _ = io::stderr().flush();
                }
            }
            for edge in &mut all_edges {
                if let Some(label) = unique_features.get(&(edge.layer, edge.feature)) {
                    edge.gate_top_token = label.clone();
                }
            }
            eprintln!("\r  {} features labeled ({:.1}s)", total, label_start.elapsed().as_secs_f64());
        }
    }

    // ── Step 2: Export edges ──
    if let Some(ref output_dir) = args.output {
        std::fs::create_dir_all(output_dir)?;
        let edge_path = output_dir.join("ov_gate_edges.jsonl");
        let mut writer = BufWriter::new(std::fs::File::create(&edge_path)?);

        // Header
        let header = serde_json::json!({
            "_header": true,
            "type": "ov_gate_coupling",
            "model": args.model,
            "num_edges": all_edges.len(),
            "min_coupling": args.min_coupling,
            "top_k": args.top_k,
        });
        serde_json::to_writer(&mut writer, &header)?;
        writer.write_all(b"\n")?;

        for edge in &all_edges {
            serde_json::to_writer(&mut writer, edge)?;
            writer.write_all(b"\n")?;
        }
        writer.flush()?;
        eprintln!("  Edges: {}", edge_path.display());
    }

    // ── Step 3: Cluster heads by shared gate feature targets ──
    eprintln!("\n── Clustering heads into circuits ──");
    let cluster_start = Instant::now();

    // Build token-based fingerprint per head: the set of top_tokens from its coupled features.
    // This enables cross-layer comparison — heads that activate semantically similar features
    // (same tokens) cluster together regardless of layer.
    let mut head_token_sets: HashMap<(usize, usize), HashMap<String, f32>> = HashMap::new();
    for edge in &all_edges {
        if !edge.gate_top_token.is_empty() {
            *head_token_sets
                .entry((edge.layer, edge.head))
                .or_default()
                .entry(edge.gate_top_token.clone())
                .or_insert(0.0) += edge.coupling;
        }
    }

    let head_keys: Vec<(usize, usize)> = head_fingerprints.keys().copied().collect();

    // Compare all head pairs by token overlap (Jaccard on token sets)
    // 272 heads = 36K pairs — manageable
    let mut adjacency: HashMap<(usize, usize), Vec<((usize, usize), f32)>> = HashMap::new();

    let head_token_vecs: HashMap<(usize, usize), Vec<String>> = head_token_sets
        .iter()
        .map(|(k, tokens)| {
            let mut toks: Vec<String> = tokens.keys().cloned().collect();
            toks.sort();
            (*k, toks)
        })
        .collect();

    for i in 0..head_keys.len() {
        for j in (i + 1)..head_keys.len() {
            let h_i = head_keys[i];
            let h_j = head_keys[j];

            let toks_i = match head_token_vecs.get(&h_i) {
                Some(t) => t,
                None => continue,
            };
            let toks_j = match head_token_vecs.get(&h_j) {
                Some(t) => t,
                None => continue,
            };

            let set_i: std::collections::HashSet<&str> =
                toks_i.iter().map(|s| s.as_str()).collect();
            let shared = toks_j.iter().filter(|t| set_i.contains(t.as_str())).count();
            let union = toks_i.len() + toks_j.len() - shared;

            if union > 0 && shared > 0 {
                let jaccard = shared as f32 / union as f32;
                if jaccard >= args.cluster_threshold {
                    adjacency.entry(h_i).or_default().push((h_j, jaccard));
                    adjacency.entry(h_j).or_default().push((h_i, jaccard));
                }
            }
        }
    }

    eprintln!(
        "  {} similarity edges above threshold {:.2}",
        adjacency.values().map(|v| v.len()).sum::<usize>() / 2,
        args.cluster_threshold
    );

    // BFS clustering using adjacency list
    let mut cluster_id: HashMap<(usize, usize), usize> = HashMap::new();
    let mut next_cluster = 0;

    for &head in &head_keys {
        if cluster_id.contains_key(&head) {
            continue;
        }

        let cid = next_cluster;
        next_cluster += 1;
        cluster_id.insert(head, cid);

        let mut queue = vec![head];
        while let Some(current) = queue.pop() {
            if let Some(neighbors) = adjacency.get(&current) {
                for &(neighbor, _sim) in neighbors {
                    if !cluster_id.contains_key(&neighbor) {
                        cluster_id.insert(neighbor, cid);
                        queue.push(neighbor);
                    }
                }
            }
        }
    }

    eprintln!("  Clustered in {:.1}s", cluster_start.elapsed().as_secs_f64());

    // Build circuits from clusters
    let mut cluster_heads: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for (&head, &cid) in &cluster_id {
        cluster_heads.entry(cid).or_default().push(head);
    }

    // Sort clusters by size (largest first)
    let mut circuits: Vec<Circuit> = Vec::new();
    let mut sorted_clusters: Vec<(usize, Vec<(usize, usize)>)> =
        cluster_heads.into_iter().collect();
    sorted_clusters.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

    for (cid, mut heads) in sorted_clusters {
        heads.sort();

        // Aggregate features across all heads in this circuit
        let mut feature_coupling: HashMap<(usize, usize), f32> = HashMap::new();
        let heads_set: std::collections::HashSet<(usize, usize)> = heads.iter().copied().collect();
        for edge in &all_edges {
            if heads_set.contains(&(edge.layer, edge.head)) {
                *feature_coupling
                    .entry((edge.layer, edge.feature))
                    .or_insert(0.0) += edge.coupling;
            }
        }

        let mut features: Vec<(usize, usize, f32)> = feature_coupling
            .into_iter()
            .map(|((l, f), c)| (l, f, c))
            .collect();
        features.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

        // Get top tokens from edge labels (already resolved)
        let top_tokens: Vec<String> = features
            .iter()
            .take(10)
            .filter_map(|&(layer, feat, _)| {
                all_edges.iter()
                    .find(|e| e.layer == layer && e.feature == feat && !e.gate_top_token.is_empty())
                    .map(|e| e.gate_top_token.clone())
            })
            .collect();

        circuits.push(Circuit {
            id: cid,
            heads,
            features: features.into_iter().take(20).collect(),
            top_tokens,
        });
    }

    // ── Step 4: Print circuits ──
    println!("\n═══ Discovered Circuits ({} total) ═══\n", circuits.len());

    let large_circuits: Vec<&Circuit> = circuits.iter().filter(|c| c.heads.len() >= 3).collect();
    let small_circuits: Vec<&Circuit> = circuits.iter().filter(|c| c.heads.len() < 3).collect();

    for circuit in &large_circuits {
        let heads_str: String = circuit
            .heads
            .iter()
            .map(|(l, h)| format!("L{}H{}", l, h))
            .collect::<Vec<_>>()
            .join(", ");

        let tokens_str = circuit.top_tokens.join(", ");

        println!(
            "Circuit {} ({} heads): [{}]",
            circuit.id,
            circuit.heads.len(),
            heads_str
        );
        println!("  Top tokens: {}", tokens_str);

        // Show layer distribution
        let mut layer_counts: HashMap<usize, usize> = HashMap::new();
        for &(l, _) in &circuit.heads {
            *layer_counts.entry(l).or_insert(0) += 1;
        }
        let mut layer_dist: Vec<(usize, usize)> = layer_counts.into_iter().collect();
        layer_dist.sort();
        let dist_str: String = layer_dist
            .iter()
            .map(|(l, c)| format!("L{}×{}", l, c))
            .collect::<Vec<_>>()
            .join(" ");
        println!("  Layers: {}", dist_str);
        println!();
    }

    println!(
        "  {} large circuits (3+ heads), {} small circuits (1-2 heads)\n",
        large_circuits.len(),
        small_circuits.len()
    );

    // Summary
    println!("═══ Summary ═══");
    println!("  Total edges: {}", all_edges.len());
    println!("  Total heads: {}", head_keys.len());
    println!("  Total circuits: {}", circuits.len());
    println!(
        "  Large circuits (3+ heads): {}",
        large_circuits.len()
    );

    if let Some(biggest) = large_circuits.first() {
        println!(
            "  Largest circuit: {} heads, tokens: {}",
            biggest.heads.len(),
            biggest.top_tokens.iter().take(5).cloned().collect::<Vec<_>>().join(", ")
        );
    }

    // ── Step 5: Export circuits ──
    if let Some(ref output_dir) = args.output {
        let circuit_path = output_dir.join("circuits.json");
        let circuit_data: Vec<serde_json::Value> = circuits
            .iter()
            .map(|c| {
                serde_json::json!({
                    "id": c.id,
                    "num_heads": c.heads.len(),
                    "heads": c.heads.iter().map(|(l, h)| format!("L{}H{}", l, h)).collect::<Vec<_>>(),
                    "top_tokens": c.top_tokens,
                    "top_features": c.features.iter().take(10).map(|(l, f, c)| {
                        serde_json::json!({"layer": l, "feature": f, "coupling": c})
                    }).collect::<Vec<_>>(),
                })
            })
            .collect();

        let json = serde_json::to_string_pretty(&circuit_data)?;
        std::fs::write(&circuit_path, json)?;
        eprintln!("\n  Circuits: {}", circuit_path.display());
    }

    Ok(())
}

fn project_top_token(
    embed: &ndarray::Array2<f32>,
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
