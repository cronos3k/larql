use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use indicatif::{ProgressBar, ProgressStyle};
use larql_core::*;
use larql_inference::walker::attention_walker::{AttentionLayerResult, AttentionWalker};
use larql_inference::walker::weight_walker::{LayerResult, WalkCallbacks, WalkConfig};

#[derive(Args)]
pub struct AttentionWalkArgs {
    /// Model path or HuggingFace model ID (e.g. google/gemma-3-4b-it).
    model: String,

    /// Output file (.larql.json or .larql.bin).
    #[arg(short, long)]
    output: PathBuf,

    /// Single layer to walk. Default: all layers.
    #[arg(short, long)]
    layer: Option<usize>,

    /// Top-k tokens per head.
    #[arg(long, default_value = "3")]
    top_k: usize,

    /// Minimum score.
    #[arg(long, default_value = "0.0")]
    min_score: f32,

    /// Write layer statistics to a separate file.
    #[arg(long)]
    stats: Option<PathBuf>,
}

struct ProgressCallbacks {
    bar: ProgressBar,
    head_bar: ProgressBar,
    output: PathBuf,
}

impl WalkCallbacks for ProgressCallbacks {
    fn on_layer_start(&mut self, layer: usize, num_heads: usize) {
        self.head_bar.set_length(num_heads as u64);
        self.head_bar.set_position(0);
        self.head_bar
            .set_message(format!("L{layer}: {num_heads} heads"));
    }

    fn on_progress(&mut self, layer: usize, heads_done: usize, total: usize) {
        self.head_bar.set_position(heads_done as u64);
        self.head_bar
            .set_message(format!("L{layer}: {heads_done}/{total}"));
    }

    fn on_layer_done(&mut self, result: &LayerResult) {
        self.head_bar.set_position(result.features_scanned as u64);
        self.bar.inc(1);
        let s = &result.stats;
        eprintln!(
            "  L{:2}: {:6} edges  ({:.0}s)  conf: avg={:.4} max={:.4}  c_in={:.2} c_out={:.2}",
            result.layer,
            result.edges_found,
            result.elapsed_ms / 1000.0,
            s.mean_confidence,
            s.max_confidence,
            s.mean_c_in,
            s.mean_c_out,
        );
    }

    fn on_checkpoint(&mut self, graph: &Graph) {
        let _ = larql_core::save(graph, &self.output);
    }
}

pub fn run(args: AttentionWalkArgs) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading model: {}", args.model);
    let walker = AttentionWalker::load(&args.model)?;
    eprintln!("  {} layers", walker.num_layers());

    // Resume
    let mut graph: Graph;
    let mut completed: HashSet<usize> = HashSet::new();

    if args.output.exists() {
        graph = larql_core::load(&args.output)?;
        for edge in graph.edges() {
            if let Some(meta) = &edge.metadata {
                if meta.get("circuit").and_then(|v| v.as_str()) == Some("OV") {
                    if let Some(layer) = meta.get("layer").and_then(|v| v.as_u64()) {
                        completed.insert(layer as usize);
                    }
                }
            }
        }
        eprintln!(
            "  Resumed: {} edges from {} layers\n",
            graph.edge_count(),
            completed.len()
        );
    } else {
        graph = Graph::new();
        graph.metadata.insert(
            "model".to_string(),
            serde_json::Value::String(args.model.clone()),
        );
        graph.metadata.insert(
            "method".to_string(),
            serde_json::Value::String("attention-walk".to_string()),
        );
    }

    let config = WalkConfig {
        top_k: args.top_k,
        min_score: args.min_score,
    };

    let all_layers: Vec<usize> = match args.layer {
        Some(l) => vec![l],
        None => (0..walker.num_layers()).collect(),
    };
    let pending: Vec<usize> = all_layers
        .iter()
        .filter(|l| !completed.contains(l))
        .copied()
        .collect();

    if pending.is_empty() {
        eprintln!("All layers already completed.");
        return Ok(());
    }

    let bar = ProgressBar::new(all_layers.len() as u64);
    bar.set_position(completed.len() as u64);
    bar.set_style(
        ProgressStyle::default_bar()
            .template("Layers: {pos}/{len}")
            .unwrap(),
    );

    let head_bar = ProgressBar::new(0);
    head_bar.set_style(
        ProgressStyle::default_bar()
            .template("{spinner} [{bar:40}] {pos}/{len} {msg}")
            .unwrap(),
    );

    let mut callbacks = ProgressCallbacks {
        bar,
        head_bar,
        output: args.output.clone(),
    };

    let start = Instant::now();
    let mut results: Vec<AttentionLayerResult> = Vec::new();

    for &layer in &pending {
        let result = walker.walk_layer(layer, &config, &mut graph, &mut callbacks)?;
        results.push(result);
    }

    callbacks.bar.finish();
    callbacks.head_bar.finish_and_clear();
    let elapsed = start.elapsed();

    let total_edges: usize = results.iter().map(|r| r.edges_found).sum();
    let total_heads: usize = results.iter().map(|r| r.heads_walked).sum();

    eprintln!("\nCompleted in {:.1}min", elapsed.as_secs_f64() / 60.0);
    eprintln!("  Layers walked:     {}", results.len());
    eprintln!("  Heads walked:      {total_heads}");
    eprintln!("  Edges extracted:   {total_edges}");
    eprintln!("  Total graph edges: {}", graph.edge_count());
    eprintln!("  Entities:          {}", graph.node_count());
    eprintln!("  Relations:         {}", graph.list_relations().len());

    // Layer stats ranking
    eprintln!("\n  Layer Confidence Ranking:");
    let mut ranked = results.clone();
    ranked.sort_by(|a, b| {
        b.stats
            .mean_confidence
            .partial_cmp(&a.stats.mean_confidence)
            .unwrap()
    });
    for (i, r) in ranked.iter().take(5).enumerate() {
        eprintln!(
            "    #{:2} L{:2}  mean={:.4}  max={:.4}  edges={}",
            i + 1,
            r.layer,
            r.stats.mean_confidence,
            r.stats.max_confidence,
            r.edges_found,
        );
    }
    if ranked.len() > 5 {
        eprintln!("    ... ({} more layers)", ranked.len() - 5);
    }

    // Write stats file if requested
    if let Some(stats_path) = &args.stats {
        let layer_stats: Vec<serde_json::Value> = results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "layer": r.layer,
                    "heads_walked": r.heads_walked,
                    "edges_found": r.edges_found,
                    "elapsed_ms": r.elapsed_ms,
                    "mean_confidence": r.stats.mean_confidence,
                    "max_confidence": r.stats.max_confidence,
                    "min_confidence": r.stats.min_confidence,
                    "mean_c_in": r.stats.mean_c_in,
                    "mean_c_out": r.stats.mean_c_out,
                })
            })
            .collect();

        let stats_json = serde_json::json!({
            "model": args.model,
            "method": "attention-walk",
            "total_heads": total_heads,
            "total_edges": total_edges,
            "total_graph_edges": graph.edge_count(),
            "total_entities": graph.node_count(),
            "total_relations": graph.list_relations().len(),
            "wall_time_secs": elapsed.as_secs_f64(),
            "layer_stats": layer_stats,
        });

        let formatted = serde_json::to_string_pretty(&stats_json)?;
        std::fs::write(stats_path, formatted)?;
        eprintln!("  Stats: {}", stats_path.display());
    }

    let size = std::fs::metadata(&args.output)?.len();
    eprintln!(
        "  Saved: {} ({:.1} MB)",
        args.output.display(),
        size as f64 / 1024.0 / 1024.0
    );

    Ok(())
}
