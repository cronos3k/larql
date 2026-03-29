use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

use crate::utils::round4;
use clap::Args;
use indicatif::{ProgressBar, ProgressStyle};
use larql_core::*;
use larql_inference::walker::weight_walker::{
    LayerResult, WalkCallbacks, WalkConfig, WeightWalker,
};

#[derive(Args)]
pub struct WeightWalkArgs {
    /// Model path or HuggingFace model ID (e.g. google/gemma-3-4b-it).
    model: String,

    /// Output file (.larql.json or .larql.bin).
    #[arg(short, long)]
    output: PathBuf,

    /// Single layer to walk. Default: all layers.
    #[arg(short, long)]
    layer: Option<usize>,

    /// Top-k tokens per feature.
    #[arg(long, default_value = "5")]
    top_k: usize,

    /// Minimum activation score for top-k selection.
    #[arg(long, default_value = "0.02")]
    min_score: f32,

    /// Minimum normalized confidence to keep an edge (0.0-1.0). Filters during extraction.
    #[arg(long, default_value = "0.0")]
    min_confidence: f64,

    /// Write layer statistics to a separate file.
    #[arg(long)]
    stats: Option<PathBuf>,
}

struct ProgressCallbacks {
    bar: ProgressBar,
    feature_bar: ProgressBar,
    output: PathBuf,
}

impl WalkCallbacks for ProgressCallbacks {
    fn on_layer_start(&mut self, layer: usize, num_features: usize) {
        self.feature_bar.set_length(num_features as u64);
        self.feature_bar.set_position(0);
        self.feature_bar
            .set_message(format!("L{layer}: {num_features} features"));
    }

    fn on_progress(&mut self, layer: usize, features_done: usize, total: usize) {
        self.feature_bar.set_position(features_done as u64);
        self.feature_bar
            .set_message(format!("L{layer}: {features_done}/{total}"));
    }

    fn on_layer_done(&mut self, result: &LayerResult) {
        self.feature_bar
            .set_position(result.features_scanned as u64);
        self.bar.inc(1);
        let s = &result.stats;
        eprintln!(
            "  L{:2}: {:6} edges  ({:.0}s)  conf={:.4}  sel={:.4}  c_in={:.2}  c_out={:.2}  loops={:.1}%",
            result.layer,
            result.edges_found,
            result.elapsed_ms / 1000.0,
            s.mean_confidence,
            s.mean_selectivity,
            s.mean_c_in,
            s.mean_c_out,
            s.self_loop_pct,
        );
    }

    fn on_checkpoint(&mut self, graph: &Graph) {
        let _ = larql_core::save(graph, &self.output);
    }
}

pub fn run(args: WeightWalkArgs) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading model: {}", args.model);
    let walker = WeightWalker::load(&args.model)?;
    eprintln!("  {} layers", walker.num_layers());

    // Resume: load existing output and find completed layers
    let mut graph: Graph;
    let mut completed: HashSet<usize> = HashSet::new();

    if args.output.exists() {
        graph = larql_core::load(&args.output)?;
        // Scan edges for completed layers
        for edge in graph.edges() {
            if let Some(meta) = &edge.metadata {
                if let Some(layer) = meta.get("layer").and_then(|v| v.as_u64()) {
                    completed.insert(layer as usize);
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
            serde_json::Value::String("weight-walk".to_string()),
        );
    }

    let config = WalkConfig {
        top_k: args.top_k,
        min_score: args.min_score,
    };

    // Determine which layers to walk
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

    let feature_bar = ProgressBar::new(0);
    feature_bar.set_style(
        ProgressStyle::default_bar()
            .template("{spinner} [{bar:40}] {pos}/{len} {msg}")
            .unwrap(),
    );

    let mut callbacks = ProgressCallbacks {
        bar,
        feature_bar,
        output: args.output.clone(),
    };

    let start = Instant::now();
    let mut results = Vec::new();

    for &layer in &pending {
        let result = walker.walk_layer(layer, &config, &mut graph, &mut callbacks)?;
        results.push(result);
    }

    callbacks.bar.finish();
    callbacks.feature_bar.finish_and_clear();
    let elapsed = start.elapsed();

    let total_edges: usize = results.iter().map(|r| r.edges_found).sum();
    let total_features: usize = results.iter().map(|r| r.features_scanned).sum();

    // Apply min-confidence filter if set
    if args.min_confidence > 0.0 {
        let before = graph.edge_count();
        let kept: Vec<Edge> = graph
            .edges()
            .iter()
            .filter(|e| e.confidence >= args.min_confidence)
            .cloned()
            .collect();
        let mut filtered = Graph::new();
        filtered.metadata = graph.metadata.clone();
        for edge in kept {
            filtered.add_edge(edge);
        }
        let removed = before - filtered.edge_count();
        graph = filtered;
        eprintln!(
            "\n  Filtered: removed {} edges below c={:.2}, kept {}",
            removed,
            args.min_confidence,
            graph.edge_count()
        );
    }

    eprintln!("\nCompleted in {:.1}min", elapsed.as_secs_f64() / 60.0);
    eprintln!("  Layers walked:     {}", results.len());
    eprintln!("  Features scanned:  {total_features}");
    eprintln!("  Edges extracted:   {total_edges}");
    eprintln!("  Total graph edges: {}", graph.edge_count());
    eprintln!("  Entities:          {}", graph.node_count());
    eprintln!("  Relations:         {}", graph.list_relations().len());

    // Layer rankings — by selectivity (factual signal) and by confidence (combined)
    eprintln!("\n  By Selectivity (factual signal):");
    let mut by_sel = results.clone();
    by_sel.sort_by(|a, b| {
        b.stats
            .mean_selectivity
            .partial_cmp(&a.stats.mean_selectivity)
            .unwrap()
    });
    for (i, r) in by_sel.iter().take(5).enumerate() {
        eprintln!(
            "    #{:2} L{:2}  sel={:.4}  conf={:.4}  c_in={:.2}  loops={:.1}%  edges={}",
            i + 1,
            r.layer,
            r.stats.mean_selectivity,
            r.stats.mean_confidence,
            r.stats.mean_c_in,
            r.stats.self_loop_pct,
            r.edges_found,
        );
    }

    eprintln!("\n  By Confidence (combined c_in×c_out):");
    let mut by_conf = results.clone();
    by_conf.sort_by(|a, b| {
        b.stats
            .mean_confidence
            .partial_cmp(&a.stats.mean_confidence)
            .unwrap()
    });
    for (i, r) in by_conf.iter().take(5).enumerate() {
        eprintln!(
            "    #{:2} L{:2}  conf={:.4}  sel={:.4}  c_out={:.2}  loops={:.1}%  edges={}",
            i + 1,
            r.layer,
            r.stats.mean_confidence,
            r.stats.mean_selectivity,
            r.stats.mean_c_out,
            r.stats.self_loop_pct,
            r.edges_found,
        );
    }

    // Write stats file if requested
    if let Some(stats_path) = &args.stats {
        let layer_stats: Vec<serde_json::Value> = results
            .iter()
            .map(|r| {
                let s = &r.stats;
                let top_subj: Vec<serde_json::Value> = s
                    .top_subjects
                    .iter()
                    .map(|(name, count, avg_c)| {
                        serde_json::json!({"entity": name, "count": count, "avg_confidence": round4(*avg_c)})
                    })
                    .collect();
                let top_obj: Vec<serde_json::Value> = s
                    .top_objects
                    .iter()
                    .map(|(name, count, avg_c)| {
                        serde_json::json!({"entity": name, "count": count, "avg_confidence": round4(*avg_c)})
                    })
                    .collect();
                let ct = &s.threshold_counts;
                let st = &s.selectivity_threshold_counts;
                serde_json::json!({
                    "layer": r.layer,
                    "features_scanned": r.features_scanned,
                    "edges_found": r.edges_found,
                    "elapsed_ms": r.elapsed_ms,
                    "mean_confidence": round4(s.mean_confidence),
                    "max_confidence": round4(s.max_confidence),
                    "mean_selectivity": round4(s.mean_selectivity),
                    "max_selectivity": round4(s.max_selectivity),
                    "mean_c_in": round4(s.mean_c_in),
                    "mean_c_out": round4(s.mean_c_out),
                    "self_loop_count": s.self_loop_count,
                    "self_loop_pct": round4(s.self_loop_pct),
                    "confidence_thresholds": {
                        "c>=0.01": ct.t_01, "c>=0.05": ct.t_05,
                        "c>=0.10": ct.t_10, "c>=0.25": ct.t_25,
                        "c>=0.50": ct.t_50, "c>=0.75": ct.t_75,
                        "c>=0.90": ct.t_90,
                    },
                    "selectivity_thresholds": {
                        "s>=0.01": st.t_01, "s>=0.05": st.t_05,
                        "s>=0.10": st.t_10, "s>=0.25": st.t_25,
                        "s>=0.50": st.t_50, "s>=0.75": st.t_75,
                        "s>=0.90": st.t_90,
                    },
                    "top_subjects": top_subj,
                    "top_objects": top_obj,
                })
            })
            .collect();

        let stats_json = serde_json::json!({
            "model": args.model,
            "total_features_scanned": total_features,
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

    // Save final graph
    larql_core::save(&graph, &args.output)?;
    let size = std::fs::metadata(&args.output)?.len();
    eprintln!(
        "  Saved: {} ({:.1} MB)",
        args.output.display(),
        size as f64 / 1024.0 / 1024.0
    );

    Ok(())
}
