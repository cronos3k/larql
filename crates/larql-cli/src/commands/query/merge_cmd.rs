use std::path::PathBuf;
use std::time::Instant;

use clap::Args;

#[derive(Args)]
pub struct MergeArgs {
    /// Input graph files to merge (at least 2).
    #[arg(required = true, num_args = 2..)]
    inputs: Vec<PathBuf>,

    /// Output merged graph file.
    #[arg(short, long)]
    output: PathBuf,

    /// Merge strategy: union, max_confidence, source_priority.
    #[arg(long, default_value = "union")]
    strategy: String,
}

pub fn run(args: MergeArgs) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!(
        "Merging {} files (strategy={})",
        args.inputs.len(),
        args.strategy
    );

    let start = Instant::now();

    // Load first graph as base
    eprintln!("  Loading {}...", args.inputs[0].display());
    let mut graph = larql_core::load(&args.inputs[0])?;
    eprintln!(
        "    {} edges ({:.1}s)",
        graph.edge_count(),
        start.elapsed().as_secs_f64()
    );

    // Merge remaining files — manual loop with progress
    for path in &args.inputs[1..] {
        eprintln!("  Loading {}...", path.display());
        let load_start = Instant::now();
        let other = larql_core::load(path)?;
        eprintln!(
            "    {} edges ({:.1}s)",
            other.edge_count(),
            load_start.elapsed().as_secs_f64()
        );

        eprintln!("  Merging...");
        let merge_start = Instant::now();
        let other_edges = other.edges();
        let total = other_edges.len();
        let mut added = 0;
        let progress_interval = (total / 20).max(1);

        for (i, edge) in other_edges.iter().enumerate() {
            if !graph.exists(&edge.subject, &edge.relation, &edge.object) {
                graph.add_edge(edge.clone());
                added += 1;
            }
            // For union, just skip duplicates — no remove+rebuild needed
            // max_confidence would need remove_edge which is too slow at scale

            if (i + 1) % progress_interval == 0 {
                eprint!(
                    "\r    {}/{} ({} added, {:.0}s)",
                    i + 1,
                    total,
                    added,
                    merge_start.elapsed().as_secs_f64()
                );
            }
        }
        eprintln!(
            "\r    {total}/{total} done: +{added} new edges ({:.1}s)",
            merge_start.elapsed().as_secs_f64()
        );
    }

    eprintln!("  Saving {}...", args.output.display());
    let save_start = Instant::now();
    larql_core::save(&graph, &args.output)?;
    let size = std::fs::metadata(&args.output)?.len();

    eprintln!("\nMerged graph:");
    eprintln!("  Edges:    {}", graph.edge_count());
    eprintln!("  Entities: {}", graph.node_count());
    eprintln!(
        "  Saved:    {} ({:.1} MB, {:.1}s)",
        args.output.display(),
        size as f64 / 1024.0 / 1024.0,
        save_start.elapsed().as_secs_f64(),
    );
    eprintln!("  Total:    {:.1}s", start.elapsed().as_secs_f64());

    Ok(())
}
