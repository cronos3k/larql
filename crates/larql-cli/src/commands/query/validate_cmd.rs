use std::path::PathBuf;

use clap::Args;

#[derive(Args)]
pub struct ValidateArgs {
    /// Path to graph file (.larql.json or .larql.bin).
    graph: PathBuf,
}

pub fn run(args: ValidateArgs) -> Result<(), Box<dyn std::error::Error>> {
    let graph = larql_core::load(&args.graph)?;
    let stats = graph.stats();

    let mut warnings = Vec::new();

    // Check for zero-confidence edges
    let zero_conf = graph.edges().iter().filter(|e| e.confidence <= 0.0).count();
    if zero_conf > 0 {
        warnings.push(format!(
            "{zero_conf} edges with zero or negative confidence"
        ));
    }

    // Check for self-loops
    let self_loops = graph
        .edges()
        .iter()
        .filter(|e| e.subject == e.object)
        .count();
    if self_loops > 0 {
        warnings.push(format!("{self_loops} self-loop edges"));
    }

    // Check for empty subjects/objects
    let empty = graph
        .edges()
        .iter()
        .filter(|e| e.subject.trim().is_empty() || e.object.trim().is_empty())
        .count();
    if empty > 0 {
        warnings.push(format!("{empty} edges with empty subject or object"));
    }

    println!("Validated: {}", args.graph.display());
    println!(
        "  {} entities, {} edges, {} relations",
        stats.entities, stats.edges, stats.relations
    );

    if warnings.is_empty() {
        println!("  OK — no issues found");
    } else {
        println!("  Warnings:");
        for w in &warnings {
            println!("    - {w}");
        }
    }

    Ok(())
}
