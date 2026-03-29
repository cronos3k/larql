use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use indicatif::{ProgressBar, ProgressStyle};
use larql_core::*;

#[derive(Args)]
pub struct BfsArgs {
    /// Comma-separated seed entities.
    #[arg(short, long)]
    seeds: String,

    /// Output file (.larql.json or .larql.bin).
    #[arg(short, long)]
    output: PathBuf,

    /// Model endpoint URL (e.g., http://localhost:11434/v1).
    #[arg(short, long)]
    endpoint: Option<String>,

    /// Model name for the endpoint.
    #[arg(short, long)]
    model: Option<String>,

    /// Use mock provider (for testing/demos).
    #[arg(long)]
    mock: bool,

    /// Path to templates JSON file.
    #[arg(short, long)]
    templates: PathBuf,

    /// Maximum BFS depth.
    #[arg(long, default_value = "3")]
    max_depth: u32,

    /// Maximum entities to probe.
    #[arg(long, default_value = "1000")]
    max_entities: usize,

    /// Minimum edge confidence.
    #[arg(long, default_value = "0.3")]
    min_confidence: f64,

    /// Resume from checkpoint.
    #[arg(long)]
    resume: Option<PathBuf>,

    /// Path to mock knowledge JSON file (required with --mock).
    #[arg(long)]
    mock_knowledge: Option<PathBuf>,
}

struct ProgressCallbacks {
    bar: ProgressBar,
    checkpoint: Option<CheckpointLog>,
    last_save: Instant,
}

impl BfsCallbacks for ProgressCallbacks {
    fn on_entity(&mut self, entity: &str, depth: u32, visited: usize, queue: usize) {
        self.bar
            .set_message(format!("[d={depth}] {entity:30} queue={queue}"));
        self.bar.set_position(visited as u64);
    }

    fn on_edge(&mut self, edge: &Edge, _depth: u32) {
        if let Some(ref mut cp) = self.checkpoint {
            let _ = cp.append(edge);
        }
    }

    fn on_checkpoint(&mut self, _graph: &Graph) {
        if self.last_save.elapsed().as_secs() >= 30 {
            self.last_save = Instant::now();
        }
    }
}

#[allow(clippy::type_complexity)]
fn load_mock_knowledge(
    path: &PathBuf,
) -> Result<Vec<(String, String, f64)>, Box<dyn std::error::Error>> {
    let contents = std::fs::read_to_string(path)?;
    let value: serde_json::Value = serde_json::from_str(&contents)?;
    let mut entries = Vec::new();
    if let Some(arr) = value.as_array() {
        for item in arr {
            let prompt = item["prompt"].as_str().unwrap_or_default().to_string();
            let answer = item["answer"].as_str().unwrap_or_default().to_string();
            let prob = item["probability"].as_f64().unwrap_or(1.0);
            entries.push((prompt, answer, prob));
        }
    }
    Ok(entries)
}

pub fn run(args: BfsArgs) -> Result<(), Box<dyn std::error::Error>> {
    // Load templates from file
    let tmpl_contents = std::fs::read_to_string(&args.templates).map_err(|e| {
        format!(
            "Failed to read templates file '{}': {e}",
            args.templates.display()
        )
    })?;
    let tmpl_value: serde_json::Value = serde_json::from_str(&tmpl_contents)
        .map_err(|e| format!("Failed to parse templates JSON: {e}"))?;
    let templates = TemplateRegistry::from_json_value(&tmpl_value);

    // Build provider
    let provider: Box<dyn ModelProvider> = if args.mock {
        let knowledge = if let Some(ref kp) = args.mock_knowledge {
            load_mock_knowledge(kp)?
        } else {
            Vec::new()
        };
        Box::new(larql_core::engine::mock_provider::MockProvider::with_knowledge(knowledge))
    } else {
        let endpoint = args
            .endpoint
            .as_deref()
            .unwrap_or("http://localhost:11434/v1");
        let model = args
            .model
            .as_deref()
            .ok_or("--model required when not using --mock")?;
        Box::new(larql_core::engine::http_provider::HttpProvider::new(
            endpoint, model,
        ))
    };

    let seeds: Vec<String> = args
        .seeds
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();

    // Resume from checkpoint or start fresh
    let mut graph = if let Some(ref cp_path) = args.resume {
        let cp = CheckpointLog::open(cp_path)?;
        let g = cp.replay()?;
        eprintln!("Resumed: {} edges from checkpoint", g.edge_count());
        g
    } else {
        Graph::new()
    };

    let config = BfsConfig {
        max_depth: args.max_depth,
        max_entities: args.max_entities,
        min_confidence: args.min_confidence,
        ..Default::default()
    };

    let bar = ProgressBar::new(args.max_entities as u64);
    bar.set_style(
        ProgressStyle::default_bar()
            .template("{spinner} [{bar:40}] {pos}/{len} {msg}")
            .unwrap(),
    );

    let checkpoint = args.resume.as_ref().or(Some(&args.output)).and_then(|p| {
        let mut cp_path = p.clone();
        cp_path.set_extension("checkpoint");
        CheckpointLog::open(cp_path).ok()
    });

    let mut callbacks = ProgressCallbacks {
        bar,
        checkpoint,
        last_save: Instant::now(),
    };

    eprintln!("BFS extracting from: {}", seeds.join(", "));
    eprintln!("  Provider: {}", provider.model_name());
    eprintln!(
        "  Max depth: {}, Max entities: {}",
        config.max_depth, config.max_entities
    );

    let start = Instant::now();
    let result = extract_bfs(
        provider.as_ref(),
        &templates,
        &seeds,
        &config,
        &mut graph,
        &mut callbacks,
    );

    callbacks.bar.finish_with_message("done");
    let elapsed = start.elapsed();

    eprintln!("\nCompleted in {:.1}s", elapsed.as_secs_f64());
    eprintln!("  Entities visited:  {}", result.entities_visited);
    eprintln!("  Edges added:       {}", result.edges_added);
    eprintln!("  Forward passes:    {}", result.total_forward_passes);
    eprintln!("  Queue remaining:   {}", result.queue_remaining);
    eprintln!("  Total graph edges: {}", graph.edge_count());

    // Save final graph (format detected from output extension)
    larql_core::save(&graph, &args.output)?;
    let size = std::fs::metadata(&args.output)?.len();
    eprintln!(
        "  Saved: {} ({:.1} KB)",
        args.output.display(),
        size as f64 / 1024.0
    );

    Ok(())
}
