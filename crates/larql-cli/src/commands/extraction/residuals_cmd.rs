use std::path::PathBuf;
use std::time::Instant;

use clap::{Args, Subcommand};
use larql_inference::{CaptureCallbacks, CaptureConfig, InferenceModel};

#[derive(Args)]
pub struct ResidualsArgs {
    #[command(subcommand)]
    command: ResidualsCommand,
}

#[derive(Subcommand)]
enum ResidualsCommand {
    /// Capture residual stream vectors and activation traces for entities.
    Capture(CaptureArgs),
}

#[derive(Args)]
struct CaptureArgs {
    /// Model path or HuggingFace model ID.
    model: String,

    /// Comma-separated entities, or path to a text file (one per line).
    #[arg(short, long)]
    entities: String,

    /// Layer to capture at. Can specify multiple times.
    #[arg(short, long)]
    layer: Option<Vec<usize>>,

    /// Capture at all layers.
    #[arg(long)]
    all_layers: bool,

    /// Output directory.
    #[arg(short, long)]
    output: PathBuf,

    /// Prompt template. {entity} is replaced with the entity name.
    /// Default: bare entity name (single-token trace).
    #[arg(long)]
    template: Option<String>,

    /// Also capture sparse FFN activations (top-K features per layer).
    #[arg(long)]
    activations: bool,

    /// Number of top features to record per layer when --activations is set.
    #[arg(long, default_value = "50")]
    activation_top_k: usize,
}

struct ProgressCallbacks;

impl CaptureCallbacks for ProgressCallbacks {
    fn on_entity_start(&mut self, entity: &str, index: usize, total: usize) {
        eprint!("  [{}/{}] {entity}...", index + 1, total);
    }

    fn on_entity_done(&mut self, _entity: &str, layers_captured: usize, elapsed_ms: f64) {
        eprintln!(" {layers_captured} layers ({:.1}s)", elapsed_ms / 1000.0);
    }
}

pub fn run(args: ResidualsArgs) -> Result<(), Box<dyn std::error::Error>> {
    match args.command {
        ResidualsCommand::Capture(capture) => run_capture(capture),
    }
}

fn run_capture(args: CaptureArgs) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading model: {}", args.model);
    let capturer = InferenceModel::load(&args.model)?;
    eprintln!(
        "  {} layers, hidden_size={}",
        capturer.num_layers(),
        capturer.hidden_size()
    );

    let entities: Vec<String> = if std::path::Path::new(&args.entities).exists() {
        std::fs::read_to_string(&args.entities)?
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect()
    } else {
        args.entities
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    };

    let layers: Vec<usize> = if args.all_layers {
        (0..capturer.num_layers()).collect()
    } else {
        args.layer.unwrap_or_else(|| vec![25])
    };

    eprintln!(
        "  entities: {} ({} total)",
        entities
            .iter()
            .take(5)
            .cloned()
            .collect::<Vec<_>>()
            .join(", "),
        entities.len()
    );
    eprintln!("  layers: {:?}", layers);
    if args.activations {
        eprintln!("  activations: top-{}", args.activation_top_k);
    }
    if let Some(ref tmpl) = args.template {
        eprintln!("  template: {tmpl}");
    }
    eprintln!();

    let config = CaptureConfig {
        layers,
        prompt_template: args.template,
        capture_activations: args.activations,
        activation_top_k: args.activation_top_k,
    };

    let mut callbacks = ProgressCallbacks;
    let start = Instant::now();

    let (res_count, act_count) =
        capturer.capture(&entities, &config, &args.output, &mut callbacks)?;

    let elapsed = start.elapsed();

    eprintln!("\nCompleted in {:.1}s", elapsed.as_secs_f64());
    eprintln!("  Residuals: {res_count}");
    if args.activations {
        eprintln!("  Activations: {act_count}");
    }
    eprintln!("  Entities: {}", entities.len());
    eprintln!("  Output: {}", args.output.display());

    // Show output file sizes
    let res_path = args.output.join("residuals.vectors.jsonl");
    if res_path.exists() {
        let size = std::fs::metadata(&res_path)?.len();
        eprintln!(
            "    residuals.vectors.jsonl ({:.1} KB)",
            size as f64 / 1024.0
        );
    }
    if args.activations {
        let act_path = args.output.join("activations.jsonl");
        if act_path.exists() {
            let size = std::fs::metadata(&act_path)?.len();
            eprintln!("    activations.jsonl ({:.1} KB)", size as f64 / 1024.0);
        }
    }

    Ok(())
}
