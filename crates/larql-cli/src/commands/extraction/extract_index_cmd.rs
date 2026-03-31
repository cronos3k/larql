use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use indicatif::{ProgressBar, ProgressStyle};
use larql_inference::vindex::IndexBuildCallbacks;
use larql_inference::{write_model_weights, InferenceModel};

#[derive(Args)]
pub struct ExtractIndexArgs {
    /// Model path or HuggingFace model ID (extracts directly from weights).
    /// Not needed if --from-vectors is used.
    model: Option<String>,

    /// Output path for the .vindex directory.
    #[arg(short, long)]
    output: PathBuf,

    /// Build from already-extracted NDJSON vector files instead of model weights.
    /// Point to the directory containing ffn_gate.vectors.jsonl, etc.
    #[arg(long)]
    from_vectors: Option<PathBuf>,

    /// Top-K tokens to store per feature in down metadata (only for model extraction).
    #[arg(long, default_value = "10")]
    down_top_k: usize,

    /// Include full model weights (attention + FFN + norms) for self-contained inference.
    /// Adds model_weights.bin (~4.5GB for Gemma 3 4B). Enables --predict without --model.
    #[arg(long)]
    include_weights: bool,

    /// Skip stages that already have output files (resume interrupted builds).
    #[arg(long)]
    resume: bool,
}

struct CliBuildCallbacks {
    stage_start: Option<Instant>,
    feature_bar: ProgressBar,
}

impl CliBuildCallbacks {
    fn new() -> Self {
        let feature_bar = ProgressBar::new(0);
        feature_bar.set_style(
            ProgressStyle::default_bar()
                .template("  {spinner} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("█▓░"),
        );
        feature_bar.set_draw_target(indicatif::ProgressDrawTarget::stderr());

        Self {
            stage_start: None,
            feature_bar,
        }
    }
}

impl IndexBuildCallbacks for CliBuildCallbacks {
    fn on_stage(&mut self, stage: &str) {
        self.feature_bar.finish_and_clear();
        eprintln!("\n── {stage} ──");
        self.stage_start = Some(Instant::now());
    }

    fn on_layer_start(&mut self, component: &str, layer: usize, total: usize) {
        self.feature_bar.reset();
        self.feature_bar
            .set_message(format!("{component} L{layer} ({}/{})", layer + 1, total));
    }

    fn on_feature_progress(
        &mut self,
        component: &str,
        _layer: usize,
        done: usize,
        total: usize,
    ) {
        if total > 0 {
            self.feature_bar.set_length(total as u64);
        }
        self.feature_bar.set_position(done as u64);
        if total == 0 {
            self.feature_bar
                .set_message(format!("{component} {done} records"));
        }
    }

    fn on_layer_done(&mut self, component: &str, layer: usize, elapsed_ms: f64) {
        self.feature_bar.finish_and_clear();
        eprintln!("  {component} L{layer:2}: {:.1}s", elapsed_ms / 1000.0);
    }

    fn on_stage_done(&mut self, stage: &str, _elapsed_ms: f64) {
        self.feature_bar.finish_and_clear();
        if let Some(start) = self.stage_start.take() {
            eprintln!("  {stage}: {:.1}s", start.elapsed().as_secs_f64());
        }
    }
}

pub fn run(args: ExtractIndexArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut callbacks = CliBuildCallbacks::new();
    let build_start = Instant::now();

    if let Some(ref vectors_dir) = args.from_vectors {
        // Build from existing NDJSON files
        eprintln!("Building vindex from vectors: {}", vectors_dir.display());
        eprintln!("Output: {}", args.output.display());

        larql_inference::vindex::build_vindex_from_vectors(vectors_dir, &args.output, &mut callbacks)?;

        if args.include_weights {
            // Need model for weights even when building from vectors
            let model_name = args.model.as_deref().ok_or(
                "--model required with --include-weights (need model to extract attention/norm weights)",
            )?;
            eprintln!("\nLoading model for weights: {}", model_name);
            let model = InferenceModel::load(model_name)?;
            write_model_weights(model.weights(), &args.output, &mut callbacks)?;
        }
    } else {
        // Build from model weights
        let model_name = args
            .model
            .as_deref()
            .ok_or("Either provide a model name or use --from-vectors")?;

        eprintln!("Loading model: {}", model_name);
        let start = Instant::now();
        let model = InferenceModel::load(model_name)?;
        eprintln!(
            "  {} layers, hidden_size={}, intermediate_size={}, vocab_size={} ({:.1}s)",
            model.num_layers(),
            model.hidden_size(),
            model.weights().intermediate_size,
            model.weights().vocab_size,
            start.elapsed().as_secs_f64()
        );

        eprintln!("\nBuilding index: {}", args.output.display());

        let output = &args.output;

        if args.resume {
            // Skip stages that already have output files
            let has_gate = output.join("gate_vectors.bin").exists();
            let has_embed = output.join("embeddings.bin").exists();
            let has_down = output.join("down_meta.jsonl").exists()
                && std::fs::metadata(output.join("down_meta.jsonl"))
                    .map(|m| m.len() > 1000)
                    .unwrap_or(false);
            let has_weights = output.join("model_weights.bin").exists();

            if has_gate && has_embed && has_down {
                eprintln!("  Resuming: gate_vectors, embeddings, down_meta exist — skipping");
                // Just write index.json, tokenizer, clustering, and optionally weights
                larql_inference::vindex::build_vindex_resume(
                    model.weights(),
                    model.tokenizer(),
                    model_name,
                    output,
                    &mut callbacks,
                )?;
            } else {
                eprintln!("  Resume: missing core files — full rebuild");
                larql_inference::vindex::build_vindex(
                    model.weights(),
                    model.tokenizer(),
                    model_name,
                    output,
                    args.down_top_k,
                    &mut callbacks,
                )?;
            }

            if args.include_weights && !has_weights {
                write_model_weights(model.weights(), output, &mut callbacks)?;
            } else if has_weights {
                eprintln!("  Resuming: model_weights.bin exists — skipping");
            }
        } else {
            larql_inference::vindex::build_vindex(
                model.weights(),
                model.tokenizer(),
                model_name,
                output,
                args.down_top_k,
                &mut callbacks,
            )?;

            if args.include_weights {
                write_model_weights(model.weights(), output, &mut callbacks)?;
            }
        }
    }

    callbacks.feature_bar.finish_and_clear();
    let build_elapsed = build_start.elapsed();

    // Print summary
    eprintln!("\n── Summary ──");
    eprintln!("  Output: {}", args.output.display());

    if build_elapsed.as_secs() >= 60 {
        eprintln!(
            "  Build time: {:.1}min",
            build_elapsed.as_secs_f64() / 60.0
        );
    } else {
        eprintln!("  Build time: {:.1}s", build_elapsed.as_secs_f64());
    }

    for name in &[
        "index.json",
        "gate_vectors.bin",
        "embeddings.bin",
        "down_meta.jsonl",
        "tokenizer.json",
        "model_weights.bin",
        "weight_manifest.json",
    ] {
        let path = args.output.join(name);
        if let Ok(meta) = std::fs::metadata(&path) {
            let size_mb = meta.len() as f64 / (1024.0 * 1024.0);
            if size_mb > 1024.0 {
                eprintln!("  {name}: {:.2} GB", size_mb / 1024.0);
            } else if size_mb > 0.1 {
                eprintln!("  {name}: {:.1} MB", size_mb);
            } else {
                let size_kb = meta.len() as f64 / 1024.0;
                eprintln!("  {name}: {:.1} KB", size_kb);
            }
        } else {
            eprintln!("  {name}: (not found)");
        }
    }

    let total_size: u64 = [
        "index.json",
        "gate_vectors.bin",
        "embeddings.bin",
        "down_meta.jsonl",
        "tokenizer.json",
        "model_weights.bin",
        "weight_manifest.json",
    ]
    .iter()
    .filter_map(|name| std::fs::metadata(args.output.join(name)).ok())
    .map(|m| m.len())
    .sum();
    eprintln!(
        "  Total: {:.2} GB",
        total_size as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    eprintln!("\nUsage:");
    eprintln!(
        "  larql walk --index {} -p \"The capital of France is\"",
        args.output.display()
    );

    Ok(())
}
