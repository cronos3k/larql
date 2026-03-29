use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use indicatif::{ProgressBar, ProgressStyle};
use larql_inference::walker::vector_extractor::{
    ExtractCallbacks, ExtractConfig, VectorExtractor, ALL_COMPONENTS,
};

#[derive(Args)]
pub struct VectorExtractArgs {
    /// Model path or HuggingFace model ID (e.g. google/gemma-3-4b-it).
    model: String,

    /// Output directory for .vectors.jsonl files.
    #[arg(short, long)]
    output: PathBuf,

    /// Components to extract (comma-separated).
    /// Options: ffn_down,ffn_gate,ffn_up,attn_ov,attn_qk,embeddings
    #[arg(long, value_delimiter = ',')]
    components: Option<Vec<String>>,

    /// Layers to extract (comma-separated). Default: all.
    #[arg(long, value_delimiter = ',')]
    layers: Option<Vec<usize>>,

    /// Top-k tokens for metadata per vector.
    #[arg(long, default_value = "10")]
    top_k: usize,

    /// Resume from existing output files.
    #[arg(long)]
    resume: bool,
}

struct ProgressCallbacks {
    component_bar: ProgressBar,
    feature_bar: ProgressBar,
}

impl ExtractCallbacks for ProgressCallbacks {
    fn on_component_start(&mut self, component: &str, total_layers: usize) {
        self.component_bar.set_length(total_layers as u64);
        self.component_bar.set_position(0);
        self.component_bar
            .set_message(format!("{component}: {total_layers} layers"));
    }

    fn on_layer_start(&mut self, component: &str, layer: usize, num_vectors: usize) {
        self.feature_bar.set_length(num_vectors as u64);
        self.feature_bar.set_position(0);
        self.feature_bar
            .set_message(format!("{component} L{layer}: {num_vectors} vectors"));
    }

    fn on_progress(&mut self, component: &str, layer: usize, done: usize, total: usize) {
        self.feature_bar.set_position(done as u64);
        self.feature_bar
            .set_message(format!("{component} L{layer}: {done}/{total}"));
    }

    fn on_layer_done(
        &mut self,
        component: &str,
        layer: usize,
        vectors_written: usize,
        elapsed_ms: f64,
    ) {
        self.feature_bar
            .set_position(self.feature_bar.length().unwrap_or(0));
        self.component_bar.inc(1);
        eprintln!(
            "  {component} L{layer:2}: {vectors_written:6} vectors  ({:.0}s)",
            elapsed_ms / 1000.0,
        );
    }

    fn on_component_done(&mut self, component: &str, total_written: usize) {
        eprintln!("  {component}: {total_written} vectors total\n");
    }
}

pub fn run(args: VectorExtractArgs) -> Result<(), Box<dyn std::error::Error>> {
    // Validate components
    let components: Vec<String> = match args.components {
        Some(ref cs) => {
            for c in cs {
                if !ALL_COMPONENTS.contains(&c.as_str()) {
                    return Err(format!(
                        "unknown component: {c}. Options: {}",
                        ALL_COMPONENTS.join(", ")
                    )
                    .into());
                }
            }
            cs.clone()
        }
        None => ALL_COMPONENTS.iter().map(|s| s.to_string()).collect(),
    };

    eprintln!("Loading model: {}", args.model);
    let extractor = VectorExtractor::load(&args.model)?;
    eprintln!(
        "  {} layers, hidden_size={}",
        extractor.num_layers(),
        extractor.hidden_size()
    );
    eprintln!("  components: {}", components.join(", "));
    if let Some(ref layers) = args.layers {
        eprintln!("  layers: {:?}", layers);
    }
    eprintln!();

    let config = ExtractConfig {
        components,
        layers: args.layers,
        top_k: args.top_k,
    };

    let component_bar = ProgressBar::new(0);
    component_bar.set_style(
        ProgressStyle::default_bar()
            .template("Layers: {pos}/{len} {msg}")
            .unwrap(),
    );

    let feature_bar = ProgressBar::new(0);
    feature_bar.set_style(
        ProgressStyle::default_bar()
            .template("{spinner} [{bar:40}] {pos}/{len} {msg}")
            .unwrap(),
    );

    let mut callbacks = ProgressCallbacks {
        component_bar,
        feature_bar,
    };

    let start = Instant::now();
    let summary = extractor.extract_all(&config, &args.output, args.resume, &mut callbacks)?;

    callbacks.component_bar.finish();
    callbacks.feature_bar.finish_and_clear();

    let elapsed = start.elapsed();
    eprintln!("Completed in {:.1}min", elapsed.as_secs_f64() / 60.0);
    eprintln!("  Total vectors: {}", summary.total_vectors);

    for cs in &summary.components {
        if cs.vectors_written == 0 {
            eprintln!("  {}: skipped (not yet implemented)", cs.component);
            continue;
        }
        let size = std::fs::metadata(&cs.output_path)
            .map(|m| m.len())
            .unwrap_or(0);
        eprintln!(
            "  {}: {} vectors ({:.1} MB) → {}",
            cs.component,
            cs.vectors_written,
            size as f64 / 1024.0 / 1024.0,
            cs.output_path.display(),
        );
    }

    Ok(())
}
