use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use larql_vindex::{
    load_vindex_embeddings, load_vindex_tokenizer,
    IndexLoadCallbacks, SilentLoadCallbacks, VectorIndex, ndarray, tokenizers,
};
use larql_inference::{
    load_model_weights_from_vindex,
    predict_with_ffn, predict_with_router, InferenceModel, LayerFfnRouter, ModelWeights,
    SparseFfn, WeightFfn,
    vindex::WalkFfn,
};

#[derive(Args)]
pub struct WalkArgs {
    /// Prompt text to walk through the model.
    #[arg(short, long)]
    prompt: String,

    /// Path to a .vindex directory (self-contained, no model needed).
    #[arg(long)]
    index: Option<PathBuf>,

    /// Model path or HuggingFace model ID (needed for --predict/--compare,
    /// or when not using --index).
    #[arg(short, long)]
    model: Option<String>,

    /// Path to extracted ffn_gate vectors (alternative to --index).
    #[arg(long)]
    gate_vectors: Option<PathBuf>,

    /// Path to extracted ffn_down vectors (alternative to --index).
    #[arg(long)]
    down_vectors: Option<PathBuf>,

    /// Top-K features per layer for the gate KNN.
    #[arg(short = 'k', long, default_value = "10")]
    top_k: usize,

    /// Layers to walk. Comma-separated or range (e.g., "26,27,28" or "24-33").
    /// Default: all layers.
    #[arg(short, long)]
    layers: Option<String>,

    /// Number of top predictions to show.
    #[arg(long, default_value = "10")]
    predict_top_k: usize,

    /// Run full forward pass with walk FFN and show predictions (requires --model).
    #[arg(long)]
    predict: bool,

    /// Compare walk FFN predictions against dense ground truth (requires --model).
    #[arg(long)]
    compare: bool,

    /// Number of down tokens to show per feature.
    #[arg(long, default_value = "5")]
    down_top_k: usize,

    /// Show verbose loading and timing info.
    #[arg(short, long)]
    verbose: bool,
}

struct VerboseLoadCallbacks;

impl IndexLoadCallbacks for VerboseLoadCallbacks {
    fn on_file_start(&mut self, component: &str, path: &str) {
        eprintln!("Loading {component}: {path}");
    }
    fn on_progress(&mut self, records: usize) {
        eprint!("\r  {records} records...");
    }
    fn on_file_done(&mut self, component: &str, records: usize, elapsed_ms: f64) {
        eprintln!(
            "\r  {component}: {records} records ({:.1}s)",
            elapsed_ms / 1000.0
        );
    }
}

/// Log to stderr only if verbose.
macro_rules! vlog {
    ($verbose:expr, $($arg:tt)*) => {
        if $verbose { eprintln!($($arg)*); }
    };
}

pub fn run(args: WalkArgs) -> Result<(), Box<dyn std::error::Error>> {
    let verbose = args.verbose;
    let load_start = Instant::now();

    // Load the index — either from .vindex or from separate NDJSON files
    let index = if let Some(ref vindex_path) = args.index {
        vlog!(verbose, "Loading vindex: {}", vindex_path.display());
        if verbose {
            let mut cb = VerboseLoadCallbacks;
            VectorIndex::load_vindex(vindex_path, &mut cb)?
        } else {
            let mut cb = SilentLoadCallbacks;
            VectorIndex::load_vindex(vindex_path, &mut cb)?
        }
    } else if let Some(ref gate_path) = args.gate_vectors {
        let mut idx = if verbose {
            let mut cb = VerboseLoadCallbacks;
            VectorIndex::load_gates(gate_path, &mut cb)?
        } else {
            let mut cb = SilentLoadCallbacks;
            VectorIndex::load_gates(gate_path, &mut cb)?
        };
        if let Some(ref down_path) = args.down_vectors {
            if verbose {
                let mut cb = VerboseLoadCallbacks;
                idx.load_down_meta(down_path, &mut cb)?;
            } else {
                let mut cb = SilentLoadCallbacks;
                idx.load_down_meta(down_path, &mut cb)?;
            }
        }
        idx
    } else {
        return Err("Either --index (vindex directory) or --gate-vectors required".into());
    };

    vlog!(
        verbose,
        "Index: {} layers, {} gate vectors, {} down meta entries ({:.1}s)",
        index.num_layers,
        index.total_gate_vectors(),
        index.total_down_meta(),
        load_start.elapsed().as_secs_f64()
    );

    // Parse layer selection
    let all_layers = index.loaded_layers();
    let layers = match &args.layers {
        Some(spec) => parse_layer_spec(spec)?,
        None => all_layers.clone(),
    };

    if args.predict || args.compare {
        if let Some(model_name) = args.model.as_deref() {
            // Load from safetensors
            run_with_model(model_name, &args, &index, &layers)?;
        } else if let Some(ref vindex_path) = args.index {
            // Try loading weights from vindex
            run_with_vindex_weights(vindex_path, &args, &index, &layers, verbose)?;
        } else {
            return Err("--model or --index (with --include-weights) required for --predict".into());
        }
    } else if let Some(ref vindex_path) = args.index {
        run_vindex_walk(vindex_path, &args, &index, &layers)?;
    } else {
        let model_name = args.model.as_deref().ok_or(
            "--model required for embedding walk (or use --index for standalone)",
        )?;
        run_model_embedding_walk(model_name, &args, &index, &layers)?;
    }

    Ok(())
}

/// Walk using embeddings from the .vindex directory. No model needed.
fn run_vindex_walk(
    vindex_path: &std::path::Path,
    args: &WalkArgs,
    index: &VectorIndex,
    layers: &[usize],
) -> Result<(), Box<dyn std::error::Error>> {
    let verbose = args.verbose;

    vlog!(verbose, "Loading embeddings from vindex...");
    let (embed, embed_scale) = load_vindex_embeddings(vindex_path)?;
    let tokenizer = load_vindex_tokenizer(vindex_path)?;

    let encoding = tokenizer
        .encode(args.prompt.as_str(), true)
        .map_err(|e| format!("tokenize error: {e}"))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    vlog!(
        verbose,
        "Prompt: {:?} ({} tokens: {:?})",
        args.prompt,
        token_ids.len(),
        token_ids
    );

    let last_tok = *token_ids.last().ok_or("empty prompt")?;
    let embed_row = embed.row(last_tok as usize);
    let query: ndarray::Array1<f32> = embed_row.mapv(|v| v * embed_scale);

    let token_str = tokenizer
        .decode(&[last_tok], true)
        .unwrap_or_else(|_| format!("T{last_tok}"));
    vlog!(verbose, "Query: embedding for {:?} (T{last_tok})", token_str.trim());

    let walk_start = Instant::now();
    let trace = index.walk(&query, layers, args.top_k);
    let walk_ms = walk_start.elapsed().as_secs_f64() * 1000.0;

    print_walk_trace(&trace, args.down_top_k);

    eprintln!(
        "\nWalk: {} layers, top-{}, {:.1}ms ({:.2}ms/layer)",
        layers.len(),
        args.top_k,
        walk_ms,
        walk_ms / layers.len() as f64
    );

    Ok(())
}

/// Walk using the model's embedding for the last token as the query vector.
fn run_model_embedding_walk(
    model_name: &str,
    args: &WalkArgs,
    index: &VectorIndex,
    layers: &[usize],
) -> Result<(), Box<dyn std::error::Error>> {
    let verbose = args.verbose;

    vlog!(verbose, "Loading model: {}", model_name);
    let model = InferenceModel::load(model_name)?;
    let weights = model.weights();

    let encoding = model
        .tokenizer()
        .encode(args.prompt.as_str(), true)
        .map_err(|e| format!("tokenize error: {e}"))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    vlog!(
        verbose,
        "Prompt: {:?} ({} tokens: {:?})",
        args.prompt,
        token_ids.len(),
        token_ids
    );

    let last_tok = *token_ids.last().ok_or("empty prompt")?;
    let embed_scale = weights.arch.embed_scale();
    let embed_row = weights.embed.row(last_tok as usize);
    let query: ndarray::Array1<f32> = embed_row.mapv(|v| v * embed_scale);

    let token_str = model
        .tokenizer()
        .decode(&[last_tok], true)
        .unwrap_or_else(|_| format!("T{last_tok}"));
    vlog!(verbose, "Query: embedding for {:?} (T{last_tok})", token_str.trim());

    let walk_start = Instant::now();
    let trace = index.walk(&query, layers, args.top_k);
    let walk_ms = walk_start.elapsed().as_secs_f64() * 1000.0;

    print_walk_trace(&trace, args.down_top_k);

    eprintln!(
        "\nWalk: {} layers, top-{}, {:.1}ms ({:.2}ms/layer)",
        layers.len(),
        args.top_k,
        walk_ms,
        walk_ms / layers.len() as f64
    );

    Ok(())
}

/// Walk with full forward pass — uses WalkFfn as the FFN backend.
/// Walk with full forward pass — loads model from safetensors.
fn run_with_model(
    model_name: &str,
    args: &WalkArgs,
    index: &VectorIndex,
    _layers: &[usize],
) -> Result<(), Box<dyn std::error::Error>> {
    vlog!(args.verbose, "Loading model: {}", model_name);
    let model_start = Instant::now();
    let model = InferenceModel::load(model_name)?;
    vlog!(
        args.verbose,
        "  {} layers, hidden_size={} ({:.1}s)",
        model.num_layers(),
        model.hidden_size(),
        model_start.elapsed().as_secs_f64()
    );

    run_predict_inner(model.weights(), model.tokenizer(), args, index)
}

/// Walk with full forward pass — loads weights from vindex (no safetensors).
fn run_with_vindex_weights(
    vindex_path: &std::path::Path,
    args: &WalkArgs,
    index: &VectorIndex,
    _layers: &[usize],
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    vlog!(verbose, "Loading model weights from vindex...");
    let load_start = Instant::now();

    let mut cb: Box<dyn IndexLoadCallbacks> = if verbose {
        Box::new(VerboseLoadCallbacks)
    } else {
        Box::new(SilentLoadCallbacks)
    };
    let weights = load_model_weights_from_vindex(vindex_path, &mut *cb)?;
    let tokenizer = load_vindex_tokenizer(vindex_path)?;

    vlog!(
        verbose,
        "  {} layers, hidden_size={} ({:.1}s)",
        weights.num_layers,
        weights.hidden_size,
        load_start.elapsed().as_secs_f64()
    );

    run_predict_inner(&weights, &tokenizer, args, index)
}

/// Core predict logic shared by model and vindex paths.
fn run_predict_inner(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    args: &WalkArgs,
    index: &VectorIndex,
) -> Result<(), Box<dyn std::error::Error>> {
    let verbose = args.verbose;

    let encoding = tokenizer
        .encode(args.prompt.as_str(), true)
        .map_err(|e| format!("tokenize error: {e}"))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    vlog!(verbose, "Prompt: {:?} ({} tokens)", args.prompt, token_ids.len());

    // Walk FFN forward pass
    let walk_ffn = WalkFfn::new(weights, index, args.top_k);
    let start = Instant::now();
    let result = predict_with_ffn(
        weights,
        tokenizer,
        &token_ids,
        args.predict_top_k,
        &walk_ffn,
    );
    let walk_elapsed = start.elapsed();

    let trace = walk_ffn.take_trace();

    if verbose {
        println!("\n── Walk Trace ──");
        print_walk_trace(&trace, args.down_top_k);
        println!();
    }

    print_predictions("walk", &result.predictions);
    vlog!(verbose, "  Walk forward: {:.1}s", walk_elapsed.as_secs_f64());

    if args.compare {
        let start = Instant::now();
        let dense_result =
            larql_inference::predict(weights, tokenizer, &token_ids, args.predict_top_k);
        let dense_elapsed = start.elapsed();

        print_predictions("dense", &dense_result.predictions);
        vlog!(verbose, "  Dense forward: {:.1}s", dense_elapsed.as_secs_f64());

        let sparse_ffn = SparseFfn {
            weights,
            top_k: args.top_k,
        };
        let start = Instant::now();
        let sparse_result = predict_with_ffn(
            weights,
            tokenizer,
            &token_ids,
            args.predict_top_k,
            &sparse_ffn,
        );
        let sparse_elapsed = start.elapsed();

        print_predictions(&format!("sparse:{}", args.top_k), &sparse_result.predictions);
        vlog!(verbose, "  Sparse forward: {:.1}s", sparse_elapsed.as_secs_f64());

        let weight_ffn = WeightFfn { weights };
        let walk_ffn2 = WalkFfn::new(weights, index, args.top_k);
        let num_layers = weights.num_layers;
        let switch = num_layers * 3 / 4;
        let mut backends: Vec<&dyn larql_inference::FfnBackend> = vec![&weight_ffn; num_layers];
        (switch..num_layers).for_each(|l| {
            backends[l] = &walk_ffn2;
        });
        let router = LayerFfnRouter::per_layer(backends);
        let start = Instant::now();
        let hybrid_result = predict_with_router(
            weights,
            tokenizer,
            &token_ids,
            args.predict_top_k,
            &router,
        );
        let hybrid_elapsed = start.elapsed();

        print_predictions(
            &format!("hybrid (dense:0-{}, walk:{}-{})", switch - 1, switch, num_layers - 1),
            &hybrid_result.predictions,
        );
        vlog!(verbose, "  Hybrid forward: {:.1}s", hybrid_elapsed.as_secs_f64());

        println!();
        println!(
            "{:<40} {:<15} {:>8} {:>8}",
            "Backend", "Top-1", "Prob", "Time"
        );
        println!("{}", "-".repeat(75));
        print_summary_row("walk", &result.predictions, walk_elapsed);
        print_summary_row("dense", &dense_result.predictions, dense_elapsed);
        print_summary_row(&format!("sparse:{}", args.top_k), &sparse_result.predictions, sparse_elapsed);
        print_summary_row(
            &format!("dense:0-{},walk:{}-{}", switch - 1, switch, num_layers - 1),
            &hybrid_result.predictions,
            hybrid_elapsed,
        );
    }

    Ok(())
}

fn print_predictions(label: &str, predictions: &[(String, f64)]) {
    println!("\nTop predictions ({label}):");
    for (i, (token, prob)) in predictions.iter().enumerate() {
        println!(
            "  {:2}. {:20} ({:.2}%)",
            i + 1,
            token,
            prob * 100.0
        );
    }
}

fn print_summary_row(label: &str, predictions: &[(String, f64)], elapsed: std::time::Duration) {
    let (top1, prob1) = predictions
        .first()
        .map(|(t, p)| (t.as_str(), *p))
        .unwrap_or(("?", 0.0));
    println!(
        "{:<40} {:<15} {:>7.2}% {:>6.0}ms",
        label,
        top1,
        prob1 * 100.0,
        elapsed.as_secs_f64() * 1000.0,
    );
}

fn print_walk_trace(trace: &larql_vindex::WalkTrace, down_top_k: usize) {
    for (layer, hits) in &trace.layers {
        if hits.is_empty() {
            continue;
        }

        println!("Layer {layer}:");
        for (i, hit) in hits.iter().enumerate() {
            let down_tokens: String = hit
                .meta
                .top_k
                .iter()
                .take(down_top_k)
                .map(|t| format!("{} ({:.2})", t.token, t.logit))
                .collect::<Vec<_>>()
                .join(", ");

            println!(
                "  {:2}. F{:<5} gate={:+.3}  hears={:15}  c={:.2}  down=[{}]",
                i + 1,
                hit.feature,
                hit.gate_score,
                format!("{:?}", hit.meta.top_token),
                hit.meta.c_score,
                down_tokens,
            );
        }
    }
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
