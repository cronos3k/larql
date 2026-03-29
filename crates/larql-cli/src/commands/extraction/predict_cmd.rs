use std::time::Instant;

use clap::Args;
use larql_inference::{predict, InferenceModel};

#[derive(Args)]
pub struct PredictArgs {
    /// Model path or HuggingFace model ID.
    model: String,

    /// Prompt text to predict the next token for.
    #[arg(short, long)]
    prompt: String,

    /// Number of top predictions to show.
    #[arg(short = 'k', long, default_value = "10")]
    top_k: usize,
}

pub fn run(args: PredictArgs) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading model: {}", args.model);
    let start = Instant::now();
    let model = InferenceModel::load(&args.model)?;
    let load_elapsed = start.elapsed();
    eprintln!(
        "  {} layers, hidden_size={} ({:.1}s)",
        model.num_layers(),
        model.hidden_size(),
        load_elapsed.as_secs_f64()
    );

    eprintln!("Prompt: {:?}", args.prompt);

    // Tokenize (add_special_tokens=true to prepend BOS for Gemma)
    let encoding = model
        .tokenizer()
        .encode(args.prompt.as_str(), true)
        .map_err(|e| format!("tokenize error: {e}"))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    eprintln!("  {} tokens: {:?}", token_ids.len(), token_ids);

    // Run full forward pass
    eprintln!("Running forward pass ({} layers)...", model.num_layers());
    let predict_start = Instant::now();
    let result = predict(model.weights(), model.tokenizer(), &token_ids, args.top_k);
    let predict_elapsed = predict_start.elapsed();
    eprintln!("  Forward pass: {:.1}s", predict_elapsed.as_secs_f64());

    // Print predictions
    println!();
    println!("Top-{} predictions:", args.top_k);
    for (i, (token, prob)) in result.predictions.iter().enumerate() {
        println!(
            "  {:2}. {:20} {:.4} ({:.2}%)",
            i + 1,
            token,
            prob,
            prob * 100.0
        );
    }

    Ok(())
}
