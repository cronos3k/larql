//! Profile the inter-gemv compute in the walk FFN.
//! Measures each step independently to find the 77ms gap.
//!
//! Usage:
//!   cargo run --release -p larql-inference --example profile_ffn_compute -- \
//!     --vindex output/gemma3-4b-v2.vindex

use std::time::Instant;
use ndarray::Array2;
use larql_inference::InferenceModel;
use larql_inference::forward::forward_to_layer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut vindex_path = std::path::PathBuf::from("output/gemma3-4b-v2.vindex");
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--vindex" { i += 1; vindex_path = std::path::PathBuf::from(&args[i]); }
        i += 1;
    }

    let model = InferenceModel::load("google/gemma-3-4b-it")?;
    let weights = model.weights();
    let tokenizer = model.tokenizer();

    let mut cb = larql_vindex::SilentLoadCallbacks;
    let mut index = larql_vindex::VectorIndex::load_vindex(&vindex_path, &mut cb)?;
    index.load_down_features(&vindex_path)?;
    index.load_up_features(&vindex_path)?;
    match index.load_interleaved(&vindex_path) {
        Ok(()) => println!("Using interleaved file"),
        Err(_) => println!("Using separate files"),
    }

    let prompt = "The capital of France is";
    let encoding = tokenizer.encode(prompt, true).map_err(|e| format!("{e}"))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    let seq_len = token_ids.len();
    let hidden = weights.hidden_size;
    let intermediate = index.num_features(13); // typical layer

    println!("\n=== FFN Inter-Gemv Compute Profile ===");
    println!("seq={seq_len}, hidden={hidden}, intermediate={intermediate}\n");

    // Get a representative hidden state
    let h = forward_to_layer(weights, &token_ids, 13);
    let norm_offset = weights.arch.norm_weight_offset();
    let h_norm = larql_inference::forward::apply_norm(
        weights, &h, &weights.arch.post_attention_layernorm_key(13), norm_offset,
    );

    let n = 20;

    // 1. Pure gate gemv (one matrix read)
    let _gate_view = if index.has_interleaved() {
        index.interleaved_gate(13).unwrap()
    } else {
        // gate_scores_batch uses internal gate mmap
        // For direct profiling, use the gate via gate_scores_batch
        index.up_layer_matrix(13).unwrap() // placeholder — will use gate_scores_batch below
    };

    // gate_scores_batch
    let _ = index.gate_scores_batch(13, &h_norm);
    let t0 = Instant::now();
    for _ in 0..n { let _ = index.gate_scores_batch(13, &h_norm); }
    let gate_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let gate_scores = index.gate_scores_batch(13, &h_norm).unwrap();

    // 2. Pure up gemv
    let up_view = if index.has_interleaved() {
        index.interleaved_up(13).unwrap()
    } else {
        index.up_layer_matrix(13).unwrap()
    };
    let _ = h_norm.dot(&up_view.t());
    let t0 = Instant::now();
    for _ in 0..n { let _ = h_norm.dot(&up_view.t()); }
    let up_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let up_scores = h_norm.dot(&up_view.t());

    // 3. GEGLU only (silu(gate) * up)
    let _ = larql_inference::ffn::silu_gate_up(&gate_scores, &up_scores);
    let t0 = Instant::now();
    for _ in 0..n { let _ = larql_inference::ffn::silu_gate_up(&gate_scores, &up_scores); }
    let geglu_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let activation = larql_inference::ffn::silu_gate_up(&gate_scores, &up_scores);

    // 4. Pure down gemv
    let down_view = if index.has_interleaved() {
        index.interleaved_down(13).unwrap()
    } else {
        index.down_layer_matrix(13).unwrap()
    };
    let _ = activation.dot(&down_view);
    let t0 = Instant::now();
    for _ in 0..n { let _ = activation.dot(&down_view); }
    let down_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;

    // 5. Pre-FFN norm
    let _ = larql_inference::forward::apply_norm(
        weights, &h, &weights.arch.post_attention_layernorm_key(13), norm_offset,
    );
    let t0 = Instant::now();
    for _ in 0..n {
        let _ = larql_inference::forward::apply_norm(
            weights, &h, &weights.arch.post_attention_layernorm_key(13), norm_offset,
        );
    }
    let norm_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;

    // 6. Residual add
    let other = Array2::<f32>::ones((seq_len, hidden));
    let t0 = Instant::now();
    for _ in 0..n { let _ = &h + &other; }
    let add_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;

    // 7. Array2 allocation for activation [seq, intermediate]
    let t0 = Instant::now();
    for _ in 0..n {
        let a = Array2::<f32>::zeros((seq_len, intermediate));
        std::hint::black_box(&a);
    }
    let alloc_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;

    // 8. Full walk FFN layer (end-to-end)
    let walk_ffn = larql_inference::vindex::WalkFfn::new(weights, &index, 8092);
    use larql_inference::ffn::FfnBackend;
    let _ = walk_ffn.forward(13, &h_norm);
    let t0 = Instant::now();
    for _ in 0..n { let _ = walk_ffn.forward(13, &h_norm); }
    let walk_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;

    println!("--- Per-layer component times (warm, layer 13) ---\n");
    println!("  Gate gemv (105MB):    {gate_ms:>6.2}ms");
    println!("  Up gemv (105MB):      {up_ms:>6.2}ms");
    println!("  GEGLU (element-wise): {geglu_ms:>6.2}ms  ← [{seq_len}, {intermediate}] silu * up");
    println!("  Down gemv (105MB):    {down_ms:>6.2}ms");
    println!("  Pre-FFN norm:         {norm_ms:>6.2}ms");
    println!("  Residual add:         {add_ms:>6.2}ms");
    println!("  Alloc [s,i]:          {alloc_ms:>6.2}ms");
    println!();

    let sum = gate_ms + up_ms + geglu_ms + down_ms + norm_ms + add_ms + alloc_ms;
    let gap = walk_ms - sum;
    println!("  Component sum:        {sum:>6.2}ms");
    println!("  Walk FFN (measured):  {walk_ms:>6.2}ms");
    println!("  Unexplained gap:      {gap:>6.2}ms");
    println!();

    // Scale to 21 layers (L13-33)
    let layers = 21;
    println!("--- Scaled to {layers} layers ---\n");
    println!("  Gate reads:    {:.0}ms  ({:.1} GB/s)", gate_ms * layers as f64, 105.0 / gate_ms);
    println!("  Up reads:      {:.0}ms  ({:.1} GB/s)", up_ms * layers as f64, 105.0 / up_ms);
    println!("  GEGLU compute: {:.0}ms", geglu_ms * layers as f64);
    println!("  Down reads:    {:.0}ms  ({:.1} GB/s)", down_ms * layers as f64, 105.0 / down_ms);
    println!("  Norm+add+alloc:{:.0}ms", (norm_ms + add_ms + alloc_ms) * layers as f64);
    println!("  Total sum:     {:.0}ms", sum * layers as f64);
    println!("  Walk measured: {:.0}ms", walk_ms * layers as f64);

    println!("\n=== Done ===");
    Ok(())
}
