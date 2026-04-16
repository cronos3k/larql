//! Benchmark walk FFN at different sequence lengths.
//! Shows where cold-cache penalty closes and architecture advantage opens.
//!
//! Usage:
//!   cargo run --release -p larql-inference --example bench_seqlen -- \
//!     --vindex output/gemma3-4b-v2.vindex

#[cfg(unix)]
extern crate blas_src;

use std::time::Instant;
use ndarray::Array2;

use larql_inference::InferenceModel;
use larql_inference::ffn::FfnBackend;
use larql_inference::vindex::WalkFfn;
use larql_vindex::{SilentLoadCallbacks, VectorIndex};

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
    let hidden = weights.hidden_size;

    let mut cb = SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex(&vindex_path, &mut cb)?;
    index.load_down_features(&vindex_path)?;
    index.load_up_features(&vindex_path)?;
    let _ = index.load_interleaved(&vindex_path);

    let walk_ffn = WalkFfn::new(weights, &index, 8092);
    let dense_ffn = larql_inference::WeightFfn { weights };
    let intermediate = index.num_features(13);

    println!("=== Sequence Length Scaling Benchmark ===");
    println!("hidden={hidden}, intermediate={intermediate}\n");
    println!("{:>5} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "seq", "Dense/L", "Walk/L", "Speedup", "Dense BW", "Walk BW");

    for &seq in &[1, 6, 16, 32, 64, 128] {
        let x = Array2::<f32>::from_elem((seq, hidden), 0.01);
        let layer = 13;
        let n = 10;

        // Warmup
        let _ = dense_ffn.forward(layer, &x);
        let _ = walk_ffn.forward(layer, &x);

        // Dense FFN
        let t0 = Instant::now();
        for _ in 0..n { let _ = dense_ffn.forward(layer, &x); }
        let dense_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;

        // Walk FFN
        let t0 = Instant::now();
        for _ in 0..n { let _ = walk_ffn.forward(layer, &x); }
        let walk_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;

        let speedup = dense_ms / walk_ms;
        let weight_bytes = intermediate * hidden * 4 * 3; // gate+up+down
        let dense_bw = weight_bytes as f64 / dense_ms / 1e6;
        let walk_bw = weight_bytes as f64 / walk_ms / 1e6;

        println!("{seq:>5} {dense_ms:>9.2}ms {walk_ms:>9.2}ms {speedup:>9.2}x {dense_bw:>8.1}GB/s {walk_bw:>8.1}GB/s");
    }

    // Also measure all 21 layers (L13-33) at different seq lengths
    println!("\n--- Full L13-33 (21 layers) ---\n");
    println!("{:>5} {:>12} {:>12} {:>10}",
        "seq", "Dense 21L", "Walk 21L", "Speedup");

    for &seq in &[1, 6, 32, 64, 128] {
        let x = Array2::<f32>::from_elem((seq, hidden), 0.01);
        let n = 3;

        // Warmup
        for layer in 13..34 {
            let _ = dense_ffn.forward(layer, &x);
            let _ = walk_ffn.forward(layer, &x);
        }

        let t0 = Instant::now();
        for _ in 0..n {
            for layer in 13..34 { let _ = dense_ffn.forward(layer, &x); }
        }
        let dense_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;

        let t0 = Instant::now();
        for _ in 0..n {
            for layer in 13..34 { let _ = walk_ffn.forward(layer, &x); }
        }
        let walk_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;

        let speedup = dense_ms / walk_ms;
        println!("{seq:>5} {dense_ms:>11.1}ms {walk_ms:>11.1}ms {speedup:>9.2}x");
    }

    println!("\n=== Done ===");
    Ok(())
}
