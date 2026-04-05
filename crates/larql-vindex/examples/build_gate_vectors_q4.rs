//! Convert gate_vectors.bin (f32/f16) → gate_vectors_q4.bin (Q4_0).
//!
//! 7x smaller gate files for fast Q4 KNN via larql-compute.
//! The Q4 file is scored directly by the compute crate's Q4 matvec shader.
//!
//! Usage:
//!   cargo run --release -p larql-vindex --example build_gate_vectors_q4 -- <vindex_dir>

use std::io::Write;
use std::path::Path;
use std::time::Instant;
use larql_compute::cpu::q4::quantize_q4_0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args().nth(1)
        .unwrap_or_else(|| { eprintln!("Usage: build_gate_vectors_q4 <vindex_dir>"); std::process::exit(1); });
    let dir = Path::new(&dir);

    // Load config
    let config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("index.json"))?
    )?;
    let num_layers = config["num_layers"].as_u64().unwrap() as usize;
    let hidden_size = config["hidden_size"].as_u64().unwrap() as usize;
    let dtype = config.get("dtype").and_then(|v| v.as_str()).unwrap_or("f32");

    // Load gate_vectors.bin
    let gate_path = dir.join("gate_vectors.bin");
    let file = std::fs::File::open(&gate_path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };

    let layers_info: Vec<(usize, usize)> = config["layers"].as_array().unwrap()
        .iter()
        .map(|l| {
            let nf = l["num_features"].as_u64().unwrap_or(0) as usize;
            (nf, nf * hidden_size)
        })
        .collect();

    println!("=== Building gate_vectors_q4.bin ===");
    println!("  Source: {} ({} layers, {})", gate_path.display(), num_layers, dtype);

    let t0 = Instant::now();
    let out_path = dir.join("gate_vectors_q4.bin");
    let mut out = std::fs::File::create(&out_path)?;
    let mut total_f32 = 0usize;
    let mut total_q4 = 0usize;

    let bpf = if dtype == "f16" { 2 } else { 4 };
    let mut byte_offset = 0usize;

    for (layer, (num_features, num_floats)) in layers_info.iter().enumerate() {
        if *num_features == 0 { continue; }

        let byte_count = num_floats * bpf;
        let raw = &mmap[byte_offset..byte_offset + byte_count];

        // Decode to f32 if needed
        let f32_data: Vec<f32> = if dtype == "f16" {
            larql_models::quant::half::decode_f16(raw)
        } else {
            unsafe {
                let ptr = raw.as_ptr() as *const f32;
                std::slice::from_raw_parts(ptr, *num_floats).to_vec()
            }
        };

        let q4 = quantize_q4_0(&f32_data);
        out.write_all(&q4)?;

        total_f32 += num_floats * 4;
        total_q4 += q4.len();
        byte_offset += byte_count;

        if (layer + 1) % 10 == 0 || layer == num_layers - 1 {
            eprint!("\r  Layer {}/{}", layer + 1, num_layers);
        }
    }
    eprintln!();

    let elapsed = t0.elapsed().as_secs_f64();
    let ratio = total_f32 as f64 / total_q4 as f64;
    println!("  Output: {} ({:.1} MB, {:.1}x compression)",
        out_path.display(), total_q4 as f64 / 1e6, ratio);
    println!("  Time: {:.1}s", elapsed);
    println!("=== Done ===");

    Ok(())
}
