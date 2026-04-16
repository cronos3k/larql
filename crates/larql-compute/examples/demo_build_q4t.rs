//! Build Q4 interleaved file with transposed down weights.
//!
//! Layout per layer: [gate Q4 | up Q4 | down_T Q4]
//!   gate: [intermediate, hidden] Q4_0  — same as before
//!   up:   [intermediate, hidden] Q4_0  — same as before
//!   down: [hidden, intermediate] Q4_0  — TRANSPOSED for matvec
//!
//! The transposed down allows the Metal q4_matvec shader to compute
//! the down projection as a gather-reduce (one thread per output element)
//! instead of scatter-accumulate (thread conflicts).
//!
//! Usage:
//!   cargo run --release -p larql-compute --example build_q4_transposed -- \
//!     --vindex output/gemma3-4b-v2.vindex

#[cfg(unix)]
extern crate blas_src;

use std::io::Write;
use std::path::Path;
use std::time::Instant;
use larql_compute::cpu::q4::quantize_q4_0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut vindex_dir = String::new();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--vindex" { i += 1; vindex_dir = args[i].clone(); }
        i += 1;
    }
    if vindex_dir.is_empty() {
        return Err("Usage: --vindex <path>".into());
    }
    let dir = Path::new(&vindex_dir);

    let config_text = std::fs::read_to_string(dir.join("index.json"))?;
    let config: serde_json::Value = serde_json::from_str(&config_text)?;
    let num_layers = config["num_layers"].as_u64().unwrap() as usize;
    let hidden = config["hidden_size"].as_u64().unwrap() as usize;
    let inter = config["intermediate_size"].as_u64().unwrap() as usize;

    // Ensure hidden is multiple of 32 (for Q4 blocks) — it's 2560, which is 80×32 ✓
    // Ensure intermediate is multiple of 32 — it's 10240, which is 320×32 ✓
    assert!(hidden.is_multiple_of(32) && inter.is_multiple_of(32));

    let floats_per_gate = inter * hidden;
    let floats_per_up = inter * hidden;
    let _floats_per_down = inter * hidden; // same total, different layout

    let q4_per_gate = floats_per_gate / 32 * 18;
    let q4_per_up = floats_per_up / 32 * 18;
    let q4_per_down_t = (hidden * inter) / 32 * 18; // transposed: [hidden, inter]

    println!("=== Build Q4 Interleaved (Transposed Down) ===\n");
    println!("Layers: {num_layers}, hidden: {hidden}, intermediate: {inter}");
    println!("Per layer: gate {:.1}MB + up {:.1}MB + down_T {:.1}MB = {:.1}MB Q4",
        q4_per_gate as f64 / 1e6, q4_per_up as f64 / 1e6, q4_per_down_t as f64 / 1e6,
        (q4_per_gate + q4_per_up + q4_per_down_t) as f64 / 1e6);

    // Read source files
    let gate_file = std::fs::File::open(dir.join("gate_vectors.bin"))?;
    let gate_mmap = unsafe { memmap2::Mmap::map(&gate_file)? };
    let up_file = std::fs::File::open(dir.join("up_features.bin"))?;
    let up_mmap = unsafe { memmap2::Mmap::map(&up_file)? };
    let down_file = std::fs::File::open(dir.join("down_features.bin"))?;
    let down_mmap = unsafe { memmap2::Mmap::map(&down_file)? };

    let f32_per_layer = inter * hidden;
    let bytes_per_layer = f32_per_layer * 4;

    let out_path = dir.join("interleaved_q4t.bin");
    let mut out = std::io::BufWriter::with_capacity(16 * 1024 * 1024, std::fs::File::create(&out_path)?);

    let t0 = Instant::now();
    let mut total_bytes: u64 = 0;

    for layer in 0..num_layers {
        let offset = layer * bytes_per_layer;

        // Gate: [inter, hidden] — quantize as-is
        let gate_f32 = unsafe {
            let ptr = gate_mmap[offset..offset + bytes_per_layer].as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, f32_per_layer)
        };
        let gate_q4 = quantize_q4_0(gate_f32);
        out.write_all(&gate_q4)?;
        total_bytes += gate_q4.len() as u64;

        // Up: [inter, hidden] — quantize as-is
        let up_f32 = unsafe {
            let ptr = up_mmap[offset..offset + bytes_per_layer].as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, f32_per_layer)
        };
        let up_q4 = quantize_q4_0(up_f32);
        out.write_all(&up_q4)?;
        total_bytes += up_q4.len() as u64;

        // Down: [inter, hidden] → transpose to [hidden, inter] → quantize
        let down_f32 = unsafe {
            let ptr = down_mmap[offset..offset + bytes_per_layer].as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, f32_per_layer)
        };
        // Transpose: row i, col j of [inter, hidden] → row j, col i of [hidden, inter]
        let mut down_t = vec![0.0f32; hidden * inter];
        for r in 0..inter {
            for c in 0..hidden {
                down_t[c * inter + r] = down_f32[r * hidden + c];
            }
        }
        let down_t_q4 = quantize_q4_0(&down_t);
        out.write_all(&down_t_q4)?;
        total_bytes += down_t_q4.len() as u64;

        if layer % 10 == 0 || layer == num_layers - 1 {
            println!("  Layer {layer}: {:.1}MB", (gate_q4.len() + up_q4.len() + down_t_q4.len()) as f64 / 1e6);
        }
    }

    out.flush()?;
    println!("\nFile: {} ({:.1}MB, {:.1}s)",
        out_path.display(), total_bytes as f64 / 1e6, t0.elapsed().as_secs_f64());
    println!("Done.");
    Ok(())
}
