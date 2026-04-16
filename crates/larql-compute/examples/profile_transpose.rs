//! Benchmark: transposed down Q4 matvec vs original Q4 vecmat.
//!
//! The original down projection is a vecmat (scatter-accumulate, GPU-hostile).
//! The transposed version is a matvec (gather-reduce, GPU-friendly).
//!
//! Usage:
//!   cargo run --release -p larql-compute --example bench_down_transpose
//!   cargo run --release -p larql-compute --features metal --example bench_down_transpose

#[cfg(unix)]
extern crate blas_src;

use std::time::Instant;
use larql_compute::{default_backend, cpu_backend};
use larql_compute::cpu::q4;
use larql_compute::cpu::q4::quantize_q4_0;

fn main() {
    let hidden = 2560;
    let inter = 10240;
    let n = 20;

    let cpu = cpu_backend();
    let default = default_backend();

    println!("=== Down Projection: Transposed vs Original ===");
    println!("CPU: {}", cpu.name());
    println!("Default: {}\n", default.name());

    // Create down weight matrix [inter, hidden] and its transpose [hidden, inter]
    let down_f32: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.0001).cos()).collect();
    let mut down_t_f32 = vec![0.0f32; hidden * inter];
    for r in 0..inter {
        for c in 0..hidden {
            down_t_f32[c * inter + r] = down_f32[r * hidden + c];
        }
    }

    let down_q4 = quantize_q4_0(&down_f32);        // [inter, hidden] Q4
    let down_t_q4 = quantize_q4_0(&down_t_f32);    // [hidden, inter] Q4

    // Activation vector (sparse — ~20% nonzero, typical of GEGLU output)
    let activation: Vec<f32> = (0..inter).map(|i| {
        if i % 5 == 0 { (i as f32 * 0.01).sin() } else { 0.0 }
    }).collect();

    println!("--- Original: vecmat out[{hidden}] = act[{inter}] @ Q4[{inter},{hidden}] ---\n");

    // CPU vecmat (original)
    {
        let _ = cpu.q4_vecmat(&activation, &down_q4, inter, hidden);
        let t0 = Instant::now();
        for _ in 0..n { let _ = cpu.q4_vecmat(&activation, &down_q4, inter, hidden); }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        println!("  CPU vecmat:       {ms:>6.2}ms");
    }

    if default.has_q4() && default.name() != cpu.name() {
        let _ = default.q4_vecmat(&activation, &down_q4, inter, hidden);
        let t0 = Instant::now();
        for _ in 0..n { let _ = default.q4_vecmat(&activation, &down_q4, inter, hidden); }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        println!("  {} vecmat: {ms:>6.2}ms", default.name());
    }

    println!("\n--- Transposed: matvec out[{hidden}] = Q4_T[{hidden},{inter}] @ act_Q8[{inter}] ---\n");

    // Quantize activation to Q8 for matvec
    let (act_q8, act_scales) = q4::quantize_to_q8(&activation);

    // CPU matvec (transposed)
    {
        let _ = cpu.q4_matvec(&down_t_q4, &act_q8, &act_scales, hidden, inter);
        let t0 = Instant::now();
        for _ in 0..n { let _ = cpu.q4_matvec(&down_t_q4, &act_q8, &act_scales, hidden, inter); }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        println!("  CPU matvec:       {ms:>6.2}ms");
    }

    if default.has_q4() && default.name() != cpu.name() {
        let _ = default.q4_matvec(&down_t_q4, &act_q8, &act_scales, hidden, inter);
        let t0 = Instant::now();
        for _ in 0..n { let _ = default.q4_matvec(&down_t_q4, &act_q8, &act_scales, hidden, inter); }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        println!("  {} matvec: {ms:>6.2}ms", default.name());
    }

    // Verify correctness: both should produce similar output
    let vecmat_out = cpu.q4_vecmat(&activation, &down_q4, inter, hidden).unwrap();
    let matvec_out = cpu.q4_matvec(&down_t_q4, &act_q8, &act_scales, hidden, inter).unwrap();
    let max_diff: f32 = vecmat_out.iter().zip(matvec_out.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let avg_mag: f32 = vecmat_out.iter().map(|v| v.abs()).sum::<f32>() / hidden as f32;
    println!("\n  Correctness: max diff = {max_diff:.4}, avg magnitude = {avg_mag:.4}");
    println!("  Relative error: {:.2e}", max_diff / avg_mag.max(1e-10));

    println!("\n=== Done ===");
}
