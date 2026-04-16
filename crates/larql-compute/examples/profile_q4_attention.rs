//! Benchmark Q4 attention projections: Q/K/V/O as Q4 matvec.
//!
//! Usage:
//!   cargo run --release -p larql-compute --features metal --example bench_q4_attention

#[cfg(unix)]
extern crate blas_src;

use std::time::Instant;
use ndarray::Array2;
use larql_compute::{default_backend, cpu_backend};
use larql_compute::cpu::q4;
use larql_compute::cpu::q4::quantize_q4_0;

fn main() {
    let hidden = 2560;
    let kv_dim = 512; // 4 KV heads × 128 dim (placeholder)
    let n = 20;
    let cpu = cpu_backend();
    let default = default_backend();

    println!("=== Q4 Attention Projection Benchmark ===");
    println!("CPU: {}, Default: {}\n", cpu.name(), default.name());

    // ── Per-layer: 4 attention projections ──
    let wq_f32: Vec<f32> = (0..hidden * hidden).map(|i| (i as f32 * 0.0001).cos()).collect();
    let wk_f32: Vec<f32> = (0..kv_dim * hidden).map(|i| (i as f32 * 0.0002).sin()).collect();
    let wq_q4 = quantize_q4_0(&wq_f32);
    let wk_q4 = quantize_q4_0(&wk_f32);

    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();
    let (q8_x, q8_s) = q4::quantize_to_q8(&x);

    println!("--- Single projection (seq=1) ---\n");

    // f32 BLAS Q proj
    {
        let wq_arr = Array2::from_shape_vec((hidden, hidden), wq_f32.clone()).unwrap();
        let x_arr = Array2::from_shape_vec((1, hidden), x.clone()).unwrap();
        let _ = cpu.matmul_transb(x_arr.view(), wq_arr.view());
        let t0 = Instant::now();
        for _ in 0..n { let _ = cpu.matmul_transb(x_arr.view(), wq_arr.view()); }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        println!("  f32 BLAS Q proj [1,2560]@[2560,2560]^T:  {ms:.2}ms");
    }

    // Q4 CPU Q proj
    {
        let _ = cpu.q4_matvec(&wq_q4, &q8_x, &q8_s, hidden, hidden);
        let t0 = Instant::now();
        for _ in 0..n { let _ = cpu.q4_matvec(&wq_q4, &q8_x, &q8_s, hidden, hidden); }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        println!("  CPU Q4 Q proj   [2560,2560] @ Q8:        {ms:.2}ms");
    }

    // Metal Q4 Q proj
    if default.has_q4() && default.name() != cpu.name() {
        let _ = default.q4_matvec(&wq_q4, &q8_x, &q8_s, hidden, hidden);
        let t0 = Instant::now();
        for _ in 0..n { let _ = default.q4_matvec(&wq_q4, &q8_x, &q8_s, hidden, hidden); }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        println!("  Metal Q4 Q proj [2560,2560] @ Q8:        {ms:.2}ms");
    }

    // K proj (smaller)
    {
        let wk_arr = Array2::from_shape_vec((kv_dim, hidden), wk_f32.clone()).unwrap();
        let x_arr = Array2::from_shape_vec((1, hidden), x.clone()).unwrap();
        let _ = cpu.matmul_transb(x_arr.view(), wk_arr.view());
        let t0 = Instant::now();
        for _ in 0..n { let _ = cpu.matmul_transb(x_arr.view(), wk_arr.view()); }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        println!("  f32 BLAS K proj [1,2560]@[512,2560]^T:   {ms:.2}ms");
    }

    if default.has_q4() && default.name() != cpu.name() {
        let _ = default.q4_matvec(&wk_q4, &q8_x, &q8_s, kv_dim, hidden);
        let t0 = Instant::now();
        for _ in 0..n { let _ = default.q4_matvec(&wk_q4, &q8_x, &q8_s, kv_dim, hidden); }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        println!("  Metal Q4 K proj [512,2560] @ Q8:         {ms:.2}ms");
    }

    // ── Full attention layer: Q+K+V+O (21 layers) ──
    println!("\n--- Full decode: 4 projections × 21 layers (seq=1) ---\n");

    {
        let wq_arr = Array2::from_shape_vec((hidden, hidden), wq_f32.clone()).unwrap();
        let wk_arr = Array2::from_shape_vec((kv_dim, hidden), wk_f32.clone()).unwrap();
        let x_arr = Array2::from_shape_vec((1, hidden), x.clone()).unwrap();
        let _ = cpu.matmul_transb(x_arr.view(), wq_arr.view());
        let t0 = Instant::now();
        for _ in 0..n {
            for _ in 0..21 {
                let _ = cpu.matmul_transb(x_arr.view(), wq_arr.view()); // Q
                let _ = cpu.matmul_transb(x_arr.view(), wk_arr.view()); // K
                let _ = cpu.matmul_transb(x_arr.view(), wk_arr.view()); // V
                let _ = cpu.matmul_transb(x_arr.view(), wq_arr.view()); // O
            }
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let tps = 1000.0 / ms;
        println!("  f32 BLAS attn (21L × 4 proj):  {ms:.1}ms  ({tps:.1} tok/s attn only)");
    }

    if default.has_q4() && default.name() != cpu.name() {
        let _ = default.q4_matvec(&wq_q4, &q8_x, &q8_s, hidden, hidden);
        let t0 = Instant::now();
        for _ in 0..n {
            for _ in 0..21 {
                let _ = default.q4_matvec(&wq_q4, &q8_x, &q8_s, hidden, hidden); // Q
                let _ = default.q4_matvec(&wk_q4, &q8_x, &q8_s, kv_dim, hidden);  // K
                let _ = default.q4_matvec(&wk_q4, &q8_x, &q8_s, kv_dim, hidden);  // V
                let _ = default.q4_matvec(&wq_q4, &q8_x, &q8_s, hidden, hidden); // O
            }
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let tps = 1000.0 / ms;
        println!("  Metal Q4 attn (21L × 4 proj):  {ms:.1}ms  ({tps:.1} tok/s attn only)");
    }

    // ── Projected full decode (attn + FFN) ──
    println!("\n--- Projected full decode (Q4 attn + Q4 FFN, 21 layers) ---\n");
    println!("  If Metal Q4 attn = ~Xms and Metal Q4 FFN = 21.8ms:");
    println!("  Total = Xms + 21.8ms + 5ms (logits) + 5ms (other)");

    println!("\n=== Done ===");
}
