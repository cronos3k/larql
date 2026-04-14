//! Three-way Q4 benchmark: BLAS f32 vs C Q4 kernel vs Metal Q4 shader.
//!
//! Usage:
//!   cargo run --release -p larql-compute --example bench_q4
//!   cargo run --release -p larql-compute --features metal --example bench_q4

#[cfg(unix)]
extern crate blas_src;

use std::time::Instant;
use larql_compute::{default_backend, cpu_backend};
use larql_compute::cpu::q4;
use larql_compute::cpu::q4::quantize_q4_0;

fn main() {
    let hidden = 2560;
    let intermediate = 10240;
    let n = 20;

    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();
    let matrix: Vec<f32> = (0..intermediate * hidden).map(|i| (i as f32 * 0.0001).cos()).collect();
    let q4_data = quantize_q4_0(&matrix);

    let cpu = cpu_backend();
    let default = default_backend();

    println!("=== Q4 Benchmark ===");
    println!("Matrix: [{intermediate}, {hidden}] = {:.1}MB f32 → {:.1}MB Q4_0",
        (intermediate * hidden * 4) as f64 / 1e6, q4_data.len() as f64 / 1e6);
    println!("CPU: {}", cpu.name());
    println!("Default: {}\n", default.name());

    // 1. BLAS f32 gemv
    {
        let mat = ndarray::ArrayView2::from_shape((intermediate, hidden), &matrix).unwrap();
        let xv = ndarray::Array1::from_vec(x.clone());
        let _ = mat.dot(&xv);
        let t0 = Instant::now();
        for _ in 0..n { let _ = mat.dot(&xv); }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let gbps = (intermediate * hidden * 4) as f64 / ms / 1e6;
        println!("  BLAS f32 gemv:     {ms:>6.2}ms  ({gbps:>5.1} GB/s on {:.1}MB)",
            (intermediate * hidden * 4) as f64 / 1e6);
    }

    // 2. C Q4 kernel (via CPU backend)
    {
        let (q8_x, q8_scales) = q4::quantize_to_q8(&x);
        let _ = cpu.q4_matvec(&q4_data, &q8_x, &q8_scales, intermediate, hidden);
        let t0 = Instant::now();
        for _ in 0..n { let _ = cpu.q4_matvec(&q4_data, &q8_x, &q8_scales, intermediate, hidden); }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let gbps = q4_data.len() as f64 / ms / 1e6;
        println!("  CPU Q4 kernel:     {ms:>6.2}ms  ({gbps:>5.1} GB/s on {:.1}MB)",
            q4_data.len() as f64 / 1e6);
    }

    // 3. Default backend Q4 (Metal if available)
    if default.has_q4() && default.name() != cpu.name() {
        let (q8_x, q8_scales) = q4::quantize_to_q8(&x);
        let _ = default.q4_matvec(&q4_data, &q8_x, &q8_scales, intermediate, hidden);
        let t0 = Instant::now();
        for _ in 0..n { let _ = default.q4_matvec(&q4_data, &q8_x, &q8_scales, intermediate, hidden); }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let gbps = q4_data.len() as f64 / ms / 1e6;
        println!("  {} Q4:  {ms:>6.2}ms  ({gbps:>5.1} GB/s on {:.1}MB)",
            default.name(), q4_data.len() as f64 / 1e6);
    }

    println!("\n=== Done ===");
}

