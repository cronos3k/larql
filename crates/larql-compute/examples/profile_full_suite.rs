//! Full benchmark suite for larql-compute.
//!
//! Tests every operation that inference and vindex need, at real matrix sizes,
//! with both CPU and Metal backends. Proves the crate is production-ready
//! before wiring into the pipeline.
//!
//! Usage:
//!   cargo run --release -p larql-compute --example bench_full
//!   cargo run --release -p larql-compute --features metal --example bench_full

#[cfg(unix)]
extern crate blas_src;

use std::time::Instant;
use ndarray::Array2;
use larql_compute::{default_backend, cpu_backend};
use larql_compute::cpu::q4;
use larql_compute::cpu::q4::quantize_q4_0;

fn synth(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut s = seed;
    Array2::from_shape_fn((rows, cols), |_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    })
}

struct Bench {
    n: usize,
}

impl Bench {
    fn run<F: FnMut()>(&self, name: &str, data_bytes: usize, mut f: F) {
        // Warmup
        f();
        let t0 = Instant::now();
        for _ in 0..self.n { f(); }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / self.n as f64;
        let gbps = data_bytes as f64 / ms / 1e6;
        println!("  {name:40} {ms:>7.2}ms  {gbps:>6.1} GB/s");
    }
}

fn main() {
    let cpu = cpu_backend();
    let default = default_backend();
    let bench = Bench { n: 20 };

    let hidden = 2560;
    let inter = 10240;
    let vocab = 262144;

    println!("=== larql-compute Full Benchmark Suite ===");
    println!("CPU:     {}", cpu.name());
    println!("Default: {} ({})", default.name(), default.device_info());
    println!();

    // ── 1. f32 matmul_transb at real sizes ──
    println!("--- 1. f32 matmul_transb (a @ b^T) ---\n");

    let sizes: Vec<(&str, usize, usize, usize)> = vec![
        ("Attention Q/O proj",  6, 2560, 2560),
        ("Attention K/V proj",  6, 512, 2560),
        ("FFN gate/up",         6, inter, hidden),
        ("Gate KNN (vindex)",   1, inter, hidden),
        ("Logits (262K vocab)", 1, vocab, hidden),
    ];

    for (label, m, n, k) in &sizes {
        let a = synth(*m, *k, 42);
        let b = synth(*n, *k, 43);
        let bytes = *n * *k * 4; // weight matrix read
        println!("  [{m},{k}] @ [{n},{k}]^T = [{m},{n}]  ({label})");
        bench.run("    CPU", bytes, || { let _ = cpu.matmul_transb(a.view(), b.view()); });
        if default.name() != cpu.name() {
            bench.run(default.name(), bytes, || { let _ = default.matmul_transb(a.view(), b.view()); });
        }
    }

    // ── 2. f32 matmul (non-transposed, FFN down) ──
    println!("\n--- 2. f32 matmul (a @ b, FFN down) ---\n");
    {
        let act = synth(6, inter, 44);
        let down = synth(inter, hidden, 45);
        let bytes = inter * hidden * 4;
        bench.run("CPU  [6,10240] @ [10240,2560]", bytes, || { let _ = cpu.matmul(act.view(), down.view()); });
        if default.name() != cpu.name() {
            bench.run(&format!("{}  [6,10240] @ [10240,2560]", default.name()), bytes,
                || { let _ = default.matmul(act.view(), down.view()); });
        }
    }

    // ── 3. Q4 matvec (gate or up) ──
    println!("\n--- 3. Q4 matvec (scores = Q4[N,K] @ Q8_x[K]) ---\n");
    {
        let matrix: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.0001).cos()).collect();
        let q4_data = quantize_q4_0(&matrix);
        let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();
        let (q8_x, q8_scales) = q4::quantize_to_q8(&x);
        let bytes = q4_data.len();

        bench.run("CPU C kernel", bytes, || {
            let _ = cpu.q4_matvec(&q4_data, &q8_x, &q8_scales, inter, hidden);
        });
        if default.has_q4() && default.name() != cpu.name() {
            bench.run(default.name(), bytes, || {
                let _ = default.q4_matvec(&q4_data, &q8_x, &q8_scales, inter, hidden);
            });
        }
    }

    // ── 4. Q4 vecmat (down projection) ──
    println!("\n--- 4. Q4 vecmat (out = act @ Q4[N,K]) ---\n");
    {
        let matrix: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.0001).cos()).collect();
        let q4_data = quantize_q4_0(&matrix);
        let activation: Vec<f32> = (0..inter).map(|i| if i % 5 == 0 { 1.0 } else { 0.0 }).collect();
        let bytes = q4_data.len();

        bench.run("CPU C kernel", bytes, || {
            let _ = cpu.q4_vecmat(&activation, &q4_data, inter, hidden);
        });
        if default.has_q4() && default.name() != cpu.name() {
            bench.run(default.name(), bytes, || {
                let _ = default.q4_vecmat(&activation, &q4_data, inter, hidden);
            });
        }
    }

    // ── 5. Q4 batched gate+up (6 seq positions) ──
    println!("\n--- 5. Q4 batched gate+up (6 positions, 1 submission) ---\n");
    {
        let gate_f32: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.0001).cos()).collect();
        let up_f32: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.0002).sin()).collect();
        let gate_q4 = quantize_q4_0(&gate_f32);
        let up_q4 = quantize_q4_0(&up_f32);
        let x_matrix: Vec<f32> = (0..6 * hidden).map(|i| (i as f32 * 0.001).sin()).collect();
        let bytes = gate_q4.len() + up_q4.len();

        if default.has_q4() {
            let result = default.q4_matvec_pair_batch(&gate_q4, &up_q4, &x_matrix, 6, inter, hidden);
            if let Some((gate_scores, up_scores)) = result {
                println!("    Batch returned: {} gate × {} up scores per position",
                    gate_scores[0].len(), up_scores[0].len());
                bench.run(&format!("{} pair_batch", default.name()), bytes, || {
                    let _ = default.q4_matvec_pair_batch(&gate_q4, &up_q4, &x_matrix, 6, inter, hidden);
                });
            } else {
                println!("    pair_batch not supported by {}", default.name());
            }
        }

        // Compare: 6 × 2 individual calls
        {
            let (_q8_x, _q8_scales) = q4::quantize_to_q8(&x_matrix[..hidden]);
            bench.run("CPU 12 individual q4_matvec calls", bytes, || {
                for s in 0..6 {
                    let (q8, sc) = q4::quantize_to_q8(&x_matrix[s * hidden..(s + 1) * hidden]);
                    let _ = cpu.q4_matvec(&gate_q4, &q8, &sc, inter, hidden);
                    let _ = cpu.q4_matvec(&up_q4, &q8, &sc, inter, hidden);
                }
            });
        }
    }

    // ── 6. Sequential multi-layer simulation ──
    println!("\n--- 6. Multi-layer simulation (21 layers, f32 FFN) ---\n");
    {
        // Simulate 21 layers of gate+up+down with different weight matrices
        let mut layers: Vec<(Array2<f32>, Array2<f32>, Array2<f32>)> = Vec::new();
        for l in 0..21 {
            layers.push((
                synth(inter, hidden, 100 + l as u64),
                synth(inter, hidden, 200 + l as u64),
                synth(inter, hidden, 300 + l as u64),
            ));
        }
        let x = synth(6, hidden, 42);
        let bytes = 3 * inter * hidden * 4 * 21;

        bench.run("CPU 21 layers × 3 matmuls", bytes, || {
            let mut h = x.clone();
            for (gate, up, down) in &layers {
                let g = cpu.matmul_transb(h.view(), gate.view());
                let u = cpu.matmul_transb(h.view(), up.view());
                // Simplified GEGLU
                let act = &g * &u;
                h = cpu.matmul(act.view(), down.view());
            }
        });

        if default.name() != cpu.name() {
            bench.run(&format!("{} 21 layers × 3 matmuls", default.name()), bytes, || {
                let mut h = x.clone();
                for (gate, up, down) in &layers {
                    let g = default.matmul_transb(h.view(), gate.view());
                    let u = default.matmul_transb(h.view(), up.view());
                    let act = &g * &u;
                    h = default.matmul(act.view(), down.view());
                }
            });
        }
    }

    // ── 7. Q4×f32 transposed down matvec ──
    println!("\n--- 7. Q4×f32 transposed down matvec ---\n");
    #[cfg(feature = "metal")]
    {
        if let Some(ref metal) = larql_compute::metal::MetalBackend::new() {
            let down_f32: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.0001).cos()).collect();
            // Transpose [inter, hidden] → [hidden, inter]
            let mut down_t: Vec<f32> = vec![0.0; hidden * inter];
            for r in 0..inter { for c in 0..hidden { down_t[c * inter + r] = down_f32[r * hidden + c]; } }
            let down_t_q4 = quantize_q4_0(&down_t);
            let activation: Vec<f32> = (0..inter).map(|i| if i % 5 == 0 { (i as f32 * 0.01).sin() } else { 0.0 }).collect();
            let bytes = down_t_q4.len();

            bench.run("Metal Q4×f32 matvec (transposed down)", bytes, || {
                let _ = metal.q4_f32_matvec_direct(&down_t_q4, &activation, hidden, inter);
            });

            // Compare with original vecmat
            let down_q4 = quantize_q4_0(&down_f32);
            bench.run("Metal Q4 vecmat (original down)", down_q4.len(), || {
                let _ = metal.q4_vecmat_direct(&activation, &down_q4, inter, hidden);
            });
        }
    }
    #[cfg(not(feature = "metal"))]
    println!("  (Metal not enabled)");

    // ── 8. Fused FFN (gate+up+GEGLU+down, one dispatch) ──
    println!("\n--- 8. Fused FFN (one Metal dispatch per position) ---\n");
    #[cfg(feature = "metal")]
    {
        if let Some(ref metal) = larql_compute::metal::MetalBackend::new() {
            let gate_f32: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.0001).cos()).collect();
            let up_f32: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.0002).sin()).collect();
            let down_f32: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.0003).cos()).collect();
            let mut down_t: Vec<f32> = vec![0.0; hidden * inter];
            for r in 0..inter { for c in 0..hidden { down_t[c * inter + r] = down_f32[r * hidden + c]; } }
            let gate_q4 = quantize_q4_0(&gate_f32);
            let up_q4 = quantize_q4_0(&up_f32);
            let down_t_q4 = quantize_q4_0(&down_t);
            let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();
            let bytes = gate_q4.len() + up_q4.len() + down_t_q4.len();

            // 3 separate dispatches (gate + up + down)
            let (q8_x, q8_s) = q4::quantize_to_q8(&x);
            bench.run("Metal 3-dispatch (pair + down)", bytes, || {
                let g = metal.q4_matvec_direct(&gate_q4, &q8_x, &q8_s, inter, hidden);
                let u = metal.q4_matvec_direct(&up_q4, &q8_x, &q8_s, inter, hidden);
                let mut act = vec![0.0f32; inter];
                for i in 0..inter { act[i] = (g[i] / (1.0 + (-g[i]).exp())) * u[i]; }
                let _ = metal.q4_f32_matvec_direct(&down_t_q4, &act, hidden, inter);
            });
        }
    }
    #[cfg(not(feature = "metal"))]
    println!("  (Metal not enabled)");

    // ── 9. Token generation (seq=1) ──
    println!("\n--- 9. Token generation (seq=1, per-layer) ---\n");
    {
        let matrix: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.0001).cos()).collect();
        let q4_data = quantize_q4_0(&matrix);
        let x1: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();
        let (q8_x1, q8_s1) = q4::quantize_to_q8(&x1);

        bench.run("CPU C kernel Q4 matvec (seq=1)", q4_data.len(), || {
            let _ = cpu.q4_matvec(&q4_data, &q8_x1, &q8_s1, inter, hidden);
        });
        bench.run("CPU BLAS f32 gemv (seq=1)", inter * hidden * 4, || {
            let mat = ndarray::ArrayView2::from_shape((inter, hidden), &matrix).unwrap();
            let xv = ndarray::ArrayView1::from(&x1);
            let _ = mat.dot(&xv);
        });
    }

    println!("\n--- 10. Correctness (CPU vs Default) ---\n");
    {
        let a = synth(6, hidden, 42);
        let b = synth(inter, hidden, 43);

        let cpu_result = cpu.matmul_transb(a.view(), b.view());
        let default_result = default.matmul_transb(a.view(), b.view());
        let diff: f32 = cpu_result.iter().zip(default_result.iter())
            .map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        println!("  f32 matmul_transb max diff: {diff:.2e} {}", if diff < 1e-4 { "✓" } else { "✗" });

        if cpu.has_q4() && default.has_q4() {
            let matrix: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.0001).cos()).collect();
            let q4_data = quantize_q4_0(&matrix);
            let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();
            let (q8_x, q8_scales) = q4::quantize_to_q8(&x);

            let cpu_q4 = cpu.q4_matvec(&q4_data, &q8_x, &q8_scales, inter, hidden).unwrap();
            let def_q4 = default.q4_matvec(&q4_data, &q8_x, &q8_scales, inter, hidden).unwrap();
            let diff: f32 = cpu_q4.iter().zip(def_q4.iter())
                .map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
            println!("  Q4 matvec max diff: {diff:.2e} {}", if diff < 1e-3 { "✓" } else { "✗" });
        }
    }

    println!("\n=== Done ===");
}
