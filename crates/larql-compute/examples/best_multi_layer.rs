//! Pipeline benchmarks: multi-layer Q4, mixed backend, batch sweep.
//!
//! Tests the actual production scenarios that matter for closing
//! the gap with Ollama.
//!
//! Usage:
//!   cargo run --release -p larql-compute --features metal --example bench_pipeline

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

struct Timer { n: usize }
impl Timer {
    fn run<F: FnMut()>(&self, name: &str, mut f: F) -> f64 {
        f(); // warmup
        let t0 = Instant::now();
        for _ in 0..self.n { f(); }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / self.n as f64;
        println!("  {name:50} {ms:>7.2}ms");
        ms
    }
}

fn main() {
    let hidden = 2560;
    let inter = 10240;
    let cpu = cpu_backend();
    let default = default_backend();
    let t = Timer { n: 5 };

    println!("=== Pipeline Benchmarks ===");
    println!("CPU: {}", cpu.name());
    println!("Default: {}\n", default.name());

    // Build 21 layers of Q4 data (gate + up + down_T)
    println!("Building 21 layers of Q4 data...");
    let mut layers_q4: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = Vec::new();
    let mut layers_f32: Vec<(Array2<f32>, Array2<f32>, Array2<f32>)> = Vec::new();
    for l in 0..21u64 {
        let g: Vec<f32> = (0..inter * hidden).map(|i| ((i as f64 + l as f64 * 1e7) * 0.0001).cos() as f32).collect();
        let u: Vec<f32> = (0..inter * hidden).map(|i| ((i as f64 + l as f64 * 2e7) * 0.0002).sin() as f32).collect();
        let d: Vec<f32> = (0..inter * hidden).map(|i| ((i as f64 + l as f64 * 3e7) * 0.0003).cos() as f32).collect();
        // Transpose down for matvec pattern
        let mut dt = vec![0.0f32; hidden * inter];
        for r in 0..inter { for c in 0..hidden { dt[c * inter + r] = d[r * hidden + c]; } }
        layers_q4.push((quantize_q4_0(&g), quantize_q4_0(&u), quantize_q4_0(&dt)));
        layers_f32.push((
            Array2::from_shape_vec((inter, hidden), g).unwrap(),
            Array2::from_shape_vec((inter, hidden), u).unwrap(),
            Array2::from_shape_vec((inter, hidden), d).unwrap(),
        ));
    }
    println!("Done.\n");

    // ── 1. 21-layer Q4 3-dispatch (Metal) ──
    println!("--- 1. 21-layer Q4 FFN (Metal 3-dispatch per layer) ---\n");
    #[cfg(feature = "metal")]
    {
        if let Some(ref metal) = larql_compute::metal::MetalBackend::new() {
            t.run("Metal Q4 21-layer FFN (3-dispatch/layer)", || {
                let mut h: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();
                for (gate_q4, up_q4, down_t_q4) in &layers_q4 {
                    let (q8, sc) = q4::quantize_to_q8(&h);
                    let g = metal.q4_matvec_direct(gate_q4, &q8, &sc, inter, hidden);
                    let u = metal.q4_matvec_direct(up_q4, &q8, &sc, inter, hidden);
                    let mut act = vec![0.0f32; inter];
                    for i in 0..inter { act[i] = (g[i] / (1.0 + (-g[i]).exp())) * u[i]; }
                    h = metal.q4_f32_matvec_direct(down_t_q4, &act, hidden, inter);
                }
            });
        }
    }

    // ── 2. 21-layer f32 FFN (CPU BLAS) ──
    println!("\n--- 2. 21-layer f32 FFN (CPU BLAS) ---\n");
    {
        t.run("CPU BLAS f32 21-layer FFN", || {
            let mut h = synth(6, hidden, 42);
            for (gate, up, down) in &layers_f32 {
                let g = cpu.matmul_transb(h.view(), gate.view());
                let u = cpu.matmul_transb(h.view(), up.view());
                let act = &g * &u; // simplified GEGLU
                h = cpu.matmul(act.view(), down.view());
            }
        });
    }

    // ── 3. 21-layer Q4 (CPU C kernel) ──
    println!("\n--- 3. 21-layer Q4 FFN (CPU C kernel) ---\n");
    {
        t.run("CPU C kernel Q4 21-layer FFN", || {
            let mut h: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();
            for (gate_q4, up_q4, down_t_q4) in &layers_q4 {
                let g = q4::q4_matvec(gate_q4, &h, inter, hidden);
                let u = q4::q4_matvec(up_q4, &h, inter, hidden);
                let mut act = vec![0.0f32; inter];
                for i in 0..inter { act[i] = (g[i] / (1.0 + (-g[i]).exp())) * u[i]; }
                // For down: use CPU vecmat (original layout would be q4_vecmat,
                // but we have transposed, so use matvec with hidden as num_rows)
                h = q4::q4_matvec(down_t_q4, &act, hidden, inter);
            }
        });
    }

    // ── 4. Mixed: CPU f32 attention + Metal Q4 FFN (per layer) ──
    println!("\n--- 4. Mixed: CPU attn + Metal Q4 FFN (per layer) ---\n");
    #[cfg(feature = "metal")]
    {
        if let Some(ref metal) = larql_compute::metal::MetalBackend::new() {
            // Simulate attention as 4 f32 matmul_transb (Q, K, V, O projections)
            let attn_weights: Vec<Array2<f32>> = (0..21).map(|l| synth(2560, 2560, 1000 + l)).collect();

            t.run("Mixed: CPU attn (f32) + Metal FFN (Q4) × 21", || {
                let h = synth(6, hidden, 42);
                for l in 0..21 {
                    // Attention (CPU f32): 4 projections
                    let _ = cpu.matmul_transb(h.view(), attn_weights[l].view());
                    let _ = cpu.matmul_transb(h.view(), attn_weights[l].view());
                    let _ = cpu.matmul_transb(h.view(), attn_weights[l].view());
                    let _ = cpu.matmul_transb(h.view(), attn_weights[l].view());

                    // FFN (Metal Q4): gate + up + down
                    let h_row = h.row(0).to_vec(); // use first position
                    let (gate_q4, up_q4, down_t_q4) = &layers_q4[l];
                    let (q8, sc) = q4::quantize_to_q8(&h_row);
                    let g = metal.q4_matvec_direct(gate_q4, &q8, &sc, inter, hidden);
                    let u = metal.q4_matvec_direct(up_q4, &q8, &sc, inter, hidden);
                    let mut act = vec![0.0f32; inter];
                    for i in 0..inter { act[i] = (g[i] / (1.0 + (-g[i]).exp())) * u[i]; }
                    let _ = metal.q4_f32_matvec_direct(down_t_q4, &act, hidden, inter);
                }
            });
        }
    }

    // ── 5. Multi-layer Q4 FFN: one command buffer for ALL 21 layers ──
    println!("\n--- 5. Multi-layer Q4 (1 command buffer, ALL 21 layers) ---\n");
    #[cfg(feature = "metal")]
    {
        if let Some(ref metal) = larql_compute::metal::MetalBackend::new() {
            let layers_refs: Vec<(&[u8], &[u8], &[u8])> = layers_q4.iter().map(|(g, u, d)| (g.as_slice(), u.as_slice(), d.as_slice())).collect();
            let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();

            t.run("Metal multi-layer Q4 (21L, 1 cmd buffer, all GPU)", || {
                let _ = metal.multi_layer_q4_ffn(&layers_refs, &x, inter, hidden);
            });
        }
    }
    #[cfg(not(feature = "metal"))]
    println!("  (Metal not enabled)");

    // ── 6. Full layer on Metal (old per-layer benchmark) (attention + FFN, one command buffer) ──
    println!("\n--- 5. Full layer on Metal (attn + FFN, 1 cmd buffer) ---\n");
    #[cfg(feature = "metal")]
    {
        if let Some(ref metal) = larql_compute::metal::MetalBackend::new() {
            let w_q: Vec<f32> = (0..hidden * hidden).map(|i| (i as f32 * 0.0001).cos()).collect();
            let w_k: Vec<f32> = (0..512 * hidden).map(|i| (i as f32 * 0.0002).sin()).collect();
            let w_v: Vec<f32> = (0..512 * hidden).map(|i| (i as f32 * 0.0003).cos()).collect();
            let w_o: Vec<f32> = (0..hidden * hidden).map(|i| (i as f32 * 0.0004).sin()).collect();
            let x: Vec<f32> = (0..6 * hidden).map(|i| (i as f32 * 0.001).sin()).collect();

            let (gate_q4, up_q4, down_t_q4) = &layers_q4[0];

            t.run("Metal full layer (attn+FFN, 1 cmd buffer)", || {
                let _ = metal.full_layer_direct(
                    &w_q, &w_k, &w_v, &w_o,
                    gate_q4, up_q4, down_t_q4,
                    &x, 6, hidden, 8, 4, 320, inter, 1.0 / (320.0f32).sqrt(),
                );
            });

            // Compare: CPU attention + Metal FFN (separate)
            let wq_arr = Array2::from_shape_vec((hidden, hidden), w_q.clone()).unwrap();
            t.run("CPU attn + Metal FFN (separate dispatches)", || {
                // 4 attention projections on CPU
                let h = synth(6, hidden, 42);
                let _ = cpu.matmul_transb(h.view(), wq_arr.view());
                let _ = cpu.matmul_transb(h.view(), wq_arr.view());
                let _ = cpu.matmul_transb(h.view(), wq_arr.view());
                let _ = cpu.matmul_transb(h.view(), wq_arr.view());
                // FFN on Metal
                let h_row = h.row(0).to_vec();
                let (q8, sc) = q4::quantize_to_q8(&h_row);
                let g = metal.q4_matvec_direct(gate_q4, &q8, &sc, inter, hidden);
                let u = metal.q4_matvec_direct(up_q4, &q8, &sc, inter, hidden);
                let mut act = vec![0.0f32; inter];
                for i in 0..inter { act[i] = (g[i] / (1.0 + (-g[i]).exp())) * u[i]; }
                let _ = metal.q4_f32_matvec_direct(down_t_q4, &act, hidden, inter);
            });
        }
    }
    #[cfg(not(feature = "metal"))]
    println!("  (Metal not enabled)");

    // ── 6. Batch size sweep (Q4 matvec) ──
    println!("\n--- 6. Batch size sweep (Q4 matvec, one matrix) ---\n");
    {
        let matrix: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.0001).cos()).collect();
        let q4_data = quantize_q4_0(&matrix);

        for &seq in &[1, 6, 16, 32] {
            let x: Vec<f32> = (0..seq * hidden).map(|i| (i as f32 * 0.001).sin()).collect();
            let label = format!("CPU Q4 matvec seq={seq} ({seq} calls)");
            t.run(&label, || {
                for s in 0..seq {
                    let slice = &x[s * hidden..(s + 1) * hidden];
                    let _ = q4::q4_matvec(&q4_data, slice, inter, hidden);
                }
            });
        }
    }

    println!("\n=== Done ===");
}
