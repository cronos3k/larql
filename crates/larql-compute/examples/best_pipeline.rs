//! Full pipeline benchmark: 21 layers × (attention + FFN) in one Metal submission.
//!
//! Usage:
//!   cargo run --release -p larql-compute --features metal --example bench_full_pipeline

#[cfg(unix)]
extern crate blas_src;

#[allow(unused_imports)]
use std::time::Instant;
#[allow(unused_imports)]
use larql_compute::cpu::q4::quantize_q4_0;

fn main() {
    #[cfg(not(feature = "metal"))]
    { println!("Run with --features metal"); return; }

    #[cfg(feature = "metal")]
    {
        use larql_compute::metal::MetalBackend;
        use larql_compute::metal::ops::full_pipeline::LayerWeights;

        let metal = MetalBackend::new().expect("Metal required");

        let hidden = 2560;
        let inter = 10240;
        let q_dim = 2560;
        let kv_dim = 512;
        let num_layers = 21;
        let n = 10;

        println!("=== Full Pipeline Benchmark (ALL Q4) ===");
        println!("{num_layers} layers × (4 Q4 attn proj + 3 Q4 FFN ops), one Metal submission\n");

        // Build ALL Q4 layer weights
        struct LayerData {
            wq_q4: Vec<u8>, wk_q4: Vec<u8>, wv_q4: Vec<u8>, wo_q4: Vec<u8>,
            gate_q4: Vec<u8>, up_q4: Vec<u8>, down_t_q4: Vec<u8>,
        }
        let mut layers_data: Vec<LayerData> = Vec::new();
        for l in 0..num_layers {
            let wq: Vec<f32> = (0..q_dim * hidden).map(|i| ((i + l * 1000) as f32 * 0.0001).cos()).collect();
            let wk: Vec<f32> = (0..kv_dim * hidden).map(|i| ((i + l * 2000) as f32 * 0.0002).sin()).collect();
            let wv: Vec<f32> = (0..kv_dim * hidden).map(|i| ((i + l * 3000) as f32 * 0.0003).cos()).collect();
            let wo: Vec<f32> = (0..hidden * q_dim).map(|i| ((i + l * 4000) as f32 * 0.0004).sin()).collect();
            let g: Vec<f32> = (0..inter * hidden).map(|i| ((i + l * 5000) as f32 * 0.0001).cos()).collect();
            let u: Vec<f32> = (0..inter * hidden).map(|i| ((i + l * 6000) as f32 * 0.0002).sin()).collect();
            let mut dt = vec![0.0f32; hidden * inter];
            for r in 0..inter { for c in 0..hidden { dt[c * inter + r] = ((r * hidden + c + l * 7000) as f32 * 0.0003).cos(); } }
            layers_data.push(LayerData {
                wq_q4: quantize_q4_0(&wq), wk_q4: quantize_q4_0(&wk),
                wv_q4: quantize_q4_0(&wv), wo_q4: quantize_q4_0(&wo),
                gate_q4: quantize_q4_0(&g), up_q4: quantize_q4_0(&u),
                down_t_q4: quantize_q4_0(&dt),
            });
        }

        let layers: Vec<LayerWeights> = layers_data.iter().map(|ld| {
            LayerWeights {
                wq_q4: &ld.wq_q4, wk_q4: &ld.wk_q4, wv_q4: &ld.wv_q4, wo_q4: &ld.wo_q4,
                gate_q4: &ld.gate_q4, up_q4: &ld.up_q4, down_t_q4: &ld.down_t_q4,
            }
        }).collect();

        let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();

        // Warmup
        let _ = metal.full_pipeline(&layers, &x, hidden, inter, q_dim, kv_dim);

        // Benchmark
        let t0 = Instant::now();
        for _ in 0..n {
            let _ = metal.full_pipeline(&layers, &x, hidden, inter, q_dim, kv_dim);
        }
        let full_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let tps = 1000.0 / full_ms;

        // FFN-only for comparison
        let layers_q4_refs: Vec<(&[u8], &[u8], &[u8])> = layers_data.iter()
            .map(|ld| (ld.gate_q4.as_slice(), ld.up_q4.as_slice(), ld.down_t_q4.as_slice())).collect();
        let _ = metal.multi_layer_q4_ffn(&layers_q4_refs, &x, inter, hidden);
        let t0 = Instant::now();
        for _ in 0..n {
            let _ = metal.multi_layer_q4_ffn(&layers_q4_refs, &x, inter, hidden);
        }
        let ffn_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;

        // Measure CPU BLAS attn for comparison
        let cpu_attn_ms = {
            let x_arr = ndarray::Array2::from_shape_vec((1, hidden), x.clone()).unwrap();
            let wq_f32: Vec<f32> = (0..q_dim * hidden).map(|i| (i as f32 * 0.0001).cos()).collect();
            let wq_arr = ndarray::Array2::from_shape_vec((q_dim, hidden), wq_f32).unwrap();
            // Warmup
            let _ = x_arr.dot(&wq_arr.t());
            let t0 = Instant::now();
            for _ in 0..n {
                for _ in 0..num_layers {
                    let _ = x_arr.dot(&wq_arr.t()); // Q
                    let _ = x_arr.dot(&wq_arr.t()); // K (approx)
                    let _ = x_arr.dot(&wq_arr.t()); // V (approx)
                    let _ = x_arr.dot(&wq_arr.t()); // O
                }
            }
            t0.elapsed().as_secs_f64() * 1000.0 / n as f64
        };

        println!("  Metal full pipeline (attn+FFN, 1 cmd):  {full_ms:>6.1}ms  ({tps:.0} tok/s)");
        println!("  Metal FFN-only (1 cmd):                 {ffn_ms:>6.1}ms");
        println!("  CPU BLAS attn-only (4 proj × {num_layers}L):    {cpu_attn_ms:>6.1}ms");
        println!("  Attention overhead in pipeline:          {:.1}ms", full_ms - ffn_ms);
        println!();
        println!("  Projected with vindex logits + cache:");
        let projected = full_ms + 5.0; // + logits + other
        println!("    {projected:.0}ms → {:.0} tok/s", 1000.0 / projected);
        println!();
        println!("  Ollama reference: ~10ms → ~100 tok/s");

        println!("\n=== Done ===");
    }
}
