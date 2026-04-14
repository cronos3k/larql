//! Micro-benchmark: single-layer Q4_K QKV + FFN to isolate per-layer cost.

#[cfg(unix)]
extern crate blas_src;

fn main() {
    #[cfg(not(feature = "metal"))]
    { println!("Run with --features metal"); return; }

    #[cfg(feature = "metal")]
    {
        use std::time::Instant;
        use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_q4_0};

        let metal = larql_compute::default_backend();
        let n = 50;

        let hidden = 2560usize;
        let inter = 10240usize;
        let num_q = 8usize; let num_kv = 4usize; let hd = 320usize;
        let q_dim = num_q * hd; let kv_dim = num_kv * hd;

        fn pad(d: &[f32]) -> Vec<f32> { let p = d.len().div_ceil(256)*256; let mut o = d.to_vec(); o.resize(p, 0.0); o }

        println!("=== Per-Layer Kernel Micro-Benchmark ===\n");

        // Build 1-layer and 21-layer configs
        for &num_layers in &[1usize, 21] {
            let mut layers_data = Vec::new();
            for l in 0..num_layers {
                let wq = quantize_q4_k(&pad(&(0..q_dim*hidden).map(|i| ((i+l*1000) as f32*0.0001).cos()).collect::<Vec<_>>()));
                let wk = quantize_q4_k(&pad(&(0..kv_dim*hidden).map(|i| ((i+l*2000) as f32*0.0002).sin()).collect::<Vec<_>>()));
                let wv = quantize_q4_k(&pad(&(0..kv_dim*hidden).map(|i| ((i+l*3000) as f32*0.0003).cos()).collect::<Vec<_>>()));
                let wo = quantize_q4_k(&pad(&(0..hidden*q_dim).map(|i| ((i+l*4000) as f32*0.0004).sin()).collect::<Vec<_>>()));
                let g = quantize_q4_0(&(0..inter*hidden).map(|i| ((i+l*5000) as f32*0.0001).cos()).collect::<Vec<_>>());
                let u = quantize_q4_0(&(0..inter*hidden).map(|i| ((i+l*6000) as f32*0.0002).sin()).collect::<Vec<_>>());
                let d = quantize_q4_0(&(0..hidden*inter).map(|i| ((i+l*7000) as f32*0.0003).cos()).collect::<Vec<_>>());
                layers_data.push((wq,wk,wv,wo,g,u,d,vec![1.0f32;hidden]));
            }

            let layers: Vec<larql_compute::FullPipelineLayer> = layers_data.iter().map(|(wq,wk,wv,wo,g,u,d,norm)| {
                larql_compute::FullPipelineLayer {
                    wq: larql_compute::QuantWeight { data: wq, scales: None, format: larql_compute::QuantFormat::Q4_K },
                    wk: larql_compute::QuantWeight { data: wk, scales: None, format: larql_compute::QuantFormat::Q4_K },
                    wv: larql_compute::QuantWeight { data: wv, scales: None, format: larql_compute::QuantFormat::Q4_K },
                    wo: larql_compute::QuantWeight { data: wo, scales: None, format: larql_compute::QuantFormat::Q4_K },
                    gate: larql_compute::QuantWeight { data: g, scales: None, format: larql_compute::QuantFormat::Q4_0 },
                    up: larql_compute::QuantWeight { data: u, scales: None, format: larql_compute::QuantFormat::Q4_0 },
                    down: larql_compute::QuantWeight { data: d, scales: None, format: larql_compute::QuantFormat::Q4_0 },
                    input_norm: norm, post_attn_norm: norm,
                    pre_ffn_norm: None, post_ffn_norm: None,
                    norm_offset: 1.0, has_post_norms: false,
                    activation: larql_compute::Activation::Silu,
                    qk_norm_offset: 0.0,
                    eps: 1e-6,
                    norm_type: larql_compute::NormType::RmsNorm,
                    ffn_type: larql_compute::FfnType::Gated,
                    attn_scale: 1.0 / (hd as f32).sqrt(),
                    head_dim: hd,
                    num_q_heads: num_q,
                    num_kv_heads: num_kv,
                    rope_base: 10000.0,
                    rotary_dim: 0,
                    sliding_window: 0,
                    has_v_norm: false,
                    layer_scalar: 0.0,
                    input_norm_bias: None,
                    post_attn_norm_bias: None,
                    ffn_up_bias: None,
                    ffn_down_bias: None,
                }
            }).collect();

            let x: Vec<f32> = (0..hidden).map(|i| (i as f32*0.001).sin()).collect();

            // Warmup
            for _ in 0..3 {
                let _ = metal.full_pipeline_q4(&layers, &x, hidden, inter, q_dim, kv_dim,
                    1, num_q, num_kv, hd, 10000.0, false, 0.0);
            }

            let t0 = Instant::now();
            for _ in 0..n {
                let _ = metal.full_pipeline_q4(&layers, &x, hidden, inter, q_dim, kv_dim,
                    1, num_q, num_kv, hd, 10000.0, false, 0.0);
            }
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
            let per_layer = ms / num_layers as f64;
            let data_mb = layers_data.iter().map(|(q,k,v,o,g,u,d,_)| q.len()+k.len()+v.len()+o.len()+g.len()+u.len()+d.len()).sum::<usize>() as f64 / 1e6 / num_layers as f64;

            println!("  {num_layers:>2} layers: {ms:>7.2}ms total, {per_layer:.3}ms/layer  ({data_mb:.1}MB/layer)");
        }

        // Ollama comparison
        println!("\n  Ollama: 9.7ms / 26 layers = 0.373ms/layer (entire layer)");
        println!("\n=== Done ===");
    }
}
