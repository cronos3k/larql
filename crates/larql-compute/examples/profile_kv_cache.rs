//! KV cache + attention benchmark.
//!
//! Simulates token generation: append K/V, attend against cache.
//! Measures: per-token attention time with growing cache.
//!
//! Usage:
//!   cargo run --release -p larql-compute --features metal --example bench_kv_cache

#[cfg(unix)]
extern crate blas_src;

#[allow(unused_imports)]
use std::time::Instant;

fn main() {
    #[cfg(not(feature = "metal"))]
    { println!("Run with --features metal"); return; }

    #[cfg(feature = "metal")]
    {
        use larql_compute::metal::MetalBackend;
        use larql_compute::metal::ops::kv_cache::{KVCache, append_and_attend};

        let metal = MetalBackend::new().expect("Metal required");
        let bufs = metal.bufs();

        let num_q_heads = 8;
        let num_kv_heads = 4;
        let head_dim = 320;   // Gemma: 2560 / 8 = 320 (approx)
        let max_seq = 512;
        let num_layers = 21;
        let n = 20;

        println!("=== KV Cache Attention Benchmark ===");
        println!("{num_layers} layers, {num_q_heads} Q heads, {num_kv_heads} KV heads, dim={head_dim}");
        println!("Max cache: {max_seq} tokens\n");

        let mut cache = KVCache::new(bufs, num_layers, max_seq, num_kv_heads, head_dim);
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Simulate generation: append tokens and measure attention time
        println!("  {:<10} {:>10} {:>10}", "Cache len", "Per-token", "tok/s (attn)");

        for &gen_tokens in &[1, 5, 10, 20, 50, 100] {
            cache.clear();

            // Fill cache to gen_tokens
            for t in 0..gen_tokens {
                let q_data: Vec<f32> = (0..num_q_heads * head_dim).map(|i| ((i + t * 100) as f32 * 0.001).sin()).collect();
                let k_data: Vec<f32> = (0..num_kv_heads * head_dim).map(|i| ((i + t * 200) as f32 * 0.002).cos()).collect();
                let v_data: Vec<f32> = (0..num_kv_heads * head_dim).map(|i| ((i + t * 300) as f32 * 0.003).sin()).collect();

                let buf_q = bufs.transient_from_f32(&q_data);
                let buf_k = bufs.transient_from_f32(&k_data);
                let buf_v = bufs.transient_from_f32(&v_data);
                let buf_out = bufs.output((num_q_heads * head_dim * 4) as u64);

                let cmd = metal.queue().new_command_buffer();
                for l in 0..num_layers {
                    append_and_attend(
                        cmd, &mut cache.layers[l],
                        &metal.kv_append_pipeline, &metal.kv_attend_pipeline,
                        &buf_k, &buf_v, &buf_q, &buf_out,
                        num_q_heads, scale,
                    );
                }
                cmd.commit();
                cmd.wait_until_completed();
            }

            // Now benchmark one more token with full cache
            let q_data: Vec<f32> = (0..num_q_heads * head_dim).map(|i| (i as f32 * 0.001).sin()).collect();
            let k_data: Vec<f32> = (0..num_kv_heads * head_dim).map(|i| (i as f32 * 0.002).cos()).collect();
            let v_data: Vec<f32> = (0..num_kv_heads * head_dim).map(|i| (i as f32 * 0.003).sin()).collect();

            let buf_q = bufs.transient_from_f32(&q_data);
            let buf_k = bufs.transient_from_f32(&k_data);
            let buf_v = bufs.transient_from_f32(&v_data);
            let buf_out = bufs.output((num_q_heads * head_dim * 4) as u64);

            // Reset cache position to gen_tokens (don't double-count)
            for l in 0..num_layers { cache.layers[l].current_len = gen_tokens; }

            // Warmup
            {
                for l in 0..num_layers { cache.layers[l].current_len = gen_tokens; }
                let cmd = metal.queue().new_command_buffer();
                for l in 0..num_layers {
                    append_and_attend(
                        cmd, &mut cache.layers[l],
                        &metal.kv_append_pipeline, &metal.kv_attend_pipeline,
                        &buf_k, &buf_v, &buf_q, &buf_out,
                        num_q_heads, scale,
                    );
                }
                cmd.commit();
                cmd.wait_until_completed();
            }

            // Benchmark
            let t0 = Instant::now();
            for _ in 0..n {
                for l in 0..num_layers { cache.layers[l].current_len = gen_tokens; }
                let cmd = metal.queue().new_command_buffer();
                for l in 0..num_layers {
                    append_and_attend(
                        cmd, &mut cache.layers[l],
                        &metal.kv_append_pipeline, &metal.kv_attend_pipeline,
                        &buf_k, &buf_v, &buf_q, &buf_out,
                        num_q_heads, scale,
                    );
                }
                cmd.commit();
                cmd.wait_until_completed();
            }
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
            let tps = 1000.0 / ms;

            println!("  T={gen_tokens:<8} {ms:>9.2}ms  {tps:>8.0}");
        }

        println!("\n  (These times are attention ONLY — add FFN for full decode)");
        println!("  FFN pipeline: ~8.5ms");
        println!("  Total decode projection: attn + 8.5ms FFN + 5ms other");

        println!("\n=== Done ===");
    }
}
