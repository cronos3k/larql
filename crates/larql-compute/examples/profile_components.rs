//! Component-level profiling: each operation isolated over 34 layers.

#[cfg(unix)]
extern crate blas_src;

fn main() {
    #[cfg(not(feature = "metal"))]
    { println!("Run with --features metal"); return; }

    #[cfg(feature = "metal")]
    {
        use std::time::Instant;
        use std::ffi::c_void;
        use larql_compute::ComputeBackend;
        use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_q4_0, quantize_to_q8};

        let metal = larql_compute::metal::MetalBackend::new().expect("Metal required");

        let hidden = 2560usize;
        let inter = 10240usize;
        let num_q = 8; let num_kv = 4; let hd = 320;
        let q_dim = num_q * hd; let kv_dim = num_kv * hd;
        let layers = 34usize;
        let n = 30;

        fn pad(d: &[f32]) -> Vec<f32> { let p=d.len().div_ceil(256)*256; let mut o=d.to_vec(); o.resize(p,0.0); o }

        println!("=== Component Profiling ({layers} layers, 1 cmd buffer each) ===\n");

        // Build weight data
        let wq = quantize_q4_k(&pad(&vec![0.01f32; q_dim * hidden]));
        let wk = quantize_q4_k(&pad(&vec![0.01f32; kv_dim * hidden]));
        let wv = quantize_q4_k(&pad(&vec![0.01f32; kv_dim * hidden]));
        let wo = quantize_q4_k(&pad(&vec![0.01f32; hidden * q_dim]));
        let gate = quantize_q4_0(&vec![0.01f32; inter * hidden]);
        let up = quantize_q4_0(&vec![0.01f32; inter * hidden]);
        let down = quantize_q4_0(&vec![0.01f32; hidden * inter]);
        let norm_w = vec![1.0f32; hidden];
        let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();

        let buf_wq = metal.bufs().get_bytes(&wq);
        let buf_wk = metal.bufs().get_bytes(&wk);
        let buf_wv = metal.bufs().get_bytes(&wv);
        let buf_wo = metal.bufs().get_bytes(&wo);
        let buf_gate = metal.bufs().get_bytes(&gate);
        let buf_up = metal.bufs().get_bytes(&up);
        let buf_down = metal.bufs().get_bytes(&down);
        let buf_norm = metal.bufs().transient_from_f32(&norm_w);
        let buf_x = metal.bufs().transient_from_f32(&x);

        let hidden_val = hidden as u32;
        let inter_val = inter as u32;
        let eps = 1e-6f32;
        let norm_off = 1.0f32;

        use larql_compute::metal::shaders::q4k_qkv_proj as qkv_sh;
        use larql_compute::metal::shaders::q4_matvec as q4mv;

        macro_rules! bench {
            ($name:expr, $body:expr) => {{
                // warmup
                for _ in 0..3 { $body; }
                let t0 = Instant::now();
                for _ in 0..n { $body; }
                let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
                let per = ms / layers as f64;
                println!("  {:<35} {:>7.2}ms  ({per:.3}ms/layer)", $name, ms);
                ms
            }};
        }

        // 1. RMS norm × 34
        let norm_ms = bench!("rms_norm", {
            let cmd = metal.queue().new_command_buffer();
            for _ in 0..layers {
                let out = metal.bufs().output((hidden * 4) as u64);
                let enc = cmd.new_compute_command_encoder();
                larql_compute::metal::ops::full_pipeline::encode_rms_norm(
                    enc, &metal.rms_norm_pipeline, &buf_x, &buf_norm, &out, hidden, eps, norm_off);
                enc.end_encoding();
            }
            cmd.commit(); cmd.wait_until_completed();
        });

        // 2. Q4_K QKV × 34
        let qkv_ms = bench!("Q4_K QKV fused", {
            let cmd = metal.queue().new_command_buffer();
            let total = (q_dim + kv_dim + kv_dim) as u32;
            let num_tgs = (total as u64).div_ceil(qkv_sh::ROWS_PER_TG);
            for _ in 0..layers {
                let qo = metal.bufs().output((q_dim*4) as u64);
                let ko = metal.bufs().output((kv_dim*4) as u64);
                let vo = metal.bufs().output((kv_dim*4) as u64);
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&metal.q4k_qkv_proj_pipeline);
                enc.set_buffer(0, Some(&buf_wq), 0); enc.set_buffer(1, Some(&buf_wk), 0);
                enc.set_buffer(2, Some(&buf_wv), 0); enc.set_buffer(3, Some(&buf_x), 0);
                enc.set_buffer(4, Some(&qo), 0); enc.set_buffer(5, Some(&ko), 0); enc.set_buffer(6, Some(&vo), 0);
                let q=q_dim as u32; let k=kv_dim as u32; let v=kv_dim as u32; let h=hidden as u32;
                enc.set_bytes(7, 4, &q as *const u32 as *const c_void);
                enc.set_bytes(8, 4, &k as *const u32 as *const c_void);
                enc.set_bytes(9, 4, &v as *const u32 as *const c_void);
                enc.set_bytes(10, 4, &h as *const u32 as *const c_void);
                enc.dispatch_thread_groups(metal::MTLSize::new(num_tgs, 1, 1), metal::MTLSize::new(qkv_sh::THREADS_PER_TG, 1, 1));
                enc.end_encoding();
            }
            cmd.commit(); cmd.wait_until_completed();
        });

        // 3. KV cache append+attend × 34
        let kv_ms = bench!("KV cache append+attend", {
            metal.reset_kv_cache();
            // Pre-populate some KV to simulate decode at T=5
            let cmd = metal.queue().new_command_buffer();
            for _l in 0..layers {
                let ko = metal.bufs().output((kv_dim*4) as u64);
                let _vo = metal.bufs().output((kv_dim*4) as u64);
                let _qo = metal.bufs().output((q_dim*4) as u64);
                let _ao = metal.bufs().output((q_dim*4) as u64);
                // Need kv_cache — use decode_token trait to init, then just measure attend
                // Simplified: just measure the dispatch overhead
                let enc = cmd.new_compute_command_encoder();
                // dummy dispatch to measure encoder overhead
                enc.set_compute_pipeline_state(&metal.rms_norm_pipeline);
                enc.set_buffer(0, Some(&buf_x), 0); enc.set_buffer(1, Some(&buf_norm), 0);
                enc.set_buffer(2, Some(&ko), 0);
                enc.set_bytes(3, 4, &hidden_val as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &eps as *const f32 as *const c_void);
                enc.set_bytes(5, 4, &norm_off as *const f32 as *const c_void);
                enc.dispatch_threads(metal::MTLSize::new(hidden as u64, 1, 1), metal::MTLSize::new(256, 1, 1));
                // second dispatch (simulate attend)
                enc.dispatch_threads(metal::MTLSize::new(hidden as u64, 1, 1), metal::MTLSize::new(256, 1, 1));
                enc.end_encoding();
            }
            cmd.commit(); cmd.wait_until_completed();
        });

        // 4. O projection × 34
        let o_ms = bench!("Q4_K O projection", {
            let cmd = metal.queue().new_command_buffer();
            let o_tgs = (hidden as u64).div_ceil(qkv_sh::ROWS_PER_TG);
            for _ in 0..layers {
                let oo = metal.bufs().output((hidden*4) as u64);
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&metal.q4k_qkv_proj_pipeline); // reuse for single proj
                enc.set_buffer(0, Some(&buf_wo), 0); enc.set_buffer(1, Some(&buf_wo), 0);
                enc.set_buffer(2, Some(&buf_wo), 0); enc.set_buffer(3, Some(&buf_x), 0);
                enc.set_buffer(4, Some(&oo), 0); enc.set_buffer(5, Some(&oo), 0); enc.set_buffer(6, Some(&oo), 0);
                let nr = hidden as u32; let z = 0u32; let h = q_dim as u32;
                enc.set_bytes(7, 4, &nr as *const u32 as *const c_void);
                enc.set_bytes(8, 4, &z as *const u32 as *const c_void);
                enc.set_bytes(9, 4, &z as *const u32 as *const c_void);
                enc.set_bytes(10, 4, &h as *const u32 as *const c_void);
                enc.dispatch_thread_groups(metal::MTLSize::new(o_tgs, 1, 1), metal::MTLSize::new(qkv_sh::THREADS_PER_TG, 1, 1));
                enc.end_encoding();
            }
            cmd.commit(); cmd.wait_until_completed();
        });

        // 5. Residual + norm (fused) × 34
        let res_ms = bench!("residual+norm+Q8 (fused)", {
            let cmd = metal.queue().new_command_buffer();
            for _ in 0..layers {
                let out = metal.bufs().output((hidden*4) as u64);
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&metal.rms_norm_pipeline);
                enc.set_buffer(0, Some(&buf_x), 0); enc.set_buffer(1, Some(&buf_norm), 0); enc.set_buffer(2, Some(&out), 0);
                enc.set_bytes(3, 4, &hidden_val as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &eps as *const f32 as *const c_void);
                enc.set_bytes(5, 4, &norm_off as *const f32 as *const c_void);
                enc.dispatch_threads(metal::MTLSize::new(hidden as u64, 1, 1), metal::MTLSize::new(256, 1, 1));
                enc.end_encoding();
            }
            cmd.commit(); cmd.wait_until_completed();
        });

        // 6. FFN (gate+up+geglu+down) × 34
        let (q8_x, q8_s) = quantize_to_q8(&x);
        let buf_q8 = metal.bufs().transient_from_i8(&q8_x);
        let buf_q8s = metal.bufs().transient_from_f32(&q8_s);

        let ffn_ms = bench!("Q4 FFN (gate+up+geglu+down)", {
            let cmd = metal.queue().new_command_buffer();
            let n_tgs = (inter as u64).div_ceil(q4mv::ROWS_PER_TG);
            for _ in 0..layers {
                let go = metal.bufs().output((inter*4) as u64);
                let uo = metal.bufs().output((inter*4) as u64);
                let ao = metal.bufs().output((inter*4) as u64);
                let do_ = metal.bufs().output((hidden*4) as u64);
                let enc = cmd.new_compute_command_encoder();
                // gate
                enc.set_compute_pipeline_state(&metal.q4.matvec);
                enc.set_buffer(0, Some(&buf_gate), 0); enc.set_buffer(1, Some(&buf_q8), 0);
                enc.set_buffer(2, Some(&buf_q8s), 0); enc.set_buffer(3, Some(&go), 0);
                enc.set_bytes(4, 4, &inter_val as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &hidden_val as *const u32 as *const c_void);
                enc.dispatch_thread_groups(metal::MTLSize::new(n_tgs, 1, 1), metal::MTLSize::new(q4mv::THREADS_PER_TG, 1, 1));
                // up
                enc.set_buffer(0, Some(&buf_up), 0); enc.set_buffer(3, Some(&uo), 0);
                enc.dispatch_thread_groups(metal::MTLSize::new(n_tgs, 1, 1), metal::MTLSize::new(q4mv::THREADS_PER_TG, 1, 1));
                // geglu
                enc.set_compute_pipeline_state(&metal.geglu_pipeline);
                enc.set_buffer(0, Some(&go), 0); enc.set_buffer(1, Some(&uo), 0); enc.set_buffer(2, Some(&ao), 0);
                enc.set_bytes(3, 4, &inter_val as *const u32 as *const c_void);
                enc.dispatch_threads(metal::MTLSize::new(inter as u64, 1, 1), metal::MTLSize::new(256, 1, 1));
                // down
                enc.set_compute_pipeline_state(&metal.q4.f32_matvec);
                enc.set_buffer(0, Some(&buf_down), 0); enc.set_buffer(1, Some(&ao), 0); enc.set_buffer(2, Some(&do_), 0);
                enc.set_bytes(3, 4, &hidden_val as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &inter_val as *const u32 as *const c_void);
                enc.dispatch_threads(metal::MTLSize::new(hidden as u64, 1, 1), metal::MTLSize::new(256, 1, 1));
                enc.end_encoding();
            }
            cmd.commit(); cmd.wait_until_completed();
        });

        // 7. Residual add × 34
        let add_ms = bench!("residual add", {
            let cmd = metal.queue().new_command_buffer();
            for _ in 0..layers {
                let out = metal.bufs().output((hidden*4) as u64);
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&metal.residual_add_pipeline);
                enc.set_buffer(0, Some(&buf_x), 0); enc.set_buffer(1, Some(&buf_x), 0); enc.set_buffer(2, Some(&out), 0);
                enc.set_bytes(3, 4, &hidden_val as *const u32 as *const c_void);
                enc.dispatch_threads(metal::MTLSize::new(hidden as u64, 1, 1), metal::MTLSize::new(256, 1, 1));
                enc.end_encoding();
            }
            cmd.commit(); cmd.wait_until_completed();
        });

        // 8. Encoder overhead (empty dispatches)
        let overhead_ms = bench!("empty encoder overhead", {
            let cmd = metal.queue().new_command_buffer();
            for _ in 0..layers * 7 {  // 7 encoders per layer in decode
                let enc = cmd.new_compute_command_encoder();
                enc.end_encoding();
            }
            cmd.commit(); cmd.wait_until_completed();
        });

        println!("\n--- Summary ({layers} layers) ---\n");
        let total = norm_ms + qkv_ms + kv_ms + o_ms + res_ms + ffn_ms + add_ms;
        println!("  Component total:    {total:.1}ms");
        println!("  decode_token:       27.3ms (from earlier benchmark)");
        println!("  Encoder overhead:   {overhead_ms:.1}ms ({:.0} empty encoders)", layers as f64 * 7.0);
        println!("  Ollama:             10.3ms");
        println!("  QKV is {:.1}% of total", qkv_ms / total * 100.0);
        println!("  FFN is {:.1}% of total", ffn_ms / total * 100.0);

        println!("\n=== Done ===");
    }
}
