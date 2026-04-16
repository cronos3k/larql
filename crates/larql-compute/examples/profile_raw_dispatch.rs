//! Raw kernel dispatch: JUST the Q4_K matvec, nothing else. Measures pure GPU cost.

#[cfg(unix)]
extern crate blas_src;

fn main() {
    #[cfg(not(feature = "metal"))]
    { println!("Run with --features metal"); return; }

    #[cfg(feature = "metal")]
    {
        use std::time::Instant;
        use larql_compute::cpu::ops::q4_common::quantize_q4_k;

        let metal = larql_compute::metal::MetalBackend::new().expect("Metal required");

        let hidden = 2560usize;
        let q_dim = 2560usize;
        let kv_dim = 1280usize;
        let n = 100;

        fn pad(d: &[f32]) -> Vec<f32> { let p = d.len().div_ceil(256)*256; let mut o = d.to_vec(); o.resize(p, 0.0); o }

        let wq = quantize_q4_k(&pad(&(0..q_dim*hidden).map(|i| (i as f32*0.0001).cos()).collect::<Vec<_>>()));
        let wk = quantize_q4_k(&pad(&(0..kv_dim*hidden).map(|i| (i as f32*0.0002).sin()).collect::<Vec<_>>()));
        let wv = quantize_q4_k(&pad(&(0..kv_dim*hidden).map(|i| (i as f32*0.0003).cos()).collect::<Vec<_>>()));
        let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();

        let buf_wq = metal.bufs().get_bytes(&wq);
        let buf_wk = metal.bufs().get_bytes(&wk);
        let buf_wv = metal.bufs().get_bytes(&wv);
        let buf_x = metal.bufs().transient_from_f32(&x);

        use larql_compute::metal::shaders::q4k_qkv_proj as sh;
        let total = (q_dim + kv_dim + kv_dim) as u32;
        let num_tgs = (total as u64).div_ceil(sh::ROWS_PER_TG);

        println!("=== Raw Q4_K QKV Kernel ===");
        println!("QKV: {total} rows × {hidden} hidden\n");

        // Single dispatch benchmark
        for _ in 0..5 {
            let buf_qo = metal.bufs().output((q_dim * 4) as u64);
            let buf_ko = metal.bufs().output((kv_dim * 4) as u64);
            let buf_vo = metal.bufs().output((kv_dim * 4) as u64);
            let cmd = metal.queue().new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&metal.q4k_qkv_proj_pipeline);
            enc.set_buffer(0, Some(&buf_wq), 0);
            enc.set_buffer(1, Some(&buf_wk), 0);
            enc.set_buffer(2, Some(&buf_wv), 0);
            enc.set_buffer(3, Some(&buf_x), 0);
            enc.set_buffer(4, Some(&buf_qo), 0);
            enc.set_buffer(5, Some(&buf_ko), 0);
            enc.set_buffer(6, Some(&buf_vo), 0);
            let q_rows = q_dim as u32; let k_rows = kv_dim as u32; let v_rows = kv_dim as u32; let k_val = hidden as u32;
            enc.set_bytes(7, 4, &q_rows as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(8, 4, &k_rows as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(9, 4, &v_rows as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(10, 4, &k_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(metal::MTLSize::new(num_tgs, 1, 1), metal::MTLSize::new(sh::THREADS_PER_TG, 1, 1));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }

        // 1 dispatch per cmd buffer
        let t0 = Instant::now();
        for _ in 0..n {
            let buf_qo = metal.bufs().output((q_dim * 4) as u64);
            let buf_ko = metal.bufs().output((kv_dim * 4) as u64);
            let buf_vo = metal.bufs().output((kv_dim * 4) as u64);
            let cmd = metal.queue().new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&metal.q4k_qkv_proj_pipeline);
            enc.set_buffer(0, Some(&buf_wq), 0); enc.set_buffer(1, Some(&buf_wk), 0);
            enc.set_buffer(2, Some(&buf_wv), 0); enc.set_buffer(3, Some(&buf_x), 0);
            enc.set_buffer(4, Some(&buf_qo), 0); enc.set_buffer(5, Some(&buf_ko), 0);
            enc.set_buffer(6, Some(&buf_vo), 0);
            let q_rows = q_dim as u32; let k_rows = kv_dim as u32; let v_rows = kv_dim as u32; let k_val = hidden as u32;
            enc.set_bytes(7, 4, &q_rows as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(8, 4, &k_rows as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(9, 4, &v_rows as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(10, 4, &k_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(metal::MTLSize::new(num_tgs, 1, 1), metal::MTLSize::new(sh::THREADS_PER_TG, 1, 1));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        let single_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;

        // 34 dispatches in ONE cmd buffer (simulating 34-layer QKV)
        let t0 = Instant::now();
        for _ in 0..n {
            let cmd = metal.queue().new_command_buffer();
            for _ in 0..34 {
                let buf_qo = metal.bufs().output((q_dim * 4) as u64);
                let buf_ko = metal.bufs().output((kv_dim * 4) as u64);
                let buf_vo = metal.bufs().output((kv_dim * 4) as u64);
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&metal.q4k_qkv_proj_pipeline);
                enc.set_buffer(0, Some(&buf_wq), 0); enc.set_buffer(1, Some(&buf_wk), 0);
                enc.set_buffer(2, Some(&buf_wv), 0); enc.set_buffer(3, Some(&buf_x), 0);
                enc.set_buffer(4, Some(&buf_qo), 0); enc.set_buffer(5, Some(&buf_ko), 0);
                enc.set_buffer(6, Some(&buf_vo), 0);
                let q_rows = q_dim as u32; let k_rows = kv_dim as u32; let v_rows = kv_dim as u32; let k_val = hidden as u32;
                enc.set_bytes(7, 4, &q_rows as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(8, 4, &k_rows as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(9, 4, &v_rows as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(10, 4, &k_val as *const u32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(metal::MTLSize::new(num_tgs, 1, 1), metal::MTLSize::new(sh::THREADS_PER_TG, 1, 1));
                enc.end_encoding();
            }
            cmd.commit();
            cmd.wait_until_completed();
        }
        let batch_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let per_layer = batch_ms / 34.0;

        let data_mb = (wq.len() + wk.len() + wv.len()) as f64 / 1e6;
        println!("  1 QKV dispatch:         {single_ms:.3}ms  ({:.1} GB/s)", data_mb / single_ms);
        println!("  34 QKV dispatches (1 cmd): {batch_ms:.2}ms  ({per_layer:.3}ms/layer)");
        println!("  Ollama total (34 layers): ~10.3ms (0.303ms/layer for EVERYTHING)");
        println!("  Our QKV alone per layer: {per_layer:.3}ms ({:.1}x Ollama's entire layer)", per_layer / 0.303);

        println!("\n=== Done ===");
    }
}
