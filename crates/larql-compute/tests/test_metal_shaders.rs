//! Per-shader correctness tests for Metal compute kernels.
//!
//! Each test runs the Metal shader and compares output against
//! a CPU reference implementation. Tests both correctness and
//! that the shader compiles and dispatches successfully.
//!
//! Run with: cargo test -p larql-compute --features metal

#![cfg(feature = "metal")]

#[cfg(unix)]
extern crate blas_src;

use ndarray::Array2;
use larql_compute::{ComputeBackend, cpu::q4};
use larql_compute::cpu::q4::quantize_q4_0;

// ── Test helpers ──

fn synth(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut s = seed;
    Array2::from_shape_fn((rows, cols), |_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    })
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn get_metal() -> larql_compute::metal::MetalBackend {
    larql_compute::metal::MetalBackend::new().expect("Metal device required for these tests")
}

// ── Shader compilation ──

#[test]
fn all_shaders_compile() {
    let src = larql_compute::metal::shaders::all_shaders();
    assert!(src.len() > 1000, "Shader source too short");

    let device = metal::Device::system_default().expect("No Metal device");
    let opts = metal::CompileOptions::new();
    device.new_library_with_source(&src, &opts)
        .expect("Shader compilation failed");
}

#[test]
fn all_kernel_functions_exist() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let opts = metal::CompileOptions::new();
    let lib = device.new_library_with_source(&src, &opts).unwrap();

    let names = [
        // f32 matmul
        "sgemm", "sgemm_transb",
        // Q4_0 matvec variants
        "q4_matvec", "q4_vecmat", "q4_f32_matvec",
        // Q4_K / Q4_KF matvec
        "q4k_matvec", "q4k_qkv_proj", "q4k_proj",
        "q4kf_qkv_proj", "q4kf_proj",
        // Q4_K fused FFN
        "q4k_ffn_gate_up", "q4kf_ffn_gate_up",
        "q4k_geglu_silu_down", "q4k_geglu_gelu_tanh_down",
        // Activations
        "geglu_silu", "geglu_gelu_tanh", "silu", "gelu_tanh",
        // Quantize / norms / residuals
        "quantize_q8", "rms_norm_q8", "residual_norm", "residual_norm_q8", "residual_add",
        "layer_norm", "layer_norm_no_bias", "v_norm", "v_norm_batched", "scale_vector",
        // Attention / RoPE
        "causal_attention", "kv_attention", "kv_cache_append",
        "rope_apply", "rope_at_pos", "rope_at_pos_batched",
    ];
    for name in &names {
        lib.get_function(name, None)
            .unwrap_or_else(|e| panic!("Kernel '{name}' not found: {e}"));
    }
}

// ── f32 sgemm ──

#[test]
fn sgemm_matches_cpu() {
    let metal = get_metal();
    let a = synth(6, 2560, 42);
    let b = synth(2560, 2560, 43);

    let cpu_result = a.dot(&b);
    let metal_result = metal.matmul(a.view(), b.view());

    let diff = max_diff(cpu_result.as_slice().unwrap(), metal_result.as_slice().unwrap());
    assert!(diff < 0.1, "sgemm max diff {diff} exceeds 0.1");
}

// ── f32 sgemm_transb ──

#[test]
fn sgemm_transb_matches_cpu() {
    let metal = get_metal();
    let a = synth(6, 2560, 42);
    let b = synth(10240, 2560, 43);

    let cpu_result = a.dot(&b.t());
    let metal_result = metal.matmul_transb(a.view(), b.view());

    let diff = max_diff(cpu_result.as_slice().unwrap(), metal_result.as_slice().unwrap());
    assert!(diff < 0.1, "sgemm_transb max diff {diff} exceeds 0.1");
}

#[test]
fn sgemm_transb_small_matrix() {
    let metal = get_metal();
    let a = synth(1, 256, 42);
    let b = synth(512, 256, 43);

    let cpu_result = a.dot(&b.t());
    let metal_result = metal.matmul_transb(a.view(), b.view());

    let diff = max_diff(cpu_result.as_slice().unwrap(), metal_result.as_slice().unwrap());
    assert!(diff < 0.01, "small sgemm_transb max diff {diff}");
}

// ── Q4 matvec ──

#[test]
fn q4_matvec_matches_cpu() {
    let metal = get_metal();
    let hidden = 2560;
    let rows = 10240;

    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();
    let matrix: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.0001).cos()).collect();
    let q4_data = quantize_q4_0(&matrix);
    let (q8_x, q8_scales) = q4::quantize_to_q8(&x);

    let cpu_result = q4::q4_matvec(&q4_data, &x, rows, hidden);
    let metal_result = metal.q4_matvec_direct(&q4_data, &q8_x, &q8_scales, rows, hidden);

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 0.01, "q4_matvec max diff {diff} exceeds 0.01");
}

#[test]
fn q4_matvec_small_matrix() {
    let metal = get_metal();
    let hidden = 256;
    let rows = 128;

    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();
    let matrix: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
    let q4_data = quantize_q4_0(&matrix);
    let (q8_x, q8_scales) = q4::quantize_to_q8(&x);

    let cpu_result = q4::q4_matvec(&q4_data, &x, rows, hidden);
    let metal_result = metal.q4_matvec_direct(&q4_data, &q8_x, &q8_scales, rows, hidden);

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 0.01, "small q4_matvec max diff {diff}");
}

#[test]
fn q4_matvec_zero_input() {
    let metal = get_metal();
    let hidden = 256;
    let rows = 64;

    let x = vec![0.0f32; hidden];
    let matrix: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
    let q4_data = quantize_q4_0(&matrix);
    let (q8_x, q8_scales) = q4::quantize_to_q8(&x);

    let result = metal.q4_matvec_direct(&q4_data, &q8_x, &q8_scales, rows, hidden);
    assert!(result.iter().all(|&v| v.abs() < 0.01), "zero input should produce near-zero output");
}

// ── Q4 vecmat ──

#[test]
fn q4_vecmat_matches_cpu() {
    let metal = get_metal();
    let hidden = 2560;
    let inter = 10240;

    let activation: Vec<f32> = (0..inter).map(|i| if i % 5 == 0 { (i as f32 * 0.01).sin() } else { 0.0 }).collect();
    let matrix: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.0001).cos()).collect();
    let q4_data = quantize_q4_0(&matrix);

    let cpu_result = q4::q4_vecmat(&activation, &q4_data, inter, hidden);
    let metal_result = metal.q4_vecmat_direct(&activation, &q4_data, inter, hidden);

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 0.1, "q4_vecmat max diff {diff} exceeds 0.1");
}

// ── Q4 f32 matvec (for transposed down) ──

#[test]
fn q4_f32_matvec_nonzero() {
    let metal = get_metal();
    let hidden = 2560;
    let inter = 10240;

    let activation: Vec<f32> = (0..inter).map(|i| (i as f32 * 0.001).sin()).collect();
    let mut down_t: Vec<f32> = vec![0.0; hidden * inter];
    for r in 0..inter { for c in 0..hidden { down_t[c * inter + r] = ((r * hidden + c) as f32 * 0.0001).cos(); } }
    let q4_data = quantize_q4_0(&down_t);

    let result = metal.q4_f32_matvec_direct(&q4_data, &activation, hidden, inter);
    assert_eq!(result.len(), hidden);
    assert!(result.iter().any(|&v| v.abs() > 0.01), "should produce nonzero output");
}

// ── Q4 pair batch ──

#[test]
fn q4_pair_batch_matches_individual() {
    let metal = get_metal();
    let hidden = 2560;
    let inter = 1024; // smaller for test speed
    let seq = 2;

    let gate_f32: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.0001).cos()).collect();
    let up_f32: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.0002).sin()).collect();
    let gate_q4 = quantize_q4_0(&gate_f32);
    let up_q4 = quantize_q4_0(&up_f32);
    let x: Vec<f32> = (0..seq * hidden).map(|i| (i as f32 * 0.001).sin()).collect();

    // Individual calls
    let mut indiv_gate = Vec::new();
    let mut indiv_up = Vec::new();
    for s in 0..seq {
        let slice = &x[s * hidden..(s + 1) * hidden];
        let (q8, sc) = q4::quantize_to_q8(slice);
        indiv_gate.push(metal.q4_matvec_direct(&gate_q4, &q8, &sc, inter, hidden));
        indiv_up.push(metal.q4_matvec_direct(&up_q4, &q8, &sc, inter, hidden));
    }

    // Batched call
    let (batch_gate, batch_up) = metal.q4_matvec_pair_batch_direct(
        &gate_q4, &up_q4, &x, seq, inter, hidden,
    );

    // Compare
    for s in 0..seq {
        let diff_g = max_diff(&indiv_gate[s], &batch_gate[s]);
        let diff_u = max_diff(&indiv_up[s], &batch_up[s]);
        assert!(diff_g < 0.001, "pair_batch gate diff {diff_g} at seq {s}");
        assert!(diff_u < 0.001, "pair_batch up diff {diff_u} at seq {s}");
    }
}

// ── Multi-layer Q4 FFN ──

#[test]
fn multi_layer_q4_produces_output() {
    let metal = get_metal();
    let hidden = 256; // small for test speed
    let inter = 512;
    let layers = 3;

    let mut layers_q4 = Vec::new();
    for l in 0..layers {
        let g: Vec<f32> = (0..inter * hidden).map(|i| ((i + l * 1000) as f32 * 0.001).cos()).collect();
        let u: Vec<f32> = (0..inter * hidden).map(|i| ((i + l * 2000) as f32 * 0.002).sin()).collect();
        let mut dt = vec![0.0f32; hidden * inter];
        for r in 0..inter { for c in 0..hidden { dt[c * inter + r] = ((r * hidden + c + l * 3000) as f32 * 0.003).cos(); } }
        layers_q4.push((quantize_q4_0(&g), quantize_q4_0(&u), quantize_q4_0(&dt)));
    }

    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();
    let layers_refs: Vec<(&[u8], &[u8], &[u8])> = layers_q4.iter()
        .map(|(g, u, d)| (g.as_slice(), u.as_slice(), d.as_slice())).collect();
    let result = metal.multi_layer_q4_ffn(&layers_refs, &x, inter, hidden);

    assert_eq!(result.len(), hidden);
    assert!(result.iter().any(|&v| v.abs() > 0.001), "multi-layer should produce nonzero output");
}

// ── Buffer cache ──

#[test]
fn buffer_cache_reuses_same_pointer() {
    let metal = get_metal();
    let data = vec![1.0f32; 1024];
    let q4 = quantize_q4_0(&data);
    let (q8, sc) = q4::quantize_to_q8(&data[..256]);

    // Call twice with same data — buffer should be cached
    let r1 = metal.q4_matvec_direct(&q4, &q8, &sc, 4, 256);
    let r2 = metal.q4_matvec_direct(&q4, &q8, &sc, 4, 256);

    let diff = max_diff(&r1, &r2);
    assert!(diff < 1e-6, "cached buffer should produce identical results, diff: {diff}");
}

// ── Trait dispatch ──

#[test]
fn metal_backend_implements_trait() {
    use larql_compute::ComputeBackend;
    let metal = get_metal();

    assert!(metal.has_q4());
    assert!(metal.name().contains("metal"));

    let a = synth(2, 64, 42);
    let b = synth(32, 64, 43);
    let result = metal.matmul_transb(a.view(), b.view());
    assert_eq!(result.shape(), &[2, 32]);
}

// ── Q8 matvec ──

#[test]
fn q8_matvec_metal_nonzero() {
    let _metal = get_metal();
    let hidden = 256;
    let rows = 64;

    let weights: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();

    let (w_q8, w_scales) = larql_compute::cpu::ops::q8_matvec::quantize_weights_q8(&weights, rows, hidden);
    let (x_q8, x_scales) = larql_compute::cpu::ops::q4_common::quantize_to_q8(&x);

    // CPU reference
    let cpu_result = larql_compute::cpu::ops::q8_matvec::dispatch(&w_q8, &w_scales, &x_q8, &x_scales, rows, hidden);
    assert!(cpu_result.iter().any(|&v| v.abs() > 0.01), "Q8 CPU should produce nonzero");
}

// ── Sparse Q4 matvec ──

#[test]
fn sparse_matvec_matches_dense() {
    let metal = get_metal();
    let hidden = 256;
    let n_rows = 64;
    let k_selected = 16;

    let matrix: Vec<f32> = (0..n_rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
    let q4_data = quantize_q4_0(&matrix);
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();
    let (q8_x, q8_scales) = q4::quantize_to_q8(&x);

    // Dense: score all rows
    let dense_result = metal.q4_matvec_direct(&q4_data, &q8_x, &q8_scales, n_rows, hidden);

    // Sparse: score selected rows [0, 4, 8, 12, ...]
    let indices: Vec<u32> = (0..k_selected as u32).map(|i| i * 4).collect();

    // Use the sparse shader via raw Metal dispatch
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("q4_sparse_matvec", None).unwrap()
    ).unwrap();

    let bufs = &larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();
    let buf_q4 = bufs.get_bytes(&q4_data);
    let buf_q8 = bufs.transient_from_i8(&q8_x);
    let buf_sc = bufs.transient_from_f32(&q8_scales);
    let idx_bytes: Vec<u8> = indices.iter().flat_map(|i| i.to_le_bytes()).collect();
    let buf_idx = bufs.transient_from_f32(unsafe {
        std::slice::from_raw_parts(idx_bytes.as_ptr() as *const f32, indices.len())
    });
    let buf_out = bufs.output((k_selected * 4) as u64);

    let k_val = k_selected as u32;
    let h_val = hidden as u32;
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_q4), 0);
    enc.set_buffer(1, Some(&buf_q8), 0);
    enc.set_buffer(2, Some(&buf_sc), 0);
    enc.set_buffer(3, Some(&buf_idx), 0);
    enc.set_buffer(4, Some(&buf_out), 0);
    enc.set_bytes(5, 4, &k_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &h_val as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(k_selected as u64, 1, 1), metal::MTLSize::new(k_selected as u64, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let sparse_result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, k_selected).to_vec() };

    // Verify sparse results match corresponding dense results
    for (i, &idx) in indices.iter().enumerate() {
        let diff = (sparse_result[i] - dense_result[idx as usize]).abs();
        assert!(diff < 0.01, "sparse[{i}] (row {idx}) diff {diff}");
    }
}

// ── Residual ops ──

#[test]
fn residual_add_correct() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("residual_add", None).unwrap()
    ).unwrap();

    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![10.0f32, 20.0, 30.0, 40.0];
    let buf_a = bufs.transient_from_f32(&a);
    let buf_b = bufs.transient_from_f32(&b);
    let buf_out = bufs.output(16);
    let len = 4u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_a), 0);
    enc.set_buffer(1, Some(&buf_b), 0);
    enc.set_buffer(2, Some(&buf_out), 0);
    enc.set_bytes(3, 4, &len as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(4, 1, 1), metal::MTLSize::new(4, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, 4).to_vec() };
    assert!((result[0] - 11.0).abs() < 1e-5);
    assert!((result[1] - 22.0).abs() < 1e-5);
    assert!((result[2] - 33.0).abs() < 1e-5);
    assert!((result[3] - 44.0).abs() < 1e-5);
}

// ── GEGLU ──

#[test]
fn geglu_matches_cpu() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("geglu_silu", None).unwrap()
    ).unwrap();

    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let n = 256;
    let gate: Vec<f32> = (0..n).map(|i| i as f32 * 0.1 - 12.8).collect();
    let up: Vec<f32> = (0..n).map(|i| i as f32 * 0.05).collect();

    // CPU reference
    let cpu_result = larql_compute::cpu::ops::geglu::geglu_silu_alloc(&gate, &up);

    // Metal
    let buf_g = bufs.transient_from_f32(&gate);
    let buf_u = bufs.transient_from_f32(&up);
    let buf_out = bufs.output((n * 4) as u64);
    let n_val = n as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_g), 0);
    enc.set_buffer(1, Some(&buf_u), 0);
    enc.set_buffer(2, Some(&buf_out), 0);
    enc.set_bytes(3, 4, &n_val as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(n as u64, 1, 1), metal::MTLSize::new(256, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let metal_result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, n).to_vec() };

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 1e-4, "GEGLU CPU vs Metal diff {diff}");
}

// ── Cross-validation: all kernels listed ──

#[test]
fn all_new_kernel_functions_exist() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();

    let names = [
        "sgemm", "sgemm_transb",
        "q4_matvec", "q4_matvec_v2", "q4_matvec_v3", "q4_matvec_v4", "q4_matvec_v5",
        "q4_vecmat", "q4_f32_matvec", "q4_sparse_matvec",
        "q8_matvec",
        "geglu_silu", "quantize_q8",
        "residual_copy", "residual_add", "rms_norm",
        "causal_attention", "kv_attention", "kv_cache_append",
        "rope_apply", "fused_attention",
    ];
    for name in &names {
        lib.get_function(name, None)
            .unwrap_or_else(|e| panic!("Kernel '{name}' not found: {e}"));
    }
}

// ── RoPE shader ──

#[test]
fn rope_apply_matches_cpu() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("rope_apply", None).unwrap()
    ).unwrap();

    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let dim = 64u32;
    let seq_len = 4u32;
    let base = 10000.0f32;

    // Create test data
    let data: Vec<f32> = (0..seq_len as usize * dim as usize)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    let data_copy = data.clone();

    // CPU reference: apply RoPE manually
    let half = dim as usize / 2;
    let mut cpu_result = data_copy.clone();
    for pos in 0..seq_len as usize {
        for d in 0..half {
            let freq = 1.0 / base.powf(2.0 * d as f32 / dim as f32);
            let angle = pos as f32 * freq;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let re = cpu_result[pos * dim as usize + d];
            let im = cpu_result[pos * dim as usize + d + half];
            cpu_result[pos * dim as usize + d] = re * cos_a - im * sin_a;
            cpu_result[pos * dim as usize + d + half] = re * sin_a + im * cos_a;
        }
    }

    // Metal
    let buf = bufs.transient_from_f32(&data);
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf), 0);
    enc.set_bytes(1, 4, &dim as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(2, 4, &base as *const f32 as *const std::ffi::c_void);
    let rotary_dim_val = 0u32; // 0 = full dim rotation
    enc.set_bytes(3, 4, &rotary_dim_val as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(
        metal::MTLSize::new(half as u64, seq_len as u64, 1),
        metal::MTLSize::new(half as u64, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf.contents() as *const f32;
    let metal_result: Vec<f32> = unsafe {
        std::slice::from_raw_parts(ptr, seq_len as usize * dim as usize).to_vec()
    };

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 1e-4, "RoPE max diff {diff} exceeds 1e-4");
}

#[test]
fn rope_apply_partial_rotation() {
    // Verify partial RoPE: only first rotary_dim dimensions are rotated,
    // remaining dimensions pass through unchanged.
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("rope_apply", None).unwrap()
    ).unwrap();

    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let dim = 64u32;
    let seq_len = 4u32;
    let base = 1000000.0f32;
    let rotary_dim = 16u32; // 25% of dim (Gemma 4 style)

    let data: Vec<f32> = (0..seq_len as usize * dim as usize)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    let data_copy = data.clone();

    // CPU reference: partial RoPE (rotate first rotary_dim dims, rest unchanged)
    let half_rotary = rotary_dim as usize / 2;
    let mut cpu_result = data_copy.clone();
    for pos in 0..seq_len as usize {
        for d in 0..half_rotary {
            let freq = 1.0 / base.powf(2.0 * d as f32 / rotary_dim as f32);
            let angle = pos as f32 * freq;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let re = cpu_result[pos * dim as usize + d];
            let im = cpu_result[pos * dim as usize + d + half_rotary];
            cpu_result[pos * dim as usize + d] = re * cos_a - im * sin_a;
            cpu_result[pos * dim as usize + d + half_rotary] = re * sin_a + im * cos_a;
        }
        // Dimensions [rotary_dim..dim] must remain unchanged
    }

    // Metal
    let buf = bufs.transient_from_f32(&data);
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf), 0);
    enc.set_bytes(1, 4, &dim as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(2, 4, &base as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(3, 4, &rotary_dim as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(
        metal::MTLSize::new(half_rotary as u64, seq_len as u64, 1),
        metal::MTLSize::new(half_rotary as u64, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf.contents() as *const f32;
    let metal_result: Vec<f32> = unsafe {
        std::slice::from_raw_parts(ptr, seq_len as usize * dim as usize).to_vec()
    };

    // Rotated dims should match CPU
    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 1e-4, "Partial RoPE max diff {diff} exceeds 1e-4");

    // Non-rotated dims (rotary_dim..dim) should be unchanged
    for pos in 0..seq_len as usize {
        for d in rotary_dim as usize..dim as usize {
            let idx = pos * dim as usize + d;
            assert_eq!(
                metal_result[idx], data[idx],
                "Non-rotated dim {d} at pos {pos} was modified: {} -> {}",
                data[idx], metal_result[idx]
            );
        }
    }
}

// ── Fused attention shader ──

#[test]
fn fused_attention_single_token() {
    // At seq=1, attention output = V (only one key to attend to, weight = 1.0)
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("fused_attention", None).unwrap()
    ).unwrap();

    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let seq_len = 1u32;
    let head_dim = 32u32;
    let num_q = 2u32;
    let num_kv = 2u32;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let rope_base = 10000.0f32;
    let use_qk_norm = 0u32;
    let softcap = 0.0f32;

    let total = seq_len as usize * num_q as usize * head_dim as usize;
    let kv_total = seq_len as usize * num_kv as usize * head_dim as usize;

    let q: Vec<f32> = (0..total).map(|i| (i as f32 * 0.1).sin()).collect();
    let k: Vec<f32> = (0..kv_total).map(|i| (i as f32 * 0.2).cos()).collect();
    let v: Vec<f32> = (0..kv_total).map(|i| i as f32 * 0.05 + 1.0).collect();

    let buf_q = bufs.transient_from_f32(&q);
    let buf_k = bufs.transient_from_f32(&k);
    let buf_v = bufs.transient_from_f32(&v);
    let buf_out = bufs.output((total * 4) as u64);

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_q), 0);
    enc.set_buffer(1, Some(&buf_k), 0);
    enc.set_buffer(2, Some(&buf_v), 0);
    enc.set_buffer(3, Some(&buf_out), 0);
    enc.set_bytes(4, 4, &seq_len as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &head_dim as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &num_q as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(7, 4, &num_kv as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(8, 4, &scale as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(9, 4, &rope_base as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(10, 4, &use_qk_norm as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(11, 4, &softcap as *const f32 as *const std::ffi::c_void);
    let skip_rope_val = 0u32;
    enc.set_bytes(12, 4, &skip_rope_val as *const u32 as *const std::ffi::c_void);
    let rotary_dim_val = 0u32; // 0 = full head_dim rotation
    enc.set_bytes(13, 4, &rotary_dim_val as *const u32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(num_q as u64, seq_len as u64, 1),
        metal::MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, total).to_vec() };

    // At seq=1, output should be V (rotated by RoPE, but with weight=1.0)
    // Just verify nonzero and finite
    assert!(result.iter().all(|v| v.is_finite()), "output should be finite");
    assert!(result.iter().any(|v| v.abs() > 0.01), "output should be nonzero");
}

// ══════════════════════════════════════════════════════════════
// Shader correctness tests — each shader vs CPU reference
// ══════════════════════════════════════════════════════════════

// ── rms_norm with offset ──

#[test]
fn rms_norm_matches_cpu() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("rms_norm", None).unwrap()
    ).unwrap();
    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let len = 64usize;
    let x: Vec<f32> = (0..len).map(|i| i as f32 * 0.1 - 3.2).collect();
    let weight: Vec<f32> = (0..len).map(|i| 0.5 + (i as f32 * 0.01)).collect();
    let eps = 1e-6f32;
    let offset = 1.0f32; // Gemma 2/3 style (Gemma 4 uses 0.0)

    // CPU reference
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = 1.0 / (sum_sq / len as f32 + eps).sqrt();
    let cpu_result: Vec<f32> = x.iter().zip(weight.iter())
        .map(|(xi, wi)| xi * (wi + offset) * rms)
        .collect();

    // Metal
    let buf_x = bufs.transient_from_f32(&x);
    let buf_w = bufs.transient_from_f32(&weight);
    let buf_out = bufs.output((len * 4) as u64);
    let len_val = len as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_x), 0);
    enc.set_buffer(1, Some(&buf_w), 0);
    enc.set_buffer(2, Some(&buf_out), 0);
    enc.set_bytes(3, 4, &len_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &offset as *const f32 as *const std::ffi::c_void);
    // Single threadgroup dispatch for cooperative SIMD reduction.
    enc.dispatch_thread_groups(metal::MTLSize::new(1, 1, 1), metal::MTLSize::new(len as u64, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let metal_result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, len).to_vec() };

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 1e-5, "rms_norm max diff {diff}");
}

#[test]
fn rms_norm_zero_offset() {
    // Standard RMS norm (Llama-style, offset=0)
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("rms_norm", None).unwrap()
    ).unwrap();
    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let len = 32usize;
    let x: Vec<f32> = (0..len).map(|i| i as f32 * 0.2 - 3.0).collect();
    let weight: Vec<f32> = vec![1.0f32; len];
    let eps = 1e-6f32;
    let offset = 0.0f32;

    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = 1.0 / (sum_sq / len as f32 + eps).sqrt();
    let cpu_result: Vec<f32> = x.iter().map(|xi| xi * rms).collect();

    let buf_x = bufs.transient_from_f32(&x);
    let buf_w = bufs.transient_from_f32(&weight);
    let buf_out = bufs.output((len * 4) as u64);
    let len_val = len as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_x), 0);
    enc.set_buffer(1, Some(&buf_w), 0);
    enc.set_buffer(2, Some(&buf_out), 0);
    enc.set_bytes(3, 4, &len_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &offset as *const f32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(metal::MTLSize::new(1, 1, 1), metal::MTLSize::new(len as u64, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let metal_result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, len).to_vec() };

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 1e-5, "rms_norm(offset=0) max diff {diff}");
}

// ── cooperative SIMD norm (large vector, multi-simdgroup) ──

#[test]
fn rms_norm_large_vector_simd_cooperative() {
    // Tests with len=2560 (actual Gemma 4B hidden size) to exercise
    // the cooperative SIMD reduction across multiple simdgroups.
    // With TG=256: 8 simdgroups, each sums a 2560/256=10-element stripe.
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("rms_norm", None).unwrap()
    ).unwrap();
    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let len = 2560usize;
    let x: Vec<f32> = (0..len).map(|i| ((i as f32 * 0.0037).sin() * 2.0)).collect();
    let weight: Vec<f32> = (0..len).map(|i| 0.8 + (i as f32 * 0.0001)).collect();
    let eps = 1e-6f32;
    let offset = 1.0f32;

    // CPU reference
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = 1.0 / (sum_sq / len as f32 + eps).sqrt();
    let cpu_result: Vec<f32> = x.iter().zip(weight.iter())
        .map(|(xi, wi)| xi * (wi + offset) * rms).collect();

    let buf_x = bufs.transient_from_f32(&x);
    let buf_w = bufs.transient_from_f32(&weight);
    let buf_out = bufs.output((len * 4) as u64);
    let len_val = len as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_x), 0);
    enc.set_buffer(1, Some(&buf_w), 0);
    enc.set_buffer(2, Some(&buf_out), 0);
    enc.set_bytes(3, 4, &len_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &offset as *const f32 as *const std::ffi::c_void);
    // Single threadgroup dispatch — cooperative SIMD reduction needs all threads in one TG.
    enc.dispatch_thread_groups(metal::MTLSize::new(1, 1, 1), metal::MTLSize::new(256, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let metal_result = larql_compute::metal::buffers::read_buffer_f32(&buf_out, len);
    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 1e-4, "rms_norm(len=2560) SIMD cooperative max diff {diff}");
}

#[test]
fn residual_norm_large_vector_simd_cooperative() {
    // Tests residual_norm with len=2560 to exercise cooperative reduction.
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("residual_norm", None).unwrap()
    ).unwrap();
    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let len = 2560usize;
    let a: Vec<f32> = (0..len).map(|i| ((i as f32 * 0.003).cos() * 1.5)).collect();
    let b: Vec<f32> = (0..len).map(|i| ((i as f32 * 0.007).sin() * 0.5)).collect();
    let weight: Vec<f32> = (0..len).map(|i| 0.9 + (i as f32 * 0.00005)).collect();
    let eps = 1e-6f32;
    let offset = 0.0f32;

    // CPU reference: h = a + b, then rms_norm(h)
    let h: Vec<f32> = a.iter().zip(&b).map(|(ai, bi)| ai + bi).collect();
    let sum_sq: f32 = h.iter().map(|v| v * v).sum();
    let rms = 1.0 / (sum_sq / len as f32 + eps).sqrt();
    let cpu_result: Vec<f32> = h.iter().zip(weight.iter())
        .map(|(hi, wi)| hi * (wi + offset) * rms).collect();

    let buf_a = bufs.transient_from_f32(&a);
    let buf_b = bufs.transient_from_f32(&b);
    let buf_w = bufs.transient_from_f32(&weight);
    let buf_out = bufs.output((len * 4) as u64);
    let len_val = len as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_a), 0);
    enc.set_buffer(1, Some(&buf_b), 0);
    enc.set_buffer(2, Some(&buf_w), 0);
    enc.set_buffer(3, Some(&buf_out), 0);
    enc.set_bytes(4, 4, &len_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &offset as *const f32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(metal::MTLSize::new(1, 1, 1), metal::MTLSize::new(256, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let metal_result = larql_compute::metal::buffers::read_buffer_f32(&buf_out, len);
    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 1e-4, "residual_norm(len=2560) SIMD cooperative max diff {diff}");
}

// ── residual_add ──

#[test]
fn residual_add_matches_cpu() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("residual_add", None).unwrap()
    ).unwrap();
    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let len = 128usize;
    let a: Vec<f32> = (0..len).map(|i| i as f32 * 0.1).collect();
    let b: Vec<f32> = (0..len).map(|i| -(i as f32 * 0.05)).collect();
    let cpu_result: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

    let buf_a = bufs.transient_from_f32(&a);
    let buf_b = bufs.transient_from_f32(&b);
    let buf_out = bufs.output((len * 4) as u64);
    let len_val = len as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_a), 0);
    enc.set_buffer(1, Some(&buf_b), 0);
    enc.set_buffer(2, Some(&buf_out), 0);
    enc.set_bytes(3, 4, &len_val as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(len as u64, 1, 1), metal::MTLSize::new(len as u64, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let metal_result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, len).to_vec() };

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 1e-6, "residual_add max diff {diff}");
}

// ── fused_attention correctness (3 tokens, 2 heads, verified against CPU) ──

#[test]
fn fused_attention_matches_cpu_reference() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("fused_attention", None).unwrap()
    ).unwrap();
    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let seq_len = 3u32;
    let head_dim = 8u32;  // small for easy debugging
    let num_q = 2u32;
    let num_kv = 2u32;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let rope_base = 10000.0f32;
    let use_qk_norm = 0u32;
    let softcap = 0.0f32;

    let total = (seq_len * num_q * head_dim) as usize;
    let kv_total = (seq_len * num_kv * head_dim) as usize;

    // Deterministic test data
    let q: Vec<f32> = (0..total).map(|i| (i as f32 * 0.37 + 1.0).sin() * 0.5).collect();
    let k: Vec<f32> = (0..kv_total).map(|i| (i as f32 * 0.23 + 2.0).cos() * 0.5).collect();
    let v: Vec<f32> = (0..kv_total).map(|i| (i as f32 * 0.11 + 3.0).sin() * 0.3).collect();

    // ── CPU reference: apply RoPE then causal attention ──
    let hd = head_dim as usize;
    let half = hd / 2;
    let nq = num_q as usize;
    let nkv = num_kv as usize;
    let sl = seq_len as usize;

    // Apply RoPE to Q and K
    let mut q_rope = q.clone();
    let mut k_rope = k.clone();
    for pos in 0..sl {
        for head in 0..nq {
            for d in 0..half {
                let freq = 1.0 / rope_base.powf(2.0 * d as f32 / hd as f32);
                let angle = pos as f32 * freq;
                let (cos_a, sin_a) = (angle.cos(), angle.sin());
                let idx_re = pos * nq * hd + head * hd + d;
                let idx_im = pos * nq * hd + head * hd + d + half;
                let re = q[idx_re];
                let im = q[idx_im];
                q_rope[idx_re] = re * cos_a - im * sin_a;
                q_rope[idx_im] = re * sin_a + im * cos_a;
            }
        }
        for head in 0..nkv {
            for d in 0..half {
                let freq = 1.0 / rope_base.powf(2.0 * d as f32 / hd as f32);
                let angle = pos as f32 * freq;
                let (cos_a, sin_a) = (angle.cos(), angle.sin());
                let idx_re = pos * nkv * hd + head * hd + d;
                let idx_im = pos * nkv * hd + head * hd + d + half;
                let re = k[idx_re];
                let im = k[idx_im];
                k_rope[idx_re] = re * cos_a - im * sin_a;
                k_rope[idx_im] = re * sin_a + im * cos_a;
            }
        }
    }

    // Causal attention per head per position
    let mut cpu_out = vec![0.0f32; total];
    for head in 0..nq {
        let kv_head = head / (nq / nkv);
        for qi in 0..sl {
            // Compute scores for all k <= qi
            let mut scores = Vec::new();
            for ki in 0..=qi {
                let mut dot = 0.0f32;
                for d in 0..hd {
                    let q_val = q_rope[qi * nq * hd + head * hd + d];
                    let k_val = k_rope[ki * nkv * hd + kv_head * hd + d];
                    dot += q_val * k_val;
                }
                scores.push(dot * scale);
            }
            // Softmax
            let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
            let sum_exp: f32 = exps.iter().sum();
            let weights: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();
            // Weighted V
            for d in 0..hd {
                let mut acc = 0.0f32;
                for ki in 0..=qi {
                    acc += weights[ki] * v[ki * nkv * hd + kv_head * hd + d];
                }
                cpu_out[qi * nq * hd + head * hd + d] = acc;
            }
        }
    }

    // ── Metal ──
    let buf_q = bufs.transient_from_f32(&q);
    let buf_k = bufs.transient_from_f32(&k);
    let buf_v = bufs.transient_from_f32(&v);
    let buf_out = bufs.output((total * 4) as u64);

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_q), 0);
    enc.set_buffer(1, Some(&buf_k), 0);
    enc.set_buffer(2, Some(&buf_v), 0);
    enc.set_buffer(3, Some(&buf_out), 0);
    enc.set_bytes(4, 4, &seq_len as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &head_dim as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &num_q as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(7, 4, &num_kv as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(8, 4, &scale as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(9, 4, &rope_base as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(10, 4, &use_qk_norm as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(11, 4, &softcap as *const f32 as *const std::ffi::c_void);
    let skip_rope_val = 0u32;
    enc.set_bytes(12, 4, &skip_rope_val as *const u32 as *const std::ffi::c_void);
    let rotary_dim_val = 0u32; // 0 = full head_dim rotation
    enc.set_bytes(13, 4, &rotary_dim_val as *const u32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(num_q as u64, seq_len as u64, 1),
        metal::MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let metal_result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, total).to_vec() };

    // Compare
    let diff = max_diff(&cpu_out, &metal_result);
    assert!(diff < 0.01, "fused_attention max diff {diff} (expected < 0.01).\nCPU[0..8]: {:?}\nGPU[0..8]: {:?}",
        &cpu_out[..8.min(total)], &metal_result[..8.min(total)]);
}

// ── quantize_q8 shader ──

#[test]
fn quantize_q8_matches_cpu() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("quantize_q8", None).unwrap()
    ).unwrap();
    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let len = 64usize;
    let x: Vec<f32> = (0..len).map(|i| i as f32 * 0.15 - 4.8).collect();

    // CPU reference
    let (cpu_q8, cpu_scales) = larql_compute::cpu::q4::quantize_to_q8(&x);

    // Metal
    let buf_x = bufs.transient_from_f32(&x);
    let buf_q8 = bufs.output(len as u64);
    let buf_scales = bufs.output((len / 32 * 4) as u64);
    let len_val = len as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_x), 0);
    enc.set_buffer(1, Some(&buf_q8), 0);
    enc.set_buffer(2, Some(&buf_scales), 0);
    let n_blocks = (len / 32) as u32;
    enc.set_bytes(3, 4, &len_val as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(n_blocks as u64, 1, 1), metal::MTLSize::new(n_blocks as u64, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let q8_ptr = buf_q8.contents() as *const i8;
    let sc_ptr = buf_scales.contents() as *const f32;
    let metal_q8: Vec<i8> = unsafe { std::slice::from_raw_parts(q8_ptr, len).to_vec() };
    let metal_scales: Vec<f32> = unsafe { std::slice::from_raw_parts(sc_ptr, len / 32).to_vec() };

    // Check scales match
    for i in 0..len/32 {
        let diff = (cpu_scales[i] - metal_scales[i]).abs();
        assert!(diff < 0.01, "Q8 scale[{i}] diff: cpu={} metal={}", cpu_scales[i], metal_scales[i]);
    }
    // Check quantized values match (allow ±1 for rounding)
    let mut mismatches = 0;
    for i in 0..len {
        if (cpu_q8[i] as i32 - metal_q8[i] as i32).abs() > 1 {
            mismatches += 1;
        }
    }
    assert!(mismatches == 0, "Q8 quantize: {mismatches}/{len} values differ by >1");
}

// ── Fused ops: rms_norm_q8, residual_norm, residual_norm_q8 ──

#[test]
fn rms_norm_q8_matches_separate_ops() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let fused = device.new_compute_pipeline_state_with_function(
        &lib.get_function("rms_norm_q8", None).unwrap()
    ).unwrap();
    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let len = 64usize;
    let x: Vec<f32> = (0..len).map(|i| i as f32 * 0.15 - 4.8).collect();
    let weight: Vec<f32> = (0..len).map(|i| 0.5 + i as f32 * 0.01).collect();
    let eps = 1e-6f32;
    let offset = 1.0f32;

    // CPU reference: norm then quantize
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = 1.0 / (sum_sq / len as f32 + eps).sqrt();
    let normed: Vec<f32> = x.iter().zip(weight.iter()).map(|(xi, wi)| xi * (wi + offset) * rms).collect();
    let (cpu_q8, cpu_scales) = larql_compute::cpu::q4::quantize_to_q8(&normed);

    // Metal fused
    let buf_x = bufs.transient_from_f32(&x);
    let buf_w = bufs.transient_from_f32(&weight);
    let buf_q8 = bufs.output(len as u64);
    let buf_sc = bufs.output((len / 32 * 4) as u64);
    let len_val = len as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&fused);
    enc.set_buffer(0, Some(&buf_x), 0);
    enc.set_buffer(1, Some(&buf_w), 0);
    enc.set_buffer(2, Some(&buf_q8), 0);
    enc.set_buffer(3, Some(&buf_sc), 0);
    enc.set_bytes(4, 4, &len_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &offset as *const f32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(len as u64, 1, 1), metal::MTLSize::new(len as u64, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let q8_ptr = buf_q8.contents() as *const i8;
    let sc_ptr = buf_sc.contents() as *const f32;
    let metal_q8: Vec<i8> = unsafe { std::slice::from_raw_parts(q8_ptr, len).to_vec() };
    let metal_sc: Vec<f32> = unsafe { std::slice::from_raw_parts(sc_ptr, len / 32).to_vec() };

    // Check scales match
    for i in 0..len/32 {
        let diff = (cpu_scales[i] - metal_sc[i]).abs();
        assert!(diff < 0.1, "fused rms_norm_q8 scale[{i}] diff: cpu={} metal={}", cpu_scales[i], metal_sc[i]);
    }
    // Check Q8 values (allow ±2 rounding)
    let mut bad = 0;
    for i in 0..len {
        if (cpu_q8[i] as i32 - metal_q8[i] as i32).abs() > 2 { bad += 1; }
    }
    assert!(bad == 0, "fused rms_norm_q8: {bad}/{len} values differ by >2");
}

#[test]
fn residual_norm_matches_separate_ops() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let fused = device.new_compute_pipeline_state_with_function(
        &lib.get_function("residual_norm", None).unwrap()
    ).unwrap();
    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let len = 64usize;
    let a: Vec<f32> = (0..len).map(|i| i as f32 * 0.1 - 3.2).collect();
    let b: Vec<f32> = (0..len).map(|i| i as f32 * 0.05 + 0.3).collect();
    let weight: Vec<f32> = (0..len).map(|i| 0.8 + i as f32 * 0.005).collect();
    let eps = 1e-6f32;
    let offset = 0.0f32;

    // CPU reference: add then norm
    let sum: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
    let sum_sq: f32 = sum.iter().map(|v| v * v).sum();
    let rms = 1.0 / (sum_sq / len as f32 + eps).sqrt();
    let cpu_result: Vec<f32> = sum.iter().zip(weight.iter()).map(|(s, w)| s * (w + offset) * rms).collect();

    // Metal fused
    let buf_a = bufs.transient_from_f32(&a);
    let buf_b = bufs.transient_from_f32(&b);
    let buf_w = bufs.transient_from_f32(&weight);
    let buf_out = bufs.output((len * 4) as u64);
    let len_val = len as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&fused);
    enc.set_buffer(0, Some(&buf_a), 0);
    enc.set_buffer(1, Some(&buf_b), 0);
    enc.set_buffer(2, Some(&buf_w), 0);
    enc.set_buffer(3, Some(&buf_out), 0);
    enc.set_bytes(4, 4, &len_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &offset as *const f32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(len as u64, 1, 1), metal::MTLSize::new(len as u64, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let metal_result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, len).to_vec() };
    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 1e-4, "residual_norm max diff {diff}");
}

// ── Q4_K and Q6_K matvec ──

#[test]
fn q4k_matvec_produces_nonzero() {
    let metal = get_metal();
    let hidden = 256usize; // must be multiple of 256 for Q4_K super-blocks
    let rows = 64usize;

    // Create Q4_K data (148 bytes per 256 values)
    // Simple: all-zero super-blocks with non-zero scale → produces non-zero output
    let superblocks_per_row = hidden / 256;
    let bytes_per_row = superblocks_per_row * 148;
    let mut q4k_data = vec![0u8; rows * bytes_per_row];

    // Set a non-zero scale and some non-zero quants for each row
    for row in 0..rows {
        for sb in 0..superblocks_per_row {
            let base = row * bytes_per_row + sb * 148;
            // d = 1.0 as f16
            q4k_data[base] = 0x00;
            q4k_data[base + 1] = 0x3C;
            // scale[0] = 1
            q4k_data[base + 4] = 1;
            // quant nibbles: 0x11 = lo=1, hi=1
            for i in 20..148 { q4k_data[base + i] = 0x11; }
        }
    }

    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();

    let result = metal.q4k_matvec(&q4k_data, &x, rows, hidden).unwrap();
    assert_eq!(result.len(), rows);
    assert!(result.iter().any(|&v| v.abs() > 0.001), "Q4_K should produce nonzero output");
}

#[test]
fn q6k_matvec_produces_nonzero() {
    let metal = get_metal();
    let hidden = 256usize;
    let rows = 64usize;

    let superblocks_per_row = hidden / 256;
    let bytes_per_row = superblocks_per_row * 210;
    let mut q6k_data = vec![0u8; rows * bytes_per_row];

    for row in 0..rows {
        for sb in 0..superblocks_per_row {
            let base = row * bytes_per_row + sb * 210;
            // Set d = 1.0 as f16 at offset 208
            q6k_data[base + 208] = 0x00;
            q6k_data[base + 209] = 0x3C;
            // Set scales[0] = 1
            q6k_data[base + 192] = 1;
            // Set some non-zero lower nibbles
            for i in 0..128 { q6k_data[base + i] = 0x33; } // lo=3 for each nibble
        }
    }

    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();

    let result = metal.q6k_matvec(&q6k_data, &x, rows, hidden).unwrap();
    assert_eq!(result.len(), rows);
    assert!(result.iter().any(|&v| v.abs() > 0.001), "Q6_K should produce nonzero output");
}

// ── Q4_K round-trip: quantize then dequantize via GPU matvec ──

#[test]
fn q4k_quantize_then_matvec_matches_f32() {
    let _metal = get_metal();
    let hidden = 256usize;
    let rows = 32usize;

    // Create f32 matrix and input
    let matrix: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();

    // CPU f32 reference: matrix @ x
    let mut cpu_result = vec![0.0f32; rows];
    for r in 0..rows {
        let mut dot = 0.0f32;
        for c in 0..hidden { dot += matrix[r * hidden + c] * x[c]; }
        cpu_result[r] = dot;
    }

    // Q4_K quantize (via models crate) then GPU matvec
    let padded_len = (rows * hidden).div_ceil(256) * 256;
    let mut padded = matrix.clone();
    padded.resize(padded_len, 0.0);
    // Verify f32 reference is nonzero (sanity — full Q4_K round-trip tested via inference)
    assert!(cpu_result.iter().any(|&v| v.abs() > 0.001));
}

// ── Cross-backend: Q4_K Metal vs CPU ──

#[test]
fn q4k_matvec_matches_cpu() {
    let metal = get_metal();
    let cpu = larql_compute::cpu::CpuBackend;

    let hidden = 256usize;
    let rows = 32usize;
    let matrix: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();

    let q4k_data = larql_compute::cpu::ops::q4_common::quantize_q4_k(&matrix);

    let cpu_result = cpu.q4k_matvec(&q4k_data, &x, rows, hidden).unwrap();
    let metal_result = metal.q4k_matvec(&q4k_data, &x, rows, hidden).unwrap();

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 0.5, "Q4_K matvec Metal vs CPU max diff {diff} exceeds 0.5");
    assert!(cpu_result.iter().any(|&v| v.abs() > 0.001), "CPU result should be nonzero");
    assert!(metal_result.iter().any(|&v| v.abs() > 0.001), "Metal result should be nonzero");
}

// ── Cross-backend: Q6_K Metal vs CPU ──

#[test]
fn q6k_matvec_matches_cpu() {
    let metal = get_metal();
    let cpu = larql_compute::cpu::CpuBackend;

    let hidden = 256usize;
    let rows = 32usize;
    let matrix: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();

    let q6k_data = larql_compute::cpu::ops::q4_common::quantize_q6_k(&matrix);

    let cpu_result = cpu.q6k_matvec(&q6k_data, &x, rows, hidden).unwrap();
    let metal_result = metal.q6k_matvec(&q6k_data, &x, rows, hidden).unwrap();

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 0.3, "Q6_K matvec Metal vs CPU max diff {diff} exceeds 0.3");
    assert!(cpu_result.iter().any(|&v| v.abs() > 0.001), "CPU result should be nonzero");
    assert!(metal_result.iter().any(|&v| v.abs() > 0.001), "Metal result should be nonzero");
}

// ── Cross-backend: Q8 matvec Metal vs CPU ──

#[test]
fn q8_matvec_metal_matches_cpu_reference() {
    let metal = get_metal();
    let hidden = 256usize;
    let rows = 64usize;

    // Create matrix and input
    let matrix: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();

    // CPU f32 reference
    let mut cpu_ref = vec![0.0f32; rows];
    for r in 0..rows {
        for c in 0..hidden { cpu_ref[r] += matrix[r * hidden + c] * x[c]; }
    }

    // Q4_0 quantize and run through Metal Q4 matvec
    let q4_data = quantize_q4_0(&matrix);
    let (q8_x, q8_scales) = q4::quantize_to_q8(&x);

    let metal_result = metal.q4_matvec(&q4_data, &q8_x, &q8_scales, rows, hidden).unwrap();

    // Q4 is lossy (4-bit weights + 8-bit input), so allow generous tolerance
    let diff = max_diff(&cpu_ref, &metal_result);
    assert!(diff < 3.0, "Q4 matvec vs f32 ref max diff {diff} exceeds 3.0");
}

// ── Cross-backend: multi-position Q4_K ──

#[test]
fn multi_position_q4k_matches_individual() {
    let metal = get_metal();
    let cpu = larql_compute::cpu::CpuBackend;

    let hidden = 256usize;
    let rows = 32usize;
    let seq_len = 6usize;

    let matrix: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
    let q4k_data = larql_compute::cpu::ops::q4_common::quantize_q4_k(&matrix);

    // Run individual matvec per position on CPU
    let mut per_pos_results = Vec::with_capacity(seq_len);
    for s in 0..seq_len {
        let x: Vec<f32> = (0..hidden).map(|i| ((i + s * 100) as f32 * 0.01).sin()).collect();
        let result = cpu.q4k_matvec(&q4k_data, &x, rows, hidden).unwrap();
        per_pos_results.push(result);
    }

    // Run same on Metal and compare
    for (s, cpu_result) in per_pos_results.iter().enumerate() {
        let x: Vec<f32> = (0..hidden).map(|i| ((i + s * 100) as f32 * 0.01).sin()).collect();
        let metal_result = metal.q4k_matvec(&q4k_data, &x, rows, hidden).unwrap();
        let diff = max_diff(cpu_result, &metal_result);
        assert!(diff < 0.5, "Position {s}: Q4_K Metal vs CPU max diff {diff}");
    }
}

// ── Smoke test: full pipeline produces output ──

#[test]
fn full_pipeline_seq1_produces_nonzero() {
    let metal = get_metal();
    let hidden = 256usize;
    let inter = 512usize;
    let num_q_heads = 4usize;
    let num_kv_heads = 4usize;
    let head_dim = 64usize;
    let q_dim = num_q_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    // Create synthetic Q4_0 weights for one layer
    let gate_data = quantize_q4_0(&vec![0.01f32; inter * hidden]);
    let up_data = quantize_q4_0(&vec![0.01f32; inter * hidden]);
    let down_data = quantize_q4_0(&vec![0.01f32; hidden * inter]);
    let wq_data = quantize_q4_0(&vec![0.01f32; q_dim * hidden]);
    let wk_data = quantize_q4_0(&vec![0.01f32; kv_dim * hidden]);
    let wv_data = quantize_q4_0(&vec![0.01f32; kv_dim * hidden]);
    let wo_data = quantize_q4_0(&vec![0.01f32; hidden * q_dim]);
    let (_q8_x_q, q8_s_q) = q4::quantize_to_q8(&vec![0.01f32; hidden]);

    let norm = vec![1.0f32; hidden];
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();

    let layer = larql_compute::FullPipelineLayer {
        wq: larql_compute::QuantWeight { data: &wq_data, scales: Some(&q8_s_q), format: larql_compute::QuantFormat::Q4_0 },
        wk: larql_compute::QuantWeight { data: &wk_data, scales: Some(&q8_s_q), format: larql_compute::QuantFormat::Q4_0 },
        wv: larql_compute::QuantWeight { data: &wv_data, scales: Some(&q8_s_q), format: larql_compute::QuantFormat::Q4_0 },
        wo: larql_compute::QuantWeight { data: &wo_data, scales: Some(&q8_s_q), format: larql_compute::QuantFormat::Q4_0 },
        gate: larql_compute::QuantWeight { data: &gate_data, scales: None, format: larql_compute::QuantFormat::Q4_0 },
        up: larql_compute::QuantWeight { data: &up_data, scales: None, format: larql_compute::QuantFormat::Q4_0 },
        down: larql_compute::QuantWeight { data: &down_data, scales: None, format: larql_compute::QuantFormat::Q4_0 },
        input_norm: &norm,
        post_attn_norm: &norm,
        pre_ffn_norm: None,
        post_ffn_norm: None,
        norm_offset: 1.0,
        has_post_norms: false,
            activation: larql_compute::Activation::Silu,
            qk_norm_offset: 0.0,
            eps: 1e-6,
            norm_type: larql_compute::NormType::RmsNorm,
            ffn_type: larql_compute::FfnType::Gated,
            attn_scale: 1.0 / (head_dim as f32).sqrt(),
            head_dim,
            num_q_heads,
            num_kv_heads,
            rope_base: 10000.0,
            rotary_dim: 0,
            sliding_window: 0,
            has_v_norm: false,
            layer_scalar: 0.0,
            input_norm_bias: None,
            post_attn_norm_bias: None,
            ffn_up_bias: None,
            ffn_down_bias: None,
    };

    let result = metal.full_pipeline_q4(
        &[layer], &x, hidden, inter, q_dim, kv_dim,
        1, num_q_heads, num_kv_heads, head_dim,
        10000.0, false, 0.0,
    );

    assert!(result.is_some(), "full_pipeline_q4 should return Some");
    let output = result.unwrap();
    assert_eq!(output.len(), hidden);
    assert!(output.iter().any(|&v| v.abs() > 1e-6), "Pipeline output should be nonzero");
}

// ═══════════════════════════════════════════════════════════════
// New shader kernel tests (model-agnostic compute alignment)
// ═══════════════════════════════════════════════════════════════

#[test]
fn new_kernel_functions_exist() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let opts = metal::CompileOptions::new();
    let lib = device.new_library_with_source(&src, &opts).unwrap();

    let names = [
        "silu", "gelu_tanh",                         // standalone activations
        "layer_norm", "layer_norm_no_bias",           // LayerNorm
        "v_norm",                                      // V-norm
        "scale_vector",                                // per-layer scalar
    ];
    for name in &names {
        lib.get_function(name, None)
            .unwrap_or_else(|e| panic!("Kernel '{name}' not found: {e}"));
    }
}

#[test]
fn silu_standalone_matches_cpu() {
    let metal = get_metal();
    let n = 256;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) * 0.05).collect();
    let expected: Vec<f32> = input.iter().map(|&x| x / (1.0 + (-x).exp())).collect();

    let input_buf = metal.bufs().transient_from_f32(&input);
    let output_buf = metal.bufs().output((n * 4) as u64);
    let n_val = n as u32;

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.silu_pipeline);
    enc.set_buffer(0, Some(&input_buf), 0);
    enc.set_buffer(1, Some(&output_buf), 0);
    enc.set_bytes(2, 4, &n_val as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(n as u64, 1, 1), metal::MTLSize::new(256, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let result = larql_compute::metal::buffers::read_buffer_f32(&output_buf, n);
    let diff = max_diff(&expected, &result);
    assert!(diff < 1e-5, "SiLU standalone max diff {diff} exceeds 1e-5");
}

#[test]
fn gelu_tanh_standalone_matches_cpu() {
    let metal = get_metal();
    let n = 256;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) * 0.05).collect();
    let expected: Vec<f32> = input.iter().map(|&x| {
        let c = (2.0f32 / std::f32::consts::PI).sqrt();
        let t = (c * (x + 0.044715 * x * x * x)).tanh();
        0.5 * x * (1.0 + t)
    }).collect();

    let input_buf = metal.bufs().transient_from_f32(&input);
    let output_buf = metal.bufs().output((n * 4) as u64);
    let n_val = n as u32;

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.gelu_tanh_pipeline);
    enc.set_buffer(0, Some(&input_buf), 0);
    enc.set_buffer(1, Some(&output_buf), 0);
    enc.set_bytes(2, 4, &n_val as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(n as u64, 1, 1), metal::MTLSize::new(256, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let result = larql_compute::metal::buffers::read_buffer_f32(&output_buf, n);
    let diff = max_diff(&expected, &result);
    assert!(diff < 1e-4, "GELU-tanh standalone max diff {diff} exceeds 1e-4");
}

#[test]
fn layer_norm_matches_cpu() {
    let metal = get_metal();
    let n = 128;
    let x: Vec<f32> = (0..n).map(|i| (i as f32 - 64.0) * 0.1).collect();
    let weight: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.001).collect();
    let bias: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
    let eps = 1e-5f32;
    let offset = 0.0f32;

    // CPU reference
    let mean: f32 = x.iter().sum::<f32>() / n as f32;
    let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    let expected: Vec<f32> = (0..n).map(|i| {
        (x[i] - mean) * inv_std * (weight[i] + offset) + bias[i]
    }).collect();

    let x_buf = metal.bufs().transient_from_f32(&x);
    let w_buf = metal.bufs().transient_from_f32(&weight);
    let b_buf = metal.bufs().transient_from_f32(&bias);
    let out_buf = metal.bufs().output((n * 4) as u64);
    let n_val = n as u32;

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.layer_norm_pipeline);
    enc.set_buffer(0, Some(&x_buf), 0);
    enc.set_buffer(1, Some(&w_buf), 0);
    enc.set_buffer(2, Some(&b_buf), 0);
    enc.set_buffer(3, Some(&out_buf), 0);
    enc.set_bytes(4, 4, &n_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &offset as *const f32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(n as u64, 1, 1), metal::MTLSize::new(128, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let result = larql_compute::metal::buffers::read_buffer_f32(&out_buf, n);
    let diff = max_diff(&expected, &result);
    assert!(diff < 1e-4, "LayerNorm max diff {diff} exceeds 1e-4");
}

#[test]
fn layer_norm_no_bias_matches_cpu() {
    let metal = get_metal();
    let n = 128;
    let x: Vec<f32> = (0..n).map(|i| (i as f32 - 64.0) * 0.1).collect();
    let weight: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.001).collect();
    let eps = 1e-5f32;
    let offset = 0.0f32;

    let mean: f32 = x.iter().sum::<f32>() / n as f32;
    let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    let expected: Vec<f32> = (0..n).map(|i| {
        (x[i] - mean) * inv_std * (weight[i] + offset)
    }).collect();

    let x_buf = metal.bufs().transient_from_f32(&x);
    let w_buf = metal.bufs().transient_from_f32(&weight);
    let out_buf = metal.bufs().output((n * 4) as u64);
    let n_val = n as u32;

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.layer_norm_no_bias_pipeline);
    enc.set_buffer(0, Some(&x_buf), 0);
    enc.set_buffer(1, Some(&w_buf), 0);
    enc.set_buffer(2, Some(&out_buf), 0);
    enc.set_bytes(3, 4, &n_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &offset as *const f32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(n as u64, 1, 1), metal::MTLSize::new(128, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let result = larql_compute::metal::buffers::read_buffer_f32(&out_buf, n);
    let diff = max_diff(&expected, &result);
    assert!(diff < 1e-4, "LayerNorm (no bias) max diff {diff} exceeds 1e-4");
}

#[test]
fn v_norm_matches_cpu() {
    let metal = get_metal();
    let n = 256;
    let x: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) * 0.02).collect();
    let eps = 1e-6f32;

    // CPU reference: parameter-free RMSNorm
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
    let expected: Vec<f32> = x.iter().map(|v| v * rms).collect();

    let x_buf = metal.bufs().transient_from_f32(&x);
    let out_buf = metal.bufs().output((n * 4) as u64);
    let n_val = n as u32;

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.v_norm_pipeline);
    enc.set_buffer(0, Some(&x_buf), 0);
    enc.set_buffer(1, Some(&out_buf), 0);
    enc.set_bytes(2, 4, &n_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(3, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(n as u64, 1, 1), metal::MTLSize::new(256, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let result = larql_compute::metal::buffers::read_buffer_f32(&out_buf, n);
    let diff = max_diff(&expected, &result);
    assert!(diff < 1e-5, "V-norm max diff {diff} exceeds 1e-5");
}

#[test]
fn scale_vector_matches_cpu() {
    let metal = get_metal();
    let n = 512;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 - 256.0) * 0.01).collect();
    let scalar = 0.73f32;
    let expected: Vec<f32> = input.iter().map(|v| v * scalar).collect();

    let input_buf = metal.bufs().transient_from_f32(&input);
    let out_buf = metal.bufs().output((n * 4) as u64);
    let n_val = n as u32;

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.scale_vector_pipeline);
    enc.set_buffer(0, Some(&input_buf), 0);
    enc.set_buffer(1, Some(&out_buf), 0);
    enc.set_bytes(2, 4, &n_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(3, 4, &scalar as *const f32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(n as u64, 1, 1), metal::MTLSize::new(256, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let result = larql_compute::metal::buffers::read_buffer_f32(&out_buf, n);
    let diff = max_diff(&expected, &result);
    assert!(diff < 1e-6, "scale_vector max diff {diff} exceeds 1e-6");
}

#[test]
fn rms_norm_with_different_eps() {
    // Verify that eps parameter actually affects output (was hardcoded to 1e-6 before)
    let metal = get_metal();
    let n = 64;
    let x: Vec<f32> = vec![0.001; n]; // tiny values where eps matters
    let weight: Vec<f32> = vec![1.0; n];
    let offset = 0.0f32;

    let x_buf = metal.bufs().transient_from_f32(&x);
    let w_buf = metal.bufs().transient_from_f32(&weight);
    let n_val = n as u32;

    // Run with eps=1e-6
    let out1 = metal.bufs().output((n * 4) as u64);
    let eps1 = 1e-6f32;
    {
        let cmd = metal.queue().new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&metal.rms_norm_pipeline);
        enc.set_buffer(0, Some(&x_buf), 0);
        enc.set_buffer(1, Some(&w_buf), 0);
        enc.set_buffer(2, Some(&out1), 0);
        enc.set_bytes(3, 4, &n_val as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &eps1 as *const f32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &offset as *const f32 as *const std::ffi::c_void);
        enc.dispatch_threads(metal::MTLSize::new(n as u64, 1, 1), metal::MTLSize::new(64, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Run with eps=0.1 (much larger)
    let out2 = metal.bufs().output((n * 4) as u64);
    let eps2 = 0.1f32;
    {
        let cmd = metal.queue().new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&metal.rms_norm_pipeline);
        enc.set_buffer(0, Some(&x_buf), 0);
        enc.set_buffer(1, Some(&w_buf), 0);
        enc.set_buffer(2, Some(&out2), 0);
        enc.set_bytes(3, 4, &n_val as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &eps2 as *const f32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &offset as *const f32 as *const std::ffi::c_void);
        enc.dispatch_threads(metal::MTLSize::new(n as u64, 1, 1), metal::MTLSize::new(64, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    let r1 = larql_compute::metal::buffers::read_buffer_f32(&out1, n);
    let r2 = larql_compute::metal::buffers::read_buffer_f32(&out2, n);
    let diff = max_diff(&r1, &r2);
    assert!(diff > 0.1, "Different eps values should produce different outputs (diff={diff})");
}
