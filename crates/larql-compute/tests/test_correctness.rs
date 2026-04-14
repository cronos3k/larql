//! Correctness tests: verify all backends produce matching output.

#[cfg(unix)]
extern crate blas_src;

use ndarray::Array2;
use larql_compute::cpu_backend;
use larql_compute::cpu::q4::quantize_q4_0;

fn synth_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut state = seed;
    let data: Vec<f32> = (0..rows * cols)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect();
    Array2::from_shape_vec((rows, cols), data).unwrap()
}

fn max_diff(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

#[test]
fn cpu_matmul_matches_ndarray() {
    let cpu = cpu_backend();
    let a = synth_matrix(6, 2560, 42);
    let b = synth_matrix(2560, 2560, 43);
    let expected = a.dot(&b);
    let result = cpu.matmul(a.view(), b.view());
    assert!(max_diff(&expected, &result) < 1e-5, "matmul mismatch");
}

#[test]
fn cpu_matmul_transb_matches_ndarray() {
    let cpu = cpu_backend();
    let a = synth_matrix(6, 2560, 42);
    let b = synth_matrix(10240, 2560, 43);
    let expected = a.dot(&b.t());
    let result = cpu.matmul_transb(a.view(), b.view());
    assert!(max_diff(&expected, &result) < 1e-5, "matmul_transb mismatch");
}

#[test]
fn cpu_has_q4() {
    let cpu = cpu_backend();
    assert!(cpu.has_q4(), "CPU backend should support Q4");
}

#[test]
fn cpu_q4_matvec_nonzero() {
    use larql_compute::cpu::q4;

    let hidden = 256; // small for test speed
    let rows = 128;
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();
    let matrix: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();

    // Quantize matrix to Q4
    let q4_data = quantize_q4_0(&matrix);
    let (q8_x, q8_scales) = q4::quantize_to_q8(&x);

    let cpu = cpu_backend();
    let result = cpu.q4_matvec(&q4_data, &q8_x, &q8_scales, rows, hidden).unwrap();

    assert_eq!(result.len(), rows);
    assert!(result.iter().any(|&v| v.abs() > 0.01), "Q4 matvec should produce nonzero output");
}

#[test]
fn cpu_q4_vecmat_nonzero() {
    use larql_compute::cpu::q4;

    let hidden = 256;
    let inter = 128;
    let activation: Vec<f32> = (0..inter).map(|i| if i % 3 == 0 { 1.0 } else { 0.0 }).collect();
    let matrix: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
    let q4_data = quantize_q4_0(&matrix);

    let result = q4::q4_vecmat(&activation, &q4_data, inter, hidden);
    assert_eq!(result.len(), hidden);
    assert!(result.iter().any(|&v| v.abs() > 0.01), "Q4 vecmat should produce nonzero output");
}

#[test]
fn default_backend_has_name() {
    let be = larql_compute::default_backend();
    assert!(!be.name().is_empty());
}

