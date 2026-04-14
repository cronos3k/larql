//! Criterion benchmarks for compute backends.

#[cfg(unix)]
extern crate blas_src;

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use ndarray::Array2;
use larql_compute::cpu_backend;
use larql_compute::cpu::q4;

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

fn bench_matmul_transb(c: &mut Criterion) {
    let backend = cpu_backend();
    let mut group = c.benchmark_group("matmul_transb");

    for &(m, n, k) in &[(6, 2560, 2560), (6, 10240, 2560), (1, 262144, 2560)] {
        let a = synth_matrix(m, k, 42);
        let b = synth_matrix(n, k, 43);
        let label = format!("[{m},{k}]x[{n},{k}]^T");

        group.bench_with_input(BenchmarkId::new("cpu", &label), &(&a, &b), |bench, (a, b)| {
            bench.iter(|| backend.matmul_transb(a.view(), b.view()));
        });
    }

    group.finish();
}

fn bench_q4_matvec(c: &mut Criterion) {
    let hidden = 2560;
    let intermediate = 10240;
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();
    let matrix: Vec<f32> = (0..intermediate * hidden).map(|i| (i as f32 * 0.0001).cos()).collect();
    let q4_data = q4::quantize_q4_0(&matrix);

    c.bench_function("q4_matvec_cpu", |bench| {
        bench.iter(|| {
            q4::q4_matvec(&q4_data, &x, intermediate, hidden)
        });
    });
}

criterion_group!(benches, bench_matmul_transb, bench_q4_matvec);
criterion_main!(benches);
