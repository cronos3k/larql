//! CUDA compute backend — cuBLAS for f32 GEMM, scalar Q4 fallback.
//!
//! Uses the RTX 4090 (or any CUDA GPU) for large matrix multiplications,
//! which are the bottleneck during vindex extraction (down_meta, embeddings)
//! and transformer inference.
//!
//! ## How row-major ndarray talks to column-major cuBLAS
//!
//! cuBLAS assumes column-major storage. For row-major arrays we use the identity:
//!   C[m×n] = A[m×k] * B[k×n]
//!   ⟺  C^T[n×m] = B_data_colmaj[n×k] * A_data_colmaj[k×m]
//!
//! So we swap A and B in the cuBLAS call, swap M and N, and keep OP_N for both.

use std::sync::Arc;
use ndarray::{Array2, ArrayView2};
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, sys::cublasOperation_t};
use cudarc::driver::CudaDevice;

use crate::backend::ComputeBackend;

/// CUDA GPU backend using cuBLAS for GEMM operations.
///
/// Falls back to the CPU Q4 kernel for quantized operations.
pub struct CudaBackend {
    dev:  Arc<CudaDevice>,
    blas: CudaBlas,
}

impl CudaBackend {
    /// Try to initialise CUDA device 0. Returns `None` if CUDA is unavailable.
    pub fn new() -> Option<Self> {
        let dev = CudaDevice::new(0).ok()?;
        let blas = CudaBlas::new(dev.clone()).ok()?;
        eprintln!("[compute] CUDA backend: device 0 ({})", dev.name().unwrap_or_default());
        Some(Self { dev, blas })
    }

    /// Row-major SGEMM: C[m×n] = A[m×k] × B[k×n].
    ///
    /// Uploads A and B to GPU, runs cuBLAS SGEMM, downloads C.
    fn gemm_row_major(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        // Upload operands
        let a_dev = self.dev.htod_sync_copy(a).expect("htod A");
        let b_dev = self.dev.htod_sync_copy(b).expect("htod B");
        let mut c_dev = self.dev.alloc_zeros::<f32>(m * n).expect("alloc C");

        // Row-major trick: swap operands and swap m/n.
        // C^T[n×m] = B_rowmaj[n×k] (as colmaj) × A_rowmaj[k×m] (as colmaj)
        // cuBLAS: M=n, N=m, K=k, A=B_data lda=n, B=A_data ldb=k
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: 1.0f32,
            lda: n as i32,
            ldb: k as i32,
            beta: 0.0f32,
            ldc: n as i32,
        };

        unsafe {
            self.blas.gemm(cfg, &b_dev, &a_dev, &mut c_dev).expect("cuBLAS sgemm");
        }

        self.dev.dtoh_sync_copy(&c_dev).expect("dtoh C")
    }

    /// Row-major SGEMM with transposed B: C[m×n] = A[m×k] × B^T where B is [n×k].
    fn gemm_row_major_transb(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        // C[m×n] = A[m×k] × B^T  (B is [n×k])
        // C^T[n×m] = B_rowmaj[n×k] (colmaj = B^T[k×n]) OP_T × A_rowmaj[k×m] (colmaj)
        // cuBLAS: M=n, N=m, K=k, transa=T lda=k, transb=N ldb=k
        let a_dev = self.dev.htod_sync_copy(a).expect("htod A");
        let b_dev = self.dev.htod_sync_copy(b).expect("htod B");
        let mut c_dev = self.dev.alloc_zeros::<f32>(m * n).expect("alloc C");

        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: 1.0f32,
            lda: k as i32,   // B is [n×k] row-major, leading dim = k
            ldb: k as i32,   // A is [m×k] row-major, leading dim = k
            beta: 0.0f32,
            ldc: n as i32,
        };

        unsafe {
            self.blas.gemm(cfg, &b_dev, &a_dev, &mut c_dev).expect("cuBLAS sgemm_transb");
        }

        self.dev.dtoh_sync_copy(&c_dev).expect("dtoh C")
    }
}

impl ComputeBackend for CudaBackend {
    fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        let (m, k) = (a.nrows(), a.ncols());
        let n = b.ncols();
        debug_assert_eq!(k, b.nrows(), "matmul dimension mismatch");

        let a_sl = a.as_standard_layout();
        let b_sl = b.as_standard_layout();
        let c_flat = self.gemm_row_major(a_sl.as_slice().unwrap(), b_sl.as_slice().unwrap(), m, k, n);

        Array2::from_shape_vec((m, n), c_flat).expect("Array2 from cuBLAS output")
    }

    fn matmul_transb(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        let (m, k) = (a.nrows(), a.ncols());
        let n = b.nrows(); // B is [n×k]
        debug_assert_eq!(k, b.ncols(), "matmul_transb dimension mismatch");

        let a_sl = a.as_standard_layout();
        let b_sl = b.as_standard_layout();
        let c_flat = self.gemm_row_major_transb(a_sl.as_slice().unwrap(), b_sl.as_slice().unwrap(), m, k, n);

        Array2::from_shape_vec((m, n), c_flat).expect("Array2 from cuBLAS output")
    }

    // Q4 operations fall back to the CPU scalar kernel.
    fn q4_matvec(
        &self, q4_data: &[u8], q8_x: &[i8], q8_scales: &[f32],
        num_rows: usize, hidden: usize,
    ) -> Option<Vec<f32>> {
        // TODO: custom CUDA Q4 kernel — for now use the CPU path
        None
    }

    fn q4_vecmat(
        &self, activation: &[f32], q4_data: &[u8],
        intermediate: usize, hidden: usize,
    ) -> Option<Vec<f32>> {
        None
    }

    fn q4k_matvec(
        &self, _q4k_data: &[u8], _x: &[f32], _num_rows: usize, _hidden: usize,
    ) -> Option<Vec<f32>> {
        None
    }

    fn q6k_matvec(
        &self, _q6k_data: &[u8], _x: &[f32], _num_rows: usize, _hidden: usize,
    ) -> Option<Vec<f32>> {
        None
    }

    fn has_q4(&self) -> bool { false }

    fn name(&self) -> &str { "cuda (cuBLAS)" }

    fn device_info(&self) -> String {
        let name = self.dev.name().unwrap_or_else(|_| "unknown".into());
        format!("CUDA {name}")
    }
}
