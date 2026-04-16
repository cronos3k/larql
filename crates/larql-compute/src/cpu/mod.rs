//! CPU compute backend — BLAS for f32, C kernel for Q4.
//!
//! On macOS: Accelerate BLAS dispatches through Apple's AMX coprocessor.
//! On Linux: OpenBLAS or similar.
//! Q4: C kernel with ARM vdotq_s32 (0.95ms per 105MB matrix on M3 Max).
//!
//! ## Modules
//!
//! - `ops/f32_matmul`: BLAS sgemm dispatch
//! - `ops/q4_matvec`:  C kernel Q4×Q8 matrix-vector
//! - `ops/q4_vecmat`:  C kernel Q4 vector-matrix
//! - `ops/q4_common`:  Q8 quantization, C FFI declarations
//! - `ops/geglu`:      Element-wise GEGLU activation
//! - `ops/attention`:  Causal attention (fused QK softmax V)

pub mod ops;

// Re-export for backward compatibility (used by benchmarks/examples)
pub mod q4 {
    pub use super::ops::q4_common::{quantize_to_q8, quantize_q4_0, q4_0_matvec_c, q4_0_vecmat_c};
    pub use super::ops::q4_matvec::dispatch as q4_matvec;
    pub use super::ops::q4_vecmat::dispatch as q4_vecmat;
}

use ndarray::{Array2, ArrayView2};
use crate::backend::ComputeBackend;

/// CPU backend using BLAS (f32) and C kernel (Q4).
pub struct CpuBackend;

impl ComputeBackend for CpuBackend {
    fn matmul(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        ops::f32_matmul::matmul(a, b)
    }

    fn matmul_transb(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Array2<f32> {
        ops::f32_matmul::matmul_transb(a, b)
    }

    fn q4_matvec(
        &self, q4_data: &[u8], q8_x: &[i8], q8_scales: &[f32],
        num_rows: usize, hidden: usize,
    ) -> Option<Vec<f32>> {
        Some(ops::q4_matvec::dispatch_q8(q4_data, q8_x, q8_scales, num_rows, hidden))
    }

    fn q4_vecmat(
        &self, activation: &[f32], q4_data: &[u8],
        intermediate: usize, hidden: usize,
    ) -> Option<Vec<f32>> {
        Some(ops::q4_vecmat::dispatch(activation, q4_data, intermediate, hidden))
    }

    fn q4k_matvec(
        &self, q4k_data: &[u8], x: &[f32], num_rows: usize, hidden: usize,
    ) -> Option<Vec<f32>> {
        Some(ops::q4k_matvec::dispatch(q4k_data, x, num_rows, hidden))
    }

    fn q6k_matvec(
        &self, q6k_data: &[u8], x: &[f32], num_rows: usize, hidden: usize,
    ) -> Option<Vec<f32>> {
        Some(ops::q6k_matvec::dispatch(q6k_data, x, num_rows, hidden))
    }

    fn has_q4(&self) -> bool { true }

    fn name(&self) -> &str {
        "cpu (BLAS + C Q4 kernel)"
    }

    fn device_info(&self) -> String {
        #[cfg(target_os = "macos")]
        { "macOS Accelerate (AMX)".to_string() }
        #[cfg(all(unix, not(target_os = "macos")))]
        { "Linux OpenBLAS".to_string() }
        #[cfg(windows)]
        { "Windows CPU (matrixmultiply)".to_string() }
        #[cfg(not(any(unix, windows)))]
        { "CPU (matrixmultiply)".to_string() }
    }
}
