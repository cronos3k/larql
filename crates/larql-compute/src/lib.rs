//! # larql-compute
//!
//! Hardware-accelerated compute backends for LARQL.
//!
//! Provides the [`ComputeBackend`] trait that abstracts all hardware-specific
//! matrix operations. Every LARQL crate (inference, vindex) uses this trait —
//! the caller never knows whether the operation runs on CPU or GPU.
//!
//! ## Backends
//!
//! | Backend | Feature | Operations |
//! |---------|---------|------------|
//! | CPU | (always) | BLAS f32, C kernel Q4 (ARM vdotq_s32), vector ops |
//! | Metal | `metal` | Tiled f32, simdgroup Q4, multi-layer pipeline |
//! | CUDA | (planned) | — |
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use larql_compute::{ComputeBackend, default_backend, cpu_backend, dot, norm, cosine};
//!
//! let backend = default_backend();
//! println!("Using: {}", backend.name());
//! ```
//!
//! ## Feature flags
//!
//! - `metal`: Metal GPU backend (macOS only). Adds optimised Q4 shaders,
//!   multi-layer pipeline, zero-copy mmap buffers.
//! - `cuda`: (planned) CUDA GPU backend.

// Pull in the platform BLAS backend on Unix (Accelerate on macOS, OpenBLAS on Linux).
// On Windows we rely on ndarray's pure-Rust matrixmultiply backend instead.
#[cfg(unix)]
extern crate blas_src;

pub mod backend;
pub mod cpu;
pub mod pipeline;

#[cfg(feature = "metal")]
pub mod metal;

#[cfg(feature = "cuda")]
pub mod cuda;

// ── Re-exports: pipeline types ──

pub use pipeline::{
    QuantFormat, QuantWeight,
    NormType, FfnType, Activation,
    FullPipelineLayer,
};

// ── Re-exports: backend ──

pub use backend::{ComputeBackend, MatMulOp, dot_proj_gpu, matmul_gpu};
pub use cpu::CpuBackend;
pub use cpu::ops::vector::{dot, norm, cosine};

#[cfg(feature = "metal")]
pub use metal::MetalBackend;

#[cfg(feature = "cuda")]
pub use cuda::CudaBackend;

/// Create the best available backend.
///
/// Priority: Metal (macOS) > CUDA (Windows/Linux) > CPU.
///
/// - `--features metal`: Metal GPU on macOS, auto-calibrates hybrid threshold.
/// - `--features cuda`:  cuBLAS on any CUDA GPU (RTX 4090, etc.).
/// - No GPU feature:     CPU — Accelerate on macOS, OpenBLAS on Linux, matrixmultiply on Windows.
///
/// # Example
/// ```rust,no_run
/// let backend = larql_compute::default_backend();
/// println!("{} ({})", backend.name(), backend.device_info());
/// ```
pub fn default_backend() -> Box<dyn ComputeBackend> {
    #[cfg(feature = "metal")]
    {
        if let Some(m) = metal::MetalBackend::new() {
            m.calibrate();
            return Box::new(m);
        }
        eprintln!("[compute] Metal not available, falling back to CPU");
    }

    #[cfg(feature = "cuda")]
    {
        if let Some(c) = cuda::CudaBackend::new() {
            return Box::new(c);
        }
        eprintln!("[compute] CUDA not available, falling back to CPU");
    }

    Box::new(cpu::CpuBackend)
}

/// Force CPU-only backend. No GPU, no calibration overhead.
///
/// Use when you want deterministic CPU execution or to benchmark
/// CPU vs GPU paths.
pub fn cpu_backend() -> Box<dyn ComputeBackend> {
    Box::new(cpu::CpuBackend)
}
