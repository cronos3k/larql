# Fork Notes — cronos3k/larql

## Origin

This repository is a fork of **[chrishayuk/larql](https://github.com/chrishayuk/larql)**,
the original work of **Chris Hayuk** ([@chrishayuk](https://github.com/chrishayuk)).

All core concepts, the LQL language design, the vindex format, the extraction pipeline,
the inference engine, and the vast majority of the codebase are the work of Chris Hayuk.
This fork exists solely to extend platform support — the intellectual foundation is entirely his.

---

## Why This Fork Exists

The original codebase was written and tested on **macOS with Apple Silicon (Metal GPU)**.
It used Accelerate (Apple's BLAS) unconditionally and contained Metal-specific paths as
the only GPU compute backend.

This fork makes LARQL build and run correctly on **Windows and Linux** without modification,
and adds a **CUDA backend** so that NVIDIA GPU owners get hardware-accelerated extraction
and inference.

**Use this fork if you are on Windows or Linux with an NVIDIA GPU.**  
**Use the original if you are on macOS — it is better tested there.**

---

## What Is Different

### New: CUDA compute backend (`larql-compute`)

A new `CudaBackend` using [cudarc 0.12](https://github.com/coreylowman/cudarc) and cuBLAS
provides GPU-accelerated GEMM for vindex extraction and walk queries on NVIDIA hardware.

Activated automatically when a CUDA device is present (`--features cuda` at build time):

```bash
cargo build --release --features cuda
```

The dispatch priority at runtime is: **Metal → CUDA → CPU**.

Technical note: cuBLAS operates in column-major order. Row-major input matrices
`A[m×k]` and `B[k×n]` are submitted with swapped operands and swapped M/N dimensions —
`sgemm(N, N, n, m, k, B, lda=n, A, ldb=k, C, ldc=n)` — so no transpose copies are
needed.

### Fixed: Platform-conditional BLAS

The original Cargo configuration linked Apple Accelerate unconditionally, which broke
compilation on any non-Apple target.

BLAS is now selected per-platform:

| Platform     | BLAS provider            |
|--------------|--------------------------|
| macOS        | Apple Accelerate (AMX)   |
| Linux        | OpenBLAS                 |
| Windows      | ndarray matrixmultiply   |
| CUDA feature | cuBLAS (all non-macOS)   |

### Fixed: MSVC C compiler compatibility (`q4_dot.c`)

The Q4 quantisation kernel used `__builtin_memcpy`, a GCC/Clang intrinsic not available
in the MSVC toolchain. Replaced with standard `memcpy` (+ `#include <string.h>`).

### Fixed: Cross-compilation build flags (`build.rs`)

`build.rs` used Rust `#[cfg()]` attributes at compile-host time to select SIMD flags,
meaning a host running one architecture could produce wrong flags for a different target.
Replaced with `CARGO_CFG_TARGET_ARCH` and `CARGO_CFG_TARGET_ENV` environment variables,
which reflect the actual compilation target. MSVC targets receive `/arch:AVX2`; all other
x86_64 targets receive `-mavx2`.

### Fixed: Cargo.toml TOML section scoping (`larql-vindex`)

`hf-hub` and `reqwest` were accidentally placed inside a
`[target.'cfg(unix)'.dependencies]` section. TOML section headers apply to all
subsequent entries until the next header, so these dependencies were invisible on
Windows, causing `E0433` errors at compile time. Both crates are now in the
unconditional `[dependencies]` table.

### Fixed: HuggingFace model resolution on Windows (`larql-models`)

`resolve_model_path` only checked `$HOME` for the HF cache directory. Windows does not
set `$HOME`; the correct variable is `USERPROFILE`. Added fallback chain:
`HUGGINGFACE_HUB_CACHE` → `HF_HOME` → `HOME` → `USERPROFILE`. Added auto-download via
the `hf-hub` crate when a model is not present locally.

### Fixed: `extern crate blas_src` in examples, benches, and tests

All example, benchmark, and test files that contained a bare `extern crate blas_src;`
declaration now wrap it in `#[cfg(unix)]`, since that crate is only linked on Unix
platforms (macOS + Linux). Without this guard, Windows builds failed with an unresolved
extern.

---

## Verified Results

Extraction was tested on **Qwen/Qwen2.5-0.5B-Instruct** on a Windows machine with an
NVIDIA RTX 4090 (CUDA 12.6). CPU and CUDA vindexes were extracted independently and
compared across multiple walk queries. All feature IDs, gate scores, and token labels
matched exactly, confirming the CUDA extraction produces numerically identical output
to the CPU reference path.

Example — `WALK "The capital of France is"` layers 20–23 (Layer 23 top features):

```
F1030  gate=+0.122  hears="是/is"    down=[是, is, are, 的是, 是对]
F765   gate=-0.116  hears="都不能"    down=[都不能, 都无法, 均可, 都将]
F1296  gate=+0.107  hears="has"      down=[has, does, hasn, doesn]
```

Both vindexes returned the same result to all decimal places.

---

## Contributions at a Glance

| Area | Original (chrishayuk) | This fork (cronos3k) |
|---|---|---|
| LQL language design | ✓ | — |
| Vindex format | ✓ | — |
| Extraction pipeline | ✓ | — |
| Inference engine | ✓ | — |
| Metal GPU backend | ✓ | — |
| macOS / Apple Silicon support | ✓ | — |
| Windows build fixes | — | ✓ |
| Linux build fixes | — | ✓ |
| CUDA / cuBLAS backend | — | ✓ |
| Platform-conditional BLAS | — | ✓ |
| MSVC C compiler fixes | — | ✓ |
| HF model auto-download on Windows | — | ✓ |

---

## License

Apache-2.0 — same as the upstream repository.
See [LICENSE](LICENSE) for the full text.

Original copyright: Chris Hayuk  
Fork modifications copyright: Gregor Koch
