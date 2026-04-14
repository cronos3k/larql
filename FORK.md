# Fork Notes — cronos3k/larql

## Origin

This repository is a fork of **[chrishayuk/larql](https://github.com/chrishayuk/larql)**,
the original work of **Chris Hayuk** ([@chrishayuk](https://github.com/chrishayuk)).

All core concepts, the LQL language design, the vindex format, the extraction pipeline,
and the inference engine are the work of Chris Hayuk. The intellectual foundation of
this project is entirely his. This fork exists to take that foundation and make it
available to the widest possible audience.

---

## What This Fork Adds

The original was a powerful but isolated tool — command-line only, macOS/Apple Silicon
only. That means it required owning specific Apple hardware, being comfortable in a
terminal, knowing the LQL syntax, and building from source with Rust. In practice, that
limited it to a small group of researchers on a single platform.

This fork makes two complementary contributions:

**1. Platform portability** — the tool now builds and runs on Windows, Linux, and macOS,
with hardware-accelerated paths on NVIDIA GPUs (CUDA) in addition to the original
Apple Metal backend. The same vindex you extract on a Mac works identically on a Windows
server with an RTX 4090.

**2. A Gradio web interface** — a browser-based GUI that makes the entire LARQL feature
set accessible without a terminal, without knowing LQL syntax, and without installing
Rust. Anyone with a HuggingFace account can run it as a Space. Anyone with the repo
cloned can run it with `python demo/app.py`.

Together these mean the work Chris built is no longer locked to a single platform or a
single type of user. It can now be explored by ML researchers, students, and curious
people on any hardware, through any browser.

---

## The Gradio Demo (`demo/`)

A full interactive web interface built with Gradio 6, designed to be published as a
HuggingFace Space or run locally.

```bash
pip install -r demo/requirements.txt
python demo/app.py
# → http://localhost:7860
```

| Tab | Purpose |
|---|---|
| 🔍 **Walk Explorer** | Enter any prompt, choose layer range and top-K; see a parsed table of active FFN features with gate scores, "hears" tokens, and output token predictions |
| 🧪 **Knowledge Probe** | Compare three prompts side-by-side at a single layer to identify shared vs. concept-specific features |
| 💻 **LQL Console** | Free-form LQL execution with one-click example queries; active vindex injected automatically |
| 📊 **Vindex Info** | Model metadata viewer (architecture, layer bands, extract level) with live SHA256 checksum verification |
| ⬇️ **Extract** | Download a model from HuggingFace Hub and extract a vindex — no terminal needed |
| ℹ️ **Setup & About** | Build instructions, LQL quick-reference, environment diagnostics |

**HuggingFace Spaces:** `demo/README.md` contains the Spaces YAML frontmatter
(`sdk: gradio`, `sdk_version: 6.2.0`). `demo/setup.sh` builds the Rust binary from
source on the Linux container at Space startup. To deploy: create a new Gradio Space
and point it at the `demo/` folder.

---

## Platform Portability Changes

### New: CUDA compute backend (`larql-compute`)

A new `CudaBackend` using [cudarc 0.12](https://github.com/coreylowman/cudarc) and
cuBLAS provides GPU-accelerated GEMM for vindex extraction and walk queries on NVIDIA
hardware. Activated with `--features cuda` at build time.

Compute dispatch priority at runtime: **Metal → CUDA → CPU**.

Technical note: cuBLAS operates in column-major order. Row-major input matrices
`A[m×k]` and `B[k×n]` are submitted with swapped operands and swapped M/N dimensions —
`sgemm(N, N, n, m, k, B, lda=n, A, ldb=k, C, ldc=n)` — so no transpose copies are
needed.

### Fixed: Platform-conditional BLAS

The original Cargo configuration linked Apple Accelerate unconditionally, breaking
compilation on any non-Apple target. BLAS is now selected per platform:

| Platform     | BLAS provider          |
|--------------|------------------------|
| macOS        | Apple Accelerate (AMX) |
| Linux        | OpenBLAS               |
| Windows      | ndarray matrixmultiply |
| CUDA feature | cuBLAS (all non-macOS) |

### Fixed: MSVC C compiler compatibility (`q4_dot.c`)

The Q4 quantisation kernel used `__builtin_memcpy`, a GCC/Clang intrinsic unavailable
in the MSVC toolchain. Replaced with standard `memcpy` (`#include <string.h>`).

### Fixed: Cross-compilation build flags (`build.rs`)

`build.rs` used Rust `#[cfg()]` attributes at compile-host time to select SIMD flags.
Replaced with `CARGO_CFG_TARGET_ARCH` and `CARGO_CFG_TARGET_ENV`, which reflect the
actual compilation target. MSVC targets receive `/arch:AVX2`; other x86_64 targets
receive `-mavx2`.

### Fixed: Cargo.toml TOML section scoping (`larql-vindex`)

`hf-hub` and `reqwest` were inside a `[target.'cfg(unix)'.dependencies]` section due
to TOML section-header scoping rules, making them invisible on Windows and causing
`E0433` compile errors. Both are now in the unconditional `[dependencies]` table.

### Fixed: HuggingFace model resolution on Windows (`larql-models`)

`resolve_model_path` only checked `$HOME`, which is not set on Windows. Added fallback
chain: `HUGGINGFACE_HUB_CACHE` → `HF_HOME` → `HOME` → `USERPROFILE`. Added
auto-download via the `hf-hub` crate when a model is not found locally.

### Fixed: `extern crate blas_src` in examples, benches, and tests

All 25+ example and test files that contained a bare `extern crate blas_src;`
declaration now wrap it in `#[cfg(unix)]`, since that crate is only linked on Unix.
Without this guard, Windows builds failed with an unresolved extern.

---

## Verified Results

Extraction was tested on **Qwen/Qwen2.5-0.5B-Instruct** on a Windows machine with an
NVIDIA RTX 4090 (CUDA 12.6). CPU and CUDA vindexes were extracted independently and
compared across multiple walk queries. All feature IDs, gate scores, and token labels
matched exactly — the CUDA extraction produces numerically identical output to the CPU
reference path.

Sample — `WALK "The capital of France is"` layers 20–23, Layer 23 top features:

```
F1030  gate=+0.122  hears="是/is"   down=[是, is, are, 的是, 是对]
F765   gate=-0.116  hears="都不能"   down=[都不能, 都无法, 均可, 都将]
F1296  gate=+0.107  hears="has"     down=[has, does, hasn, doesn]
```

CPU and CUDA vindexes returned the same output to all decimal places across every
prompt tested.

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
| **Gradio web interface** | — | ✓ |
| **HuggingFace Spaces deployment** | — | ✓ |
| Windows build support | — | ✓ |
| Linux build support | — | ✓ |
| CUDA / cuBLAS backend | — | ✓ |
| Platform-conditional BLAS | — | ✓ |
| MSVC C compiler fixes | — | ✓ |
| HF model auto-download on Windows | — | ✓ |

---

## License

Apache-2.0 — same as the upstream repository.
See [LICENSE](LICENSE) for the full text.

Original copyright: Chris Hayuk  
Fork additions copyright: Gregor Koch
