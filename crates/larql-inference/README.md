# larql-inference

Inference engine for transformer models. Forward pass, BLAS-fused attention, hardware-accelerated matmul backends, and pluggable FFN routing.

## Overview

This crate runs transformer forward passes with Apple Accelerate (AMX) and optional Metal GPU acceleration. It uses `larql-vindex` for gate KNN (sparse feature selection) and `larql-models` for weight loading and architecture definitions.

```rust
use larql_inference::InferenceModel;

// Load a model
let model = InferenceModel::load("google/gemma-3-4b-it")?;

// Run inference
let result = larql_inference::predict(
    model.weights(), model.tokenizer(), &token_ids, 5,
);
println!("Top prediction: {} ({:.1}%)", result.predictions[0].0, result.predictions[0].1 * 100.0);
```

## Key Components

| Module | Purpose |
|--------|---------|
| `backend/` | MatMulBackend trait: CPU (Accelerate BLAS) and Metal GPU with auto-calibration |
| `attention.rs` | BLAS-fused GQA attention with online softmax (no [seq,seq] allocation) |
| `forward.rs` | Forward pass: `predict()`, `predict_with_ffn()`, `forward_to_layer()` |
| `ffn/` | FFN evaluation: dense, sparse, highway, cached, experimental |
| `residual.rs` | RMS norm, layer norm |
| `trace/` | Residual stream decomposition and tiered storage |
| `vindex/walk_ffn.rs` | WalkFfn: mmap'd FFN — faster than dense (517ms vs 535ms) |
| `capture.rs` | Residual stream vector capture for probing |
| `walker/` | Weight-level graph walkers (no forward pass) |
| `model.rs` | Model loading (re-exports from larql-models) |

## Matmul Backend

All large matrix multiplications dispatch through the `MatMulBackend` trait:

```rust
use larql_inference::backend::{default_backend, MatMulBackend};

let backend = default_backend();  // Auto-selects CPU or Metal, calibrates
println!("Using: {}", backend.name());

let c = backend.matmul_transb(&input, &weights);
```

**CPU backend** (default): ndarray + `cblas_sgemm` via Apple Accelerate. AMX hardware at ~2-4 TFLOPS f32.

**Metal GPU backend** (`--features metal`): Tiled 32x32 compute shaders with buffer cache and auto-calibrated FLOP threshold. Weight matrices are uploaded to GPU once and reused across all calls.

```bash
# Build with Metal support
cargo build --release -p larql-inference --features metal
```

See [docs/inference-engine.md](../../docs/inference-engine.md) for architecture details and benchmarks.

## BLAS-Fused Attention

The attention kernel uses BLAS `gemv` inside an online-softmax loop. For each query position:

1. `scores = K[0..=qi] @ Q[qi]` (BLAS gemv, AMX-accelerated)
2. Scale + optional softcap + two-pass softmax (f64 accumulation)
3. `output = V[0..=qi]^T @ softmax_scores` (BLAS gemv)

Never allocates the `[seq, seq]` attention matrix. At Gemma-3's head_dim=256, **1.6x faster** than the materialized path. Supports GQA, softcap (Gemma2), attention weight capture.

## WalkFfn

The WalkFfn replaces the dense down projection with a zero-copy mmap read from the vindex:

1. Gate + up projections from model weights (exact, same as dense)
2. GEGLU activation (exact, same as dense)
3. Down projection from mmap'd `down_features.bin` (zero-copy, feature-major)
4. Result is identical to dense FFN — **and faster** (517ms vs 535ms)

The mmap'd feature-major layout has better page cache behavior than the safetensors weight layout.

Build the required vindex files:
```bash
cargo run --release -p larql-vindex --example convert_gates_f32 -- path/to/vindex
cargo run --release -p larql-vindex --example build_down_features -- path/to/vindex
cargo run --release -p larql-vindex --example build_up_features -- path/to/vindex
```

### Walk-only mode

Drop FFN weights — 16.6GB → 5.5GB:

```rust
let model = InferenceModel::load_walk_only("google/gemma-3-4b-it")?;
// Frees 10.7 GB of FFN tensors. Requires down_features.bin + up_features.bin.
```

### Server

```bash
cargo run --release -p larql-server -- path/to/vindex --port 8080

curl -X POST http://localhost:8080/v1/infer \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "top": 5, "mode": "walk"}'
```

## Examples

```bash
# Walk inference benchmark (dense vs walk vs HNSW, needs model + vindex)
cargo run --release -p larql-inference --example bench_walk_inference -- \
  --model google/gemma-3-4b-it --vindex path/to/vindex

# Walk boundary sweep (correctness proof across all 34 layers)
cargo run --release -p larql-inference --example walk_boundary_sweep -- \
  --model google/gemma-3-4b-it --vindex path/to/vindex

# Fused attention demo and benchmark
cargo run --release -p larql-inference --example attention_demo
cargo run --release -p larql-inference --example bench_attention

# Backend demo and benchmark (CPU vs Metal)
cargo run --release -p larql-inference --example backend_demo --features metal
cargo run --release -p larql-inference --example bench_backend --features metal

# Full inference benchmark (needs model weights)
cargo run --release -p larql-inference --example bench_inference

# End-to-end inference demo (needs model weights)
cargo run --release -p larql-inference --example inference_demo

# Clustering and pair matching demos
cargo run -p larql-inference --example clustering_demo
cargo run -p larql-inference --example pair_matching_demo
```

### Vindex tools

```bash
# Convert gate vectors from f16 to f32 (zero-copy mmap)
cargo run --release -p larql-vindex --example convert_gates_f32 -- path/to/vindex

# Build feature-major down vectors (contiguous per-feature layout)
cargo run --release -p larql-vindex --example build_down_features -- path/to/vindex
```

## Tests

```bash
# Inference tests (115 with Metal)
cargo test -p larql-inference --features metal

# HNSW tests
cargo test -p larql-vindex --test test_hnsw --release

# Individual test suites
cargo test -p larql-inference --test test_fused_attention   # 18 tests
cargo test -p larql-inference --test test_backend           # 13+6 tests
cargo test -p larql-inference --test test_modules           # 15 tests
cargo test -p larql-inference --test test_trace             # 14 tests
cargo test -p larql-inference --test test_walkers           # 12 tests
cargo test -p larql-inference --test test_walker_utils      # 10 tests
```

| Area | Tests | Coverage |
|------|-------|----------|
| Backend (unit + integration) | 34 | Shape, correctness, batch, Metal vs CPU, factory |
| Fused attention | 18 | GQA, softcap, capture, reference agreement, edge cases |
| HNSW index | 7 | Build, search, recall, scores, empty, sorted |
| FFN | 9 | SiLU, GELU, dense, highway, multi-position |
| Attention/residual | 10 | RoPE, GQA, RMS norm, per-head norm |
| Trace stores | 14 | Write/read, tiers, boundaries, additive property |
| Walkers | 12 | Weight/attention walkers, vector extractor |
| Utils | 10 | Top-k, rounding, entity sorting |

## Crate Dependencies

```
larql-models      ModelWeights, architecture traits, quant
    |
larql-vindex      VectorIndex (types, core, gate, walk, hnsw, mutate, router)
    |
larql-inference   Forward pass, attention, backends, WalkFfn
```

### Vindex module structure

| Module | Responsibility |
|--------|---------------|
| `types` | FeatureMeta, GateIndex trait, WalkHit, callbacks |
| `core` | VectorIndex struct, constructors, loading, accessors |
| `gate` | Gate KNN: search, batch, scores, HNSW, warmup |
| `walk` | Walk FFN data: mmap'd down/up feature-major vectors |
| `hnsw` | HNSW graph index |
| `mutate` | INSERT/DELETE mutations |
| `router` | MoE expert routing |

## License

Apache-2.0
