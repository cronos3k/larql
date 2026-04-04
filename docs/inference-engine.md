# Inference Engine

The inference engine in `larql-inference` runs transformer forward passes with hardware-accelerated matmul backends and BLAS-fused attention. It supports dense, sparse, cached, and graph-walk FFN backends, and plugs into the full LARQL pipeline (INFER, TRACE, WalkFfn).

## Architecture

```
larql-inference/src/
  backend/
    mod.rs          MatMulBackend trait, factory, auto-calibration
    cpu.rs          CPU backend (ndarray + Accelerate BLAS / AMX)
    metal.rs        Metal GPU backend (tiled compute shaders, buffer cache)
  attention.rs      BLAS-fused GQA attention (online softmax, no [seq,seq] alloc)
  forward.rs        Forward pass: embed → layers → logits
  ffn/              FFN backends: dense, sparse, highway, experimental
  residual.rs       RMS norm, layer norm
  trace/            Residual stream decomposition and storage
  vindex/           WalkFfn (sparse FFN via vindex gate KNN)
  walker/           Weight-level graph walkers (no forward pass)
```

## Matmul Backend

All large matrix multiplications dispatch through the `MatMulBackend` trait, which routes to the optimal hardware path.

### CPU Backend (default)

Uses ndarray `.dot()` which dispatches through `cblas_sgemm` via Apple Accelerate on macOS. The AMX coprocessor on M-series handles the actual computation at ~2-4 TFLOPS f32. Zero dispatch overhead.

```toml
# Already configured in Cargo.toml
ndarray = { version = "0.16", features = ["blas"] }
blas-src = { version = "0.10", features = ["accelerate"] }
```

### Metal GPU Backend (optional)

Feature-gated behind `--features metal`. Uses 32x32 tiled compute shaders with threadgroup memory on Apple GPU.

```bash
cargo build --release -p larql-inference --features metal
```

Key optimisations:

- **Buffer cache**: Weight matrices from mmap'd safetensors have stable addresses. Their GPU buffers are created once on first use and reused for all subsequent calls. Only the small input residual and output buffers are allocated per call.
- **Auto-calibration**: On startup, benchmarks CPU vs Metal at representative matrix sizes (attention projections, FFN layers). Finds the lowest FLOP count where Metal with warm cache beats CPU. No magic constants.
- **FLOP-based routing**: Small operations (QK^T at 18K FLOPs) route to CPU with zero overhead. Large operations (FFN gate at 315M FLOPs) route to Metal with cached buffers.
- **Batch dispatch**: `matmul_batch()` encodes multiple matmuls into a single Metal command buffer for parallel GPU execution.

### Hybrid dispatch

The Metal backend is a hybrid — it routes each matmul to the optimal path:

```
Small matmul (< calibrated threshold):  CPU / Accelerate AMX (zero overhead)
Large matmul (> calibrated threshold):  Metal GPU (cached weight buffers)
```

The threshold adapts to hardware via auto-calibration:

| Operation | FLOPs | Route |
|-----------|-------|-------|
| QK^T (per head) | 18K | CPU |
| scores * V (per head) | 18K | CPU |
| Q/K/V/O projection | 79M | Calibrated |
| FFN gate/up | 315M | Metal (cached) |
| Logits | 1.3B | Metal (cached) |

### Usage

```rust
use larql_inference::backend::{default_backend, MatMulBackend};

// Auto-selects best backend (Metal if available, calibrates, falls back to CPU)
let backend = default_backend();
println!("Using: {}", backend.name());

// Single matmul
let c = backend.matmul_transb(&input, &weights);

// Batched (all attention heads in one GPU dispatch)
let results = backend.matmul_batch(&ops);
```

## BLAS-Fused Attention

The attention kernel uses BLAS-accelerated gemv calls inside a fused online-softmax loop. It never allocates the full `[seq, seq]` attention matrix.

### How it works

For each query position `qi` and each head:

1. **BLAS gemv**: `scores[0..=qi] = K[0..=qi] @ Q[qi]` — one `cblas_sgemv` call via ndarray
2. **Scale + softcap**: Apply `1/sqrt(head_dim)` scaling, optional Gemma2 `tanh(score/cap)*cap`
3. **Two-pass softmax**: max, exp, normalise with f64 accumulation
4. **BLAS gemv**: `output = V[0..=qi]^T @ softmax_scores` — one `cblas_sgemv` call

Two BLAS calls per position per head, both hitting AMX. The temporary buffer is `O(seq)` floats per position — no quadratic allocation.

### Supported features

| Feature | Status |
|---------|--------|
| Grouped-Query Attention (GQA) | Supported (any Q/KV ratio) |
| Softcap (Gemma2) | Supported |
| Attention weight capture | Supported (last token) |
| Causal masking | Built-in |
| f64 softmax accumulation | Preserved |
| RoPE | Applied before attention (unchanged) |
| Per-head Q/K norm | Applied before attention (unchanged) |

### Performance

Benchmarked on Apple Silicon (M-series), Gemma-3 4B dimensions:

| Config | BLAS-fused | Materialized ref | Winner |
|--------|-----------|-----------------|--------|
| seq=1, hd=32 | 3 us | 7 us | Fused 2.2x |
| seq=6, hd=32 | 20 us | 13 us | Ref 1.6x |
| seq=6, hd=128 | 29 us | 36 us | Fused 1.3x |
| seq=6, hd=256 | 42 us | 67 us | Fused 1.6x |
| seq=96 | 652 us | 524 us | Ref 1.2x |
| seq=192 | 2,140 us | 1,836 us | Ref 1.2x |

At the actual Gemma-3 head dimension (256), fused is **1.6x faster** than the materialized path.

### Memory

| seq_len | Materialized (10 heads, f32) | Fused (hd=256, f64 acc) | Savings |
|---------|------------------------------|------------------------|---------|
| 6 | 1.4 KB | 12 KB | n/a |
| 128 | 640 KB | 256 KB | 2.5x |
| 512 | 10 MB | 1 MB | 10x |
| 2048 | 160 MB | 4 MB | 40x |

## Quick wins from this session

Two changes that speed up every forward pass:

1. **f32 QK^T**: Removed unnecessary f64 promotion in the QK^T dot product. AMX's `cblas_sgemm` already uses extended-precision accumulators internally, so f32 dispatch gives the same precision at 2x the throughput of f64 `cblas_dgemm`.

2. **BLAS logits**: Replaced a per-token manual dot product loop (`vocab_size` individual dot products) with a single `cblas_sgemm` call for the final `[1, hidden] @ [vocab, hidden]^T` projection.

## Examples

```bash
# Backend demo (shows routing, cache, calibration)
cargo run --release -p larql-inference --example backend_demo --features metal

# Backend benchmark (CPU vs Metal at transformer scale)
cargo run --release -p larql-inference --example bench_backend --features metal

# Fused attention demo (correctness, GQA, softcap, capture)
cargo run --release -p larql-inference --example attention_demo

# Fused attention benchmark (fused vs materialized, scaling)
cargo run --release -p larql-inference --example bench_attention

# Full inference benchmark (needs model weights)
cargo run --release -p larql-inference --example bench_inference

# End-to-end inference demo (needs model weights)
cargo run --release -p larql-inference --example inference_demo
```

## Tests

```bash
# All tests (109 total)
cargo test -p larql-inference

# With Metal GPU tests (+6 Metal-specific tests)
cargo test -p larql-inference --features metal

# Specific test suites
cargo test -p larql-inference --test test_fused_attention   # 18 fused attention tests
cargo test -p larql-inference --test test_backend           # 13 backend integration tests
cargo test -p larql-inference --test test_modules           # 15 module unit tests
cargo test -p larql-inference --test test_trace             # 14 trace store tests
cargo test -p larql-inference --test test_walkers           # 12 walker integration tests
```

### Test coverage

| Area | Tests | What's covered |
|------|-------|----------------|
| Backend (unit) | 21 | Shape, correctness vs f64 reference, identity, zeros, batch, tall/skinny/wide, trait |
| Backend (integration) | 13+6 | Transformer-scale dims, QKV/FFN/logits shapes, factory, Metal vs CPU, batch, fallback |
| Fused attention | 18 | Single token, causal mask, GQA (2x, 5x), softcap, capture, reference agreement, edge cases |
| FFN | 9 | SiLU, GELU, dense shape, activation, highway, multi-position |
| Attention/residual | 10 | RoPE, GQA, RMS norm, layer norm, per-head norm |
| Trace stores | 14 | Write/read, bounds, tiers, additive property |
| Walkers | 12 | Weight/attention walkers, vector extractor, forward pass |
| Utils | 10 | Top-k, rounding, entity sorting, thresholds |

## Codepath coverage

The fused attention and backend changes are exercised by every inference codepath:

| Path | Entry point | Attention | FFN |
|------|-------------|-----------|-----|
| Dense inference | `predict()` | fused GQA | WeightFfn |
| Walk inference | `predict_with_ffn()` | fused GQA | WalkFfn |
| Routed inference | `predict_with_router()` | fused GQA | per-layer |
| Strategy inference | `predict_with_strategy()` | fused GQA | per-layer mode |
| Residual trace | `trace_forward()` | fused GQA | WeightFfn |
| Decomposed trace | `trace_residuals()` | fused GQA (capture) | WeightFfn |
| CachedFfn calibration | `run_attention_public()` | fused GQA | (calibration only) |
| Server /v1/infer | `predict_with_ffn()` | fused GQA | WalkFfn or dense |
| Python trace | `trace_residuals()` | fused GQA (capture) | WeightFfn |
| CLI commands | `predict*()` variants | fused GQA | depends on command |

Sparse FFN, WalkFfn, streaming extraction, and vindex operations do not call attention directly — they only implement FfnBackend. Attention always runs through the same `gqa_attention_with_weights()` path.

## Walk Boundary Sweep

The [walk boundary sweep](walk-boundary-sweep.md) proved that vindex FFN walk produces identical top-1 predictions to the all-dense forward pass at **every layer boundary from L0 to L34**. 5/5 correct, 82.63% average probability, zero divergence at every boundary.

The walk FFN with mmap'd down vectors is now **faster than dense** (517ms vs 535ms). See the [FFN graph layer](ffn-graph-layer.md) for the full architecture, optimization progression (21s → 517ms), and data path details. See the [boundary sweep](walk-boundary-sweep.md) for correctness proof.

### Profiled bottleneck breakdown

```
Component              Time      % of 541ms
─────────────────────────────────────────────
Logits projection      221ms     41%          ← #1 bottleneck
FFN × 34 layers        206ms     38%          ← solved by walk
Attention × 34 layers   84ms     16%
Framework overhead       7ms      1%
```

Memory: no leaks, mmap-managed. Walk only needs ~5.5GB of model weights (attention + embeddings + norms), not the full 16.6GB. Use `InferenceModel::load_walk_only()` to drop FFN weights (saves 10.7GB).

### Server

Walk inference is served over HTTP via `larql-server`:

```bash
cargo run --release -p larql-server -- path/to/vindex --port 8080

# Walk (faster than dense, mmap FFN)
curl -X POST http://localhost:8080/v1/infer \
  -d '{"prompt": "The capital of France is", "top": 5, "mode": "walk"}'

# Compare (walk vs dense side-by-side, identical predictions)
curl -X POST http://localhost:8080/v1/infer \
  -d '{"prompt": "The capital of France is", "top": 3, "mode": "compare"}'
```

The server loads mmap'd feature-major vectors at startup. Walk inference uses zero-copy down projection from `down_features.bin`. See [FFN graph layer](ffn-graph-layer.md) for architecture details.
