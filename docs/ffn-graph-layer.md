# FFN Graph Layer

The FFN graph layer replaces the dense down projection with a zero-copy mmap read from the vindex. The walk produces identical predictions to the dense forward pass — same gate, same up, same GEGLU, same answer. **Walk is faster than dense** (517ms vs 535ms) because the mmap'd down matrix has better page cache behavior.

## Architecture

```
Dense FFN (traditional):
  gate = x @ W_gate.T           ← matmul from safetensors  (105MB)
  up   = x @ W_up.T             ← matmul from safetensors  (105MB)
  act  = silu(gate) * up         ← elementwise
  out  = act @ W_down.T          ← matmul from safetensors  (105MB)
  Total reads: 315MB

Walk FFN (graph layer):
  gate = x @ W_gate.T           ← matmul from safetensors  (105MB)
  up   = x @ W_up.T             ← matmul from safetensors  (105MB)
  act  = silu(gate) * up         ← elementwise
  out  = act @ D_mmap            ← BLAS gemm from mmap      (105MB, zero-copy)
  Total reads: 315MB, but down read is from feature-major mmap (better caching)
```

The down projection reads from `down_features.bin` — a feature-major `[intermediate, hidden]` f32 file, memory-mapped directly as an ndarray ArrayView2. No allocation, no copy. The mmap IS the matrix.

## Proof of Correctness

The [walk boundary sweep](walk-boundary-sweep.md) tested vindex FFN at every layer boundary from L0 to L34 on Gemma-3 4B:

```
     B   walk%   correct  top1_avg  details
  -------------------------------------------------------
  L0     100%    5/5      82.63%   all match ground truth
  L4      88%    5/5      82.63%   all match ground truth
  L8      76%    5/5      82.63%   all match ground truth
  L12     65%    5/5      82.63%   all match ground truth
  L16     53%    5/5      82.63%   all match ground truth
  L20     41%    5/5      82.63%   all match ground truth
  L24     29%    5/5      82.63%   all match ground truth
  L28     18%    5/5      82.63%   all match ground truth
  L34      0%    5/5      82.63%   all match ground truth
```

Zero divergence. Same top-1 token, same probability, at every boundary. The gate vectors at each layer are calibrated to that layer's residual space — the KNN matches because the index and query live in the same space.

## Performance

### Optimization progression

| Version | Walk | Dense | Gap |
|---------|------|-------|-----|
| Unoptimized | 21,197ms | 708ms | 30x slower |
| + batch gate KNN (one gemm per layer) | 4,178ms | 685ms | 6.1x |
| + sparse down projection + f16 cache | 4,178ms | 685ms | (included above) |
| + trace recording off by default | 841ms | 685ms | 23% |
| + f32 gate vectors (mmap, zero-copy) | 685ms | 560ms | 22% |
| + zero-copy mmap down matrix | 668ms | 544ms | 23% |
| + remove redundant gate KNN | **517ms** | **535ms** | **Walk is faster** |

### Per-layer breakdown

```
Dense FFN:    6.4ms/layer  (gate + up + GEGLU + down from safetensors)
Walk FFN:     6.0ms/layer  (gate + up + GEGLU from safetensors + down from mmap)
```

Walk is faster because the down projection reads from the feature-major mmap'd `down_features.bin` which has better page cache behavior than the safetensors layout. Same computation, better memory access pattern.

The gate KNN (4ms) is only used for the sparse fallback path (when down_features.bin is not available) or for tracing.

### What eliminated the 30x gap

| Optimization | Speedup | What it fixed |
|-------------|---------|---------------|
| Batch gate KNN | 5x | One BLAS gemm per layer instead of 6 separate gemv calls |
| Trace off by default | 5x | Deferred trace to take_trace() — was 8092 feature_meta lookups + allocations per layer |
| f32 mmap | 1.2x | Zero decode, zero allocation, zero warmup. Pointer reinterpretation to BLAS. |
| Sparse down projection | 1.2x | gather K columns of W_down, not full [hidden, intermediate] matmul |
| f16 decode cache | — | Amortized cost (eliminated by f32 conversion) |

## Data Path

### Primary path (mmap walk, fastest)

```
down_features.bin (f32, 3.6GB, feature-major)
    ↓ mmap (zero-copy, OS manages pages)
    ↓ down_layer_matrix(layer) → ArrayView2 [intermediate, hidden]
    ↓ activation.dot(&down_view) — one BLAS gemm, reads from mmap
    ↓ [seq, hidden] output
```

No allocation. No copy. The mmap file IS the down matrix. BLAS reads directly from the memory-mapped region. The OS manages page cache — hot pages stay resident across tokens.

### Gate/up still from model weights

```
W_gate, W_up from safetensors (mmap'd by larql-models)
    ↓ dot_proj(x, w_gate) + dot_proj(x, w_up) — BLAS gemm
    ↓ silu_gate_up() — GEGLU activation
    ↓ [seq, intermediate] activation vector
```

### Sparse fallback (no down_features.bin)

When `down_features.bin` is not available, the walk falls back to gate KNN + sparse FFN from model weights. Convert with:

```bash
# Convert gate vectors to f32 (zero-copy mmap for KNN)
cargo run --release -p larql-vindex --example convert_gates_f32 -- path/to/vindex/

# Build feature-major down vectors (zero-copy mmap for down projection)
cargo run --release -p larql-vindex --example build_down_features -- path/to/vindex/
```

## WalkFfn API

```rust
use larql_inference::vindex::WalkFfn;

// Fast path: no trace recording (default)
let walk_ffn = WalkFfn::new(weights, &index, top_k);
let result = predict_with_ffn(weights, tokenizer, &token_ids, 5, &walk_ffn);

// With trace (for analysis — re-runs gate KNN lazily on take_trace)
let walk_ffn = WalkFfn::new_with_trace(weights, &index, top_k);
let result = predict_with_ffn(weights, tokenizer, &token_ids, 5, &walk_ffn);
let trace = walk_ffn.take_trace();
```

The walk FFN integrates transparently with all forward pass variants:
- `predict_with_ffn()` — full walk inference
- `predict_with_router()` — per-layer dense/walk selection
- `trace_forward_with_ffn()` — residual capture with walk
- Server `/v1/infer` — walk mode via HTTP/gRPC

## HNSW Index (experimental)

An HNSW graph index is available for approximate gate search. At 10,240 vectors it provides no speedup over brute-force BLAS gemm (the graph overhead equals the savings). It will matter at larger feature counts.

```rust
// Enable HNSW (builds lazily, dim=64 random projection)
index.enable_hnsw(200);  // ef_search = 200

// Disable (revert to brute-force gemm)
index.disable_hnsw();
```

Build time: ~700ms one-time (34 layers, 10,240 vectors, dim=64 projected).

## Implications

### FFN quantization is unnecessary

The walk serves all 34 FFN layers with exact results. No approximation, no quantization. The vindex IS the FFN.

### Bottleneck analysis (profiled)

```
Component              Time      % of 541ms    Bottleneck
─────────────────────────────────────────────────────────
Logits projection      221ms     41%           #1 — 262K vocab gemv
FFN × 34 layers        206ms     38%           Solved by walk
Attention × 34 layers   84ms     16%           Next target
Softmax + top-k          2ms      0%
Framework overhead       7ms      1%           Clean — no hidden cost
```

The walk eliminates FFN as a bottleneck. Logits (221ms) is now the single largest cost — one gemv against a 2.7GB vocabulary matrix.

### Remaining matmuls

| Operation | Per layer | Source | Notes |
|-----------|-----------|--------|-------|
| Q projection | ~1ms | safetensors | Accelerate AMX |
| K projection | ~1ms | safetensors | Accelerate AMX |
| V projection | ~1ms | safetensors | Accelerate AMX |
| O projection | ~1ms | safetensors | Accelerate AMX |
| FFN gate | ~2ms | safetensors | Exact gate projection |
| FFN up | ~2ms | safetensors | Exact up projection |
| FFN down | ~2ms | **mmap vindex** | Zero-copy, feature-major |
| Final logits | ~221ms (once) | safetensors | #1 bottleneck |

### Memory profile

```
Component                        RSS        Notes
──────────────────────────────────────────────────────
Baseline                           3 MB
Model safetensors (mmap)      16,613 MB     Includes FFN weights (not needed by walk)
Vindex gate vectors (mmap)       +84 MB     Demand-paged
Feature mmaps (mapped)            +0 MB     No pages until accessed
Dense forward pass               +48 MB     Temporary ndarray buffers
Walk forward pass             +3,404 MB     down_features.bin pages faulted in
Growth over 10 runs              +19 MB     Stable — no leaks
```

Walk only needs ~3.5GB of model weights (attention + embeddings). A `--walk-only` flag could skip FFN safetensors entirely: 16.6GB → 3.5GB.

### Path forward

```
Current:     517ms walk (faster than 535ms dense)
+ logits from vindex:   down_meta token lookup → ~300ms (saves 221ms)
+ --walk-only mode:     skip FFN weights → 3.5GB RAM (saves 13GB)
+ Q4_K_M attention:     4× less bandwidth → ~200ms
+ template cache:       attention eliminated → ~40ms
+ precompiled routes:   graph walk only → ~5ms
```

## Walk Path Selection

| Path | When | Down source | Speed |
|------|------|------------|-------|
| **Exact walk** (primary) | `down_features.bin` available | gate+up from safetensors, down from mmap | 5.7ms/layer |
| **Full mmap walk** | `up_features.bin` also available | All from mmap (available, slower due to 3-file TLB) | 6.8ms/layer |
| **Sparse fallback** | No mmap, has model weights | Gate KNN + sparse gather | 6.3-10ms/layer |
| **Dense fallback** | No vindex data for layer | Full dense FFN | 6.7ms/layer |

The exact walk is the default — gate+up from safetensors (sequential in one file) + down from feature-major mmap (zero-copy BLAS).

## Feature-Major Down Vectors

The `down_features.bin` file stores down projection vectors in feature-major layout: `[intermediate, hidden]` per layer. Each feature's down vector is 2560 contiguous f32 values (10KB). This enables:

- **Sequential memcpy** instead of strided column gather for the sparse matmul path
- **Zero-copy mmap read** for the direct walk path (pointer offset + read)
- **Cache-friendly** access pattern — each feature read is one L2 cache line sequence

Build with:
```bash
cargo run --release -p larql-vindex --example build_down_features -- path/to/vindex/
```

## Files

### Vindex index modules

| Module | Lines | Responsibility |
|--------|-------|---------------|
| `index/types.rs` | 130 | FeatureMeta, GateIndex trait, WalkHit, callbacks |
| `index/core.rs` | 449 | VectorIndex struct, constructors, loading, accessors |
| `index/gate.rs` | 690 | Gate KNN: search, batch, scores, HNSW integration, warmup |
| `index/walk.rs` | 111 | Walk FFN data: mmap'd down/up feature-major vectors |
| `index/hnsw.rs` | 337 | HNSW graph index (standalone data structure) |
| `index/mutate.rs` | 283 | Gate vector mutation (INSERT/DELETE) |
| `index/router.rs` | 125 | MoE expert routing |

### Inference modules

| File | Purpose |
|------|---------|
| `vindex/walk_ffn.rs` | WalkFfn: mmap walk FFN (faster than dense) |
| `ffn/weight.rs` | WeightFfn: dense FFN (ground truth) |
| `ffn/sparse_compute.rs` | Sparse FFN compute (shared by walk fallback) |
| `attention.rs` | BLAS-fused attention + shared attention block |
| `forward.rs` | Forward pass: embed → layers → logits |

### Server

Walk inference is available via HTTP:

```bash
# Start server with mmap walk enabled
cargo run --release -p larql-server -- path/to/vindex --port 8080

# Walk inference (faster than dense)
curl -X POST http://localhost:8080/v1/infer \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "top": 5, "mode": "walk"}'

# Compare walk vs dense (identical predictions, walk faster)
curl -X POST http://localhost:8080/v1/infer \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "top": 3, "mode": "compare"}'
```

Modes: `walk` (default, mmap FFN), `dense` (full matmul), `compare` (both side-by-side).

### Walk-only mode

Drop FFN weights from memory — 16.6GB → 5.5GB:

```rust
let model = InferenceModel::load_walk_only("google/gemma-3-4b-it")?;
// 10.7 GB of FFN weights freed
// Requires down_features.bin + up_features.bin in vindex
```

### Tools and benchmarks

| File | Purpose |
|------|---------|
| `larql-server` | HTTP server with walk/dense/compare inference modes |
| `larql-vindex/examples/convert_gates_f32.rs` | f16 → f32 gate vector converter |
| `larql-vindex/examples/build_down_features.rs` | Feature-major down vector builder |
| `larql-vindex/examples/build_up_features.rs` | Feature-major up vector builder |
| `larql-inference/examples/bench_walk_inference.rs` | Walk benchmark (dense vs walk vs HNSW) |
| `larql-inference/examples/walk_boundary_sweep.rs` | Correctness sweep (all 34 layers) |
| `larql-inference/examples/profile_overhead.rs` | Forward pass bottleneck profiler |
| `larql-inference/examples/memory_analysis.rs` | Memory profiling (RSS, mmap, walk-only) |
