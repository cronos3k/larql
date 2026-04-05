# larql-vindex

The queryable model format. Decompile, browse, edit, and recompile neural networks.

## What is a Vindex?

A vindex (vector index) is a directory containing a transformer model's weights reorganised for queryability. The model IS the database — each weight matrix is stored once in its optimal format.

```rust
use larql_vindex::*;

// Load (readonly base)
let index = VectorIndex::load_vindex(&path, &mut SilentLoadCallbacks)?;
let mut patched = PatchedVindex::new(index);

// Query — which features fire for "France"?
let hits = patched.gate_knn(layer, &query, 10);  // 2.7ms/layer at full dim

// Walk — multi-layer feature scan
let trace = patched.walk(&query, &layers, 10);

// Mutate via patch overlay (base files never modified)
patched.insert_feature(layer, feature, gate_vec, meta);

// Apply a saved patch
let patch = VindexPatch::load("medical.vlp")?;
patched.apply_patch(patch);

// Bake patches into a new clean VectorIndex
let baked = patched.bake_down();
baked.save_vindex(&output_path, &mut config)?;
```

## The Headline

A 1T model in 10.9 GB on a laptop.

```
Model          Full Inference RAM    Vindex Infer RAM    Ratio
Gemma 3 4B              7 GB              1.3 GB          5x
Llama 3 8B             15 GB              2.2 GB          7x
Llama 3 70B           130 GB              4.9 GB         27x
Llama 3 405B          754 GB              8.6 GB         88x
DeepSeek V3          1250 GB             10.9 GB        115x
Kimi-K2              1863 GB             10.9 GB        171x
```

Vindex inference uses mmap: only 1 layer of gate vectors + 1 layer of attention
weights are resident at a time. The rest stays on disk until touched.

## Features

- **Extract** from safetensors, GGUF, or MLX models (streaming — no full model load)
- **Gate KNN** via BLAS matmul, Q4 matvec (CPU/Metal/CUDA), or HNSW approximate search
- **Walk** across all layers with down-meta annotation
- **Readonly base** — base vindex files are never modified after extraction
- **Patch overlay** — all mutations go through PatchedVindex (INSERT/DELETE/UPDATE)
- **Patches** — stackable, reversible knowledge diffs (.vlp files)
- **Vindexfile** — declarative model builds (FROM + PATCH + INSERT, like Dockerfile)
- **HuggingFace Hub** — download and publish vindexes (`hf://user/repo` URI scheme)
- **Split weight files** — gate, up, down, attn, norms, lm_head (no duplication)
- **Zero-copy mmap** — gate vectors sliced directly from disk, no heap allocation
- **Binary down_meta** — compact binary format (no JSONL)
- **f16 storage** — halves file sizes with negligible accuracy loss
- **MoE support** — Mixtral, DeepSeek (experts as contiguous features)
- **Layer bands** — per-family boundaries (Gemma, Llama, Qwen, etc.)
- **Checksums** — SHA256 integrity verification for all binary files
- **Provenance** — model source, timestamp, version tracking
- **LM head KNN** — top-K token lookup via single BLAS gemv against output projection

## Crate Structure

```
larql-vindex/src/
├── lib.rs                      Crate root + re-exports
├── error.rs                    VindexError
├── describe.rs                 DescribeEdge, LabelSource
├── mmap_util.rs                madvise-optimized mmap helper
│
├── config/                     Configuration types
│   ├── types.rs                VindexConfig, ExtractLevel, LayerBands, MoeConfig
│   └── dtype.rs                StorageDtype (f32/f16), encode/decode
│
├── index/                      In-memory KNN engine (zero-copy mmap)
│   ├── core.rs                 VectorIndex construction + loading
│   ├── types.rs                FeatureMeta, GateIndex trait, WalkHit, WalkTrace
│   ├── gate.rs                 Gate KNN (brute-force, batched, HNSW, expert-scoped)
│   ├── hnsw.rs                 HNSW graph index (random projection, exact rescoring)
│   ├── walk.rs                 Feature-major down/up vectors, interleaved, Q4, lm_head
│   ├── mutate.rs               set/delete features, save to disk
│   └── router.rs               MoE expert router
│
├── format/                     Vindex file I/O
│   ├── load.rs                 load_vindex, load_embeddings, load_tokenizer
│   ├── down_meta.rs            Binary down_meta read/write
│   ├── weights.rs              Split weight files (attn, up, down, norms, lm_head)
│   ├── checksums.rs            SHA256 computation + verification
│   ├── huggingface.rs          HuggingFace Hub download/publish
│   └── quant/mod.rs            Re-exports from larql_models::quant
│
├── extract/                    Build pipeline (model → vindex)
│   ├── build.rs                build_vindex (full extraction + clustering)
│   ├── streaming.rs            Streaming extraction (mmap, no full model load)
│   ├── callbacks.rs            IndexBuildCallbacks trait
│   └── build_from_vectors.rs   Build from pre-extracted NDJSON
│
├── patch/                      Patch system
│   └── core.rs                 VindexPatch, PatchOp, PatchedVindex
│
├── clustering/                 Relation discovery
│   ├── kmeans.rs               k-means clustering (BLAS via larql-compute)
│   ├── labeling.rs             Pattern detection, TF-IDF labels
│   ├── categories.rs           Entity category word lists
│   ├── pair_matching.rs        Wikidata/WordNet output matching
│   └── probe.rs                Probe label loading
│
└── vindexfile/                 Declarative model builds
    ├── mod.rs                  Build executor (FROM + PATCH + INSERT → bake_down)
    └── parser.rs               Vindexfile parser (FROM, PATCH, INSERT, DELETE, etc.)
```

All matrix operations go through `larql-compute` (BLAS on CPU, Metal GPU planned for gate KNN).

## Compute Integration

| Module | Operation | Backend |
|--------|-----------|---------|
| gate.rs | Gate KNN f32 (matmul_transb) | CPU BLAS |
| gate.rs | Gate KNN Q4 (q4_matvec) | Any ComputeBackend |
| gate.rs | Gate walk (gemv) | CPU BLAS |
| gate.rs | Batch gate scores (matmul_transb) | CPU BLAS |
| hnsw.rs | Random projection (matmul) | CPU BLAS |
| hnsw.rs | Dot product (graph traversal) | CPU BLAS |
| walk.rs | LM head KNN (matmul_transb) | CPU BLAS |
| kmeans.rs | Similarity matrix (matmul_transb) | CPU BLAS |
| router.rs | MoE routing (matmul) | CPU BLAS |

## Supported Architectures

| Family | Models | FFN Type |
|--------|--------|----------|
| Gemma | Gemma 2/3 (2B-27B) | Gated (GeGLU) |
| Llama | Llama 2/3 (7B-405B) | Gated (SiLU) |
| Mistral | Mistral 7B | Gated (SiLU) |
| Mixtral | Mixtral 8x7B/8x22B | MoE (8 experts) |
| Qwen | Qwen 2/2.5 | Gated (SiLU) |
| Phi | Phi 2/3 | Gated |
| DeepSeek | DeepSeek V2/V3 | MoE (shared + routed) |
| GPT-2 | GPT-2 | Dense (GELU) |

## File Layout

```
model.vindex/
├── gate_vectors.bin        W_gate per layer (f32/f16 KNN index)
├── gate_vectors_q4.bin     W_gate Q4_0 (7x smaller, for Q4 KNN)
├── embeddings.bin          W_embed matrix
├── down_meta.bin           Per-feature output metadata (binary)
├── attn_weights.bin        Q, K, V, O per layer
├── up_weights.bin          W_up per layer
├── down_weights.bin        W_down per layer
├── norms.bin               LayerNorm parameters
├── lm_head.bin             Output projection
├── interleaved.bin         gate|up|down packed per layer (optional)
├── interleaved_q4.bin      Q4_0 quantized version (optional, 7x smaller)
├── index.json              Config, layer bands, provenance, checksums
├── tokenizer.json          Tokenizer
├── relation_clusters.json  Discovered relation types
├── feature_labels.json     Probe-confirmed labels
└── weight_manifest.json    Weight file → offset mapping
```

## Extract Levels

| Level | Size (f16) | Enables |
|-------|-----------|---------|
| Browse | ~3 GB | DESCRIBE, WALK, SELECT |
| Inference | ~6 GB | + INFER |
| All | ~8.5 GB | + COMPILE |

## Testing

```bash
cargo test -p larql-vindex                                    # 84 tests
cargo run -p larql-vindex --example vindex_demo               # Feature showcase
cargo run -p larql-vindex --example vindex_bench --release     # Core benchmarks
cargo run -p larql-vindex --example bench_scaling --release    # Production dimensions
```

## Benchmarks

### Core operations (synthetic, reduced dimensions)

| Operation | Latency |
|---|---|
| Gate KNN (per layer, 1024×256) | 0.029ms |
| Walk (8 layers) | 0.23ms |
| Feature lookup | <1ns |
| Save gates (8 MB) | 1.4ms |
| Load vindex (mmap) | 1.3ms |
| Mutate (meta + gate) | 877ns |
| Checksum (SHA256) | 19ms |
| MoE 8x scaling | 16x (sub-linear) |

### Production dimensions (M3 Max, synthetic data)

| Model | Features | Hidden | f32 BLAS | Q4 CPU | Q4 Metal | Speedup | Walk 14L |
|---|---|---|---|---|---|---|---|
| Gemma 3 4B | 10,240 | 2,560 | 2.7ms | 0.96ms | **0.50ms** | 5x | 7.0ms |
| Llama 3 8B | 14,336 | 4,096 | 15.7ms | 2.1ms | **0.95ms** | 17x | 15.2ms |
| Llama 3 70B | 28,672 | 8,192 | 98.3ms | 8.2ms | **1.31ms** | **75x** | 63.1ms |

Vindex provides Q4 gate data. Compute crate scores it. Same interface, any backend.

### HNSW vs brute-force (dim=2560)

| Features | Brute | HNSW | Winner |
|---|---|---|---|
| 1,024 | 0.18ms | 0.14ms | HNSW |
| 4,096 | 2.3ms | 1.9ms | HNSW |
| 10,240 | 2.6ms | 1.7ms | HNSW |
| 28,672 | 18.8ms | 15.2ms | HNSW |

### Memory (mmap, 34L × 4096 × 2560)

| Metric | Value |
|---|---|
| Cold KNN (first access) | 0.39ms |
| Warm KNN (paged) | 0.37ms |
| Page fault overhead | 0.02ms |
| Zero-copy mmap | true (0 bytes heap) |

## Design Principles

1. **Readonly base** — binary files on disk are never modified after extraction
2. **Patch overlay** — all mutations via in-memory PatchedVindex
3. **Zero-copy mmap** — gate vectors are sliced from the file, not loaded to heap
4. **One file per matrix type** — gate, attn, up, down stored separately
5. **Streaming extraction** — processes one layer at a time (~2 GB peak for 120B models)
6. **All compute through larql-compute** — BLAS dispatch, no raw ndarray .dot() calls

## License

Apache-2.0
