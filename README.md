# LARQL

The model IS the database. Query neural network weights like a graph database. No GPU required.

LARQL decompiles transformer models into a queryable format called a **vindex** (vector index), then provides **LQL** (Lazarus Query Language) to browse, edit, and recompile the model's knowledge.

```sql
larql> USE "gemma3-4b.vindex";
Using: gemma3-4b.vindex (34 layers, 348.2K features, relations: 512 types)

larql> DESCRIBE "France";
France
  Edges (L14-27):
    capital     → Paris              1436.9  L27  (probe)
    language    → French               35.2  L24  (probe)
    continent   → Europe               14.4  L25  (probe)
    borders     → Spain                13.3  L18  (probe)

larql> INSERT INTO EDGES (entity, relation, target)
   ...   VALUES ("John Coyle", "lives-in", "Colchester");
Inserted 1 edge. Feature F8821@L26 allocated.

larql> INFER "The capital of France is" TOP 3;
  1. Paris                (97.91%)
  2. the                  (0.42%)
  3. a                    (0.31%)
```

## Quick Start

```bash
# Build
cargo build --release

# Extract a model into a vindex (browse-only, ~3 GB at f16)
larql extract-index google/gemma-3-4b-it -o gemma3-4b.vindex --f16

# Extract with inference weights (~6 GB at f16)
larql extract-index google/gemma-3-4b-it -o gemma3-4b.vindex --level inference --f16

# Or convert from GGUF
larql convert gguf-to-vindex model.gguf -o model.vindex --f16

# Or download from HuggingFace
larql hf download chrishayuk/gemma-3-4b-it-vindex

# Start the REPL
larql repl

# Use a local vindex or HuggingFace vindex directly
larql lql 'USE "gemma3-4b.vindex"; DESCRIBE "France";'
larql lql 'USE "hf://chrishayuk/gemma-3-4b-it-vindex"; DESCRIBE "France";'
```

## What is a Vindex?

A vindex is a directory containing a model's weights reorganised for queryability. Gate vectors become a KNN index. Embeddings become token lookups. Down projections become edge labels. The model IS the database.

```
gemma3-4b.vindex/
  gate_vectors.bin         # W_gate rows (KNN index, 3.3 GB)
  embeddings.bin           # W_embed matrix (token lookup, 2.5 GB)
  down_meta.bin            # Per-feature output metadata (binary)
  index.json               # Config, layer bands, provenance
  tokenizer.json           # Tokenizer
  relation_clusters.json   # Discovered relation types
  feature_labels.json      # Probe-confirmed labels
```

Three extraction levels:

| Level | CLI Flag | LQL Syntax | Size (f16) | Enables |
|-------|----------|-----------|-----------|---------|
| Browse | `--level browse` (default) | `EXTRACT MODEL ... INTO ...` | ~3 GB | DESCRIBE, WALK, SELECT |
| Inference | `--level inference` | `... WITH INFERENCE` | ~6 GB | + INFER |
| All | `--level all` | `... WITH ALL` | ~10 GB | + COMPILE |

Add `--f16` to halve file sizes with negligible accuracy loss.

## Architecture

Eight crates. Clean dependency chain.

```
larql-models      Model config, architecture traits, weight loading, quant/dequant
    ↓
larql-vindex      Vindex lifecycle: extract, load, query, mutate, patch, save
    ↓
larql-core        Graph algorithms, merge, diff
larql-inference   Forward pass, BLAS-fused attention, Metal GPU, WalkFfn
    ↓
larql-lql         LQL parser, executor, REPL, USE REMOTE client
    ↓
larql-server      HTTP/gRPC server: serve vindexes over the network
larql-cli         CLI commands (extract-index, build, serve, repl, convert, hf, verify)
```

### larql-vindex

Owns the vindex lifecycle. Streaming extraction (mmap, no full model load), KNN via BLAS matmul,
zero-copy mmap loading, split weight files, readonly base with patch overlay, clustering, f16 storage.

```rust
// Load (readonly base)
let index = VectorIndex::load_vindex(&path, &mut cb)?;
let patched = PatchedVindex::new(index);

// Query
let hits = patched.gate_knn(layer, &query, 10);  // 0.008ms/layer
let trace = patched.walk(&query, &layers, 10);    // multi-layer scan

// Mutate (patch overlay — base files never modified)
patched.insert_feature(layer, feature, gate_vec, meta);
patched.apply_patch(VindexPatch::load("edits.vlp")?);
```

### larql-lql

LQL parser and executor. 20+ statement types across 5 categories:

- **Lifecycle**: EXTRACT, COMPILE, DIFF, USE
- **Browse**: WALK, DESCRIBE, SELECT, EXPLAIN WALK
- **Inference**: INFER, EXPLAIN INFER
- **Mutation**: INSERT, DELETE, UPDATE, MERGE
- **Patches**: BEGIN PATCH, SAVE PATCH, APPLY PATCH, SHOW PATCHES, REMOVE PATCH
- **Introspection**: SHOW RELATIONS/LAYERS/FEATURES/MODELS/PATCHES, STATS

## LQL Reference

See [docs/lql-spec.md](docs/lql-spec.md) for the full language specification and [docs/lql-guide.md](docs/lql-guide.md) for a quick start guide.

### Key Statements

```sql
-- Decompile a model
EXTRACT MODEL "google/gemma-3-4b-it" INTO "gemma3-4b.vindex" WITH ALL;

-- Browse knowledge (no GPU needed)
USE "gemma3-4b.vindex";
DESCRIBE "France";                      -- verbose by default: [relation] labels, also-tokens
DESCRIBE "Einstein" ALL LAYERS;
DESCRIBE "France" BRIEF;                -- compact view
WALK "The capital of France is" TOP 10;

-- Run inference (needs model weights in vindex)
INFER "The capital of France is" TOP 5 COMPARE;

-- Trace the residual stream (decomposed forward pass)
TRACE "The capital of France is" ANSWER "Paris";
TRACE "The capital of France is" DECOMPOSE LAYERS 22-27;
TRACE "The capital of France is" SAVE "france.trace";

-- Edit knowledge (auto-patch: base files never modified)
INSERT INTO EDGES (entity, relation, target)
    VALUES ("John Coyle", "lives-in", "Colchester");
-- "Auto-patch started (use SAVE PATCH to persist)"

-- Patches (lightweight, shareable knowledge diffs)
BEGIN PATCH "medical.vlp";
INSERT INTO EDGES (entity, relation, target)
    VALUES ("aspirin", "treats", "headache");
SAVE PATCH;
APPLY PATCH "medical.vlp";

-- Recompile to standard model format
COMPILE CURRENT INTO MODEL "edited/" FORMAT safetensors;
```

## Patches

Patches are lightweight JSON files (.vlp) that capture INSERT/DELETE/UPDATE operations. They overlay an immutable base vindex without modifying it.

```sql
-- Create a patch
BEGIN PATCH "medical-knowledge.vlp";
INSERT INTO EDGES (entity, relation, target)
    VALUES ("aspirin", "side_effect", "bleeding");
SAVE PATCH;

-- Apply patches (stackable, reversible)
APPLY PATCH "medical-knowledge.vlp";
APPLY PATCH "fix-hallucinations.vlp";
SHOW PATCHES;
REMOVE PATCH "fix-hallucinations.vlp";

-- Extract diff between two vindexes as a patch
DIFF "base.vindex" "edited.vindex" INTO PATCH "changes.vlp";
```

A single fact is ~10 KB. A 1,000-fact domain patch is ~10 MB. Compared to the full model at 8 GB, that's 1/800th the size. No fine-tuning, no GPU, no retraining.

The base vindex is always readonly. INSERT/DELETE/UPDATE automatically create a patch overlay. Edits are never written to base files.

## Vindexfile

Declarative model builds. Like a Dockerfile for model knowledge.

```dockerfile
# Vindexfile
FROM hf://chrishayuk/gemma-3-4b-it-vindex
PATCH hf://medical-ai/drug-interactions@2.1.0
PATCH ./patches/company-facts.vlp
INSERT ("Acme Corp", "headquarters", "London")
LABELS hf://chrishayuk/gemma-3-4b-it-labels@latest
EXPOSE browse inference
```

```bash
larql build .                          # build from Vindexfile
larql build . --stage prod             # named stage
larql build . --output custom.vindex   # custom output path
```

## Model Support

Input formats: **safetensors** (HuggingFace), **GGUF** (llama.cpp, dequantized to f32), **MLX** (Apple, same safetensors layout).

| Family | Models | FFN Type |
|--------|--------|----------|
| Gemma | Gemma 2/3 (2B-27B) | Gated (GeGLU) |
| Llama | Llama 2/3 (7B-405B) | Gated (SiLU) |
| Mistral | Mistral 7B | Gated (SiLU) |
| Mixtral | Mixtral 8x7B, 8x22B | MoE (8 experts) |
| Qwen | Qwen 2/2.5 (0.5B-72B) | Gated (SiLU) |
| Phi | Phi 2/3 (2.7B-14B) | Gated |
| DeepSeek | DeepSeek V2/V3 | MoE (shared + routed) |
| GPT-OSS | GPT-OSS-120B | MoE (128 experts, MXFP4) |
| GPT-2 | GPT-2 (117M-1.5B) | Dense (GELU) |

Dense and full-precision MoE models support all operations (DESCRIBE, WALK, INFER). MXFP4-quantized MoE models (GPT-OSS) can be extracted and served but DESCRIBE/WALK produce noisy results due to 4-bit weight precision — use INFER for accurate knowledge queries. See [operations spec](docs/vindex-operations-spec.md) for details.

## Benchmarks

### Vindex Operations

| Operation | Latency |
|---|---|
| Gate KNN (per layer) | 0.008ms |
| Walk (34 layers) | 0.3ms |
| Feature lookup | <1ns |
| Save gates (8 MB) | 1.1ms |
| Load vindex | 8ms |
| Mutate (meta + gate) | 617ns |

### Inference Engine (Gemma 3 4B, Apple Silicon)

| Operation | Latency |
|---|---|
| Walk prediction (no attention) | 33ms |
| INFER walk (with attention, mmap FFN) | 517ms |
| INFER dense (with attention, all matmul) | 535ms |
| DESCRIBE (knowledge browse) | 33ms |

| Component | Time | % of total |
|---|---|---|
| Logits (262K vocab gemv) | 221ms | 41% |
| FFN × 34 layers (walk) | 194ms | 36% |
| Attention × 34 layers | 84ms | 16% |
| Walk FFN per layer (mmap down) | 5.7ms | — |
| Dense FFN per layer | 6.7ms | — |
| BLAS-fused attention per head | 42us | — |

Walk is **faster than dense** (517ms vs 535ms). FFN down projection reads from mmap'd vindex (zero-copy BLAS). Walk only needs ~3.5GB of model weights (attention + embeddings), not 16.6GB. No quantization. See [docs/ffn-graph-layer.md](docs/ffn-graph-layer.md) for architecture and [docs/inference-engine.md](docs/inference-engine.md) for engine details.

## Residual Stream Trace

Capture the complete record of inference — every layer, every contribution, queryable.

```sql
-- LQL: answer trajectory through all layers
larql> TRACE "The capital of France is" ANSWER "Paris";
  Layer   Rank     Prob      Attn       FFN      Who
    L22     50    0.002     +22.2     +34.4   BOTH ↑
    L23     10    0.024     -16.9     +55.9    FFN ↑
    L24      1    0.714    +105.7     +24.4   BOTH ↑  ← phase transition
    L25      1    0.997      +4.3     +94.4    FFN ↑
    L26      1    0.999     +83.1     +18.7   BOTH ↑

-- Attn vs FFN decomposition at the phase transition
larql> TRACE "The capital of France is" DECOMPOSE LAYERS 22-27;

-- Persist for later analysis
larql> TRACE "The capital of France is" SAVE "france.trace";
```

```python
# Python: same trace, programmatic access
import larql

wm = larql.WalkModel("gemma3-4b.vindex")
t = wm.trace("The capital of France is")
t.answer_trajectory("Paris")   # rank, prob, attn/ffn logits per layer
t.top_k(24)                    # [('Paris', 0.714), ...]
t.save("trace.bin")            # mmap'd store
```

### Tiered Context (infinite context without KV cache)

| Storage | Per window | 370K tokens | vs KV cache |
|---|---|---|---|
| Boundary residual | 10 KB | 18.9 MB | 3,100x |
| Tier 4 int8 (bit-perfect) | 58 KB | 110 MB | 511x |
| KV cache | ~30 MB | 56,000 MB | 1x |

```python
from larql._native import BoundaryWriter, BoundaryStore

# Write boundary residuals — one per 200-token window
writer = BoundaryWriter("context.bndx", hidden_size=2560, window_size=200)
writer.append(token_offset=0, window_tokens=200, residual=boundary_vec)
writer.finish()

# Mmap'd read — OS pages on demand, RSS ≈ one boundary
store = BoundaryStore("context.bndx")
store.residual(42)  # zero-copy from mmap
```

See [docs/residual-trace.md](docs/residual-trace.md) for the full writeup.

## Documentation

| Doc | Description |
|---|---|
| [docs/lql-spec.md](docs/lql-spec.md) | LQL language specification (v0.3) |
| [docs/vindex-format-spec.md](docs/vindex-format-spec.md) | Vindex file format specification (v0.3, ~98% implemented) |
| [docs/vindex-operations-spec.md](docs/vindex-operations-spec.md) | Vindex operations, API, patches (~98% implemented) |
| [docs/vindex-ecosystem-spec.md](docs/vindex-ecosystem-spec.md) | Distributed hosting, HuggingFace, Vindexfile (~85% implemented) |
| [docs/lql-guide.md](docs/lql-guide.md) | LQL quick start guide |
| [docs/cli.md](docs/cli.md) | CLI reference |
| [docs/inference-engine.md](docs/inference-engine.md) | Inference engine — BLAS-fused attention, Metal GPU, auto-calibration |
| [docs/ffn-graph-layer.md](docs/ffn-graph-layer.md) | FFN graph layer — mmap walk faster than dense (517ms vs 535ms), all 34 layers |
| [docs/walk-boundary-sweep.md](docs/walk-boundary-sweep.md) | Walk boundary sweep — correctness proof across all layer boundaries |
| [docs/knowledge-pipeline.md](docs/knowledge-pipeline.md) | Knowledge labelling pipeline |
| [docs/residual-trace.md](docs/residual-trace.md) | Residual stream trace — decomposition, storage, tiered context |
| [docs/trace-format-spec.md](docs/trace-format-spec.md) | Trace file format specification (.bin, .bndx, .ctxt) |

## Building & Testing

```bash
cargo build --release                    # optimised build
cargo build --release --features metal   # with Metal GPU backend
cargo test                               # all tests across all crates
cargo test -p larql-inference            # inference engine tests (109 tests)
cargo test -p larql-inference --features metal  # + Metal GPU tests (115 tests)

# Inference engine examples
cargo run --release -p larql-inference --example attention_demo    # fused attention demo
cargo run --release -p larql-inference --example bench_attention   # attention benchmarks
cargo run --release -p larql-inference --example backend_demo --features metal   # backend demo
cargo run --release -p larql-inference --example bench_backend --features metal  # backend benchmarks
cargo run --release -p larql-inference --example bench_inference   # full inference benchmarks

# Vindex tools (build once, enables mmap walk)
cargo run --release -p larql-vindex --example convert_gates_f32 -- path/to/vindex   # f16→f32 gate vectors
cargo run --release -p larql-vindex --example build_down_features -- path/to/vindex  # feature-major down vectors
cargo run --release -p larql-vindex --example build_up_features -- path/to/vindex    # feature-major up vectors

# Server (walk inference over HTTP)
cargo run --release -p larql-server -- path/to/vindex --port 8080

# Vindex and LQL examples
cargo run -p larql-vindex --example vindex_demo    # vindex feature demo
cargo run -p larql-vindex --example vindex_bench --release  # benchmarks
cargo run -p larql-lql --example parser_demo       # parser demo
cargo run -p larql-lql --example lql_demo          # LQL spec compliance
```

## License

Apache-2.0
