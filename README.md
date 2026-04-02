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
larql-inference   Forward pass, attention, WalkFfn (uses vindex KNN)
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
DESCRIBE "France";
DESCRIBE "Einstein" ALL LAYERS VERBOSE;
WALK "The capital of France is" TOP 10;

-- Run inference (needs model weights in vindex)
INFER "The capital of France is" TOP 5 COMPARE;

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

### Inference (Gemma 3 4B, Apple Silicon)

| Operation | Latency |
|---|---|
| Walk prediction (no attention) | 33ms |
| INFER prediction (with attention) | ~11s |
| DESCRIBE (knowledge browse) | 33ms |

## Documentation

| Doc | Description |
|---|---|
| [docs/lql-spec.md](docs/lql-spec.md) | LQL language specification (v0.3) |
| [docs/vindex-format-spec.md](docs/vindex-format-spec.md) | Vindex file format specification (v0.3, ~98% implemented) |
| [docs/vindex-operations-spec.md](docs/vindex-operations-spec.md) | Vindex operations, API, patches (~98% implemented) |
| [docs/vindex-ecosystem-spec.md](docs/vindex-ecosystem-spec.md) | Distributed hosting, HuggingFace, Vindexfile (~85% implemented) |
| [docs/lql-guide.md](docs/lql-guide.md) | LQL quick start guide |
| [docs/cli.md](docs/cli.md) | CLI reference |
| [docs/knowledge-pipeline.md](docs/knowledge-pipeline.md) | Knowledge labelling pipeline |

## Building & Testing

```bash
cargo build --release       # optimized build
cargo test                  # 124+ vindex tests, 545+ across all crates
cargo run -p larql-vindex --example vindex_demo    # vindex feature demo
cargo run -p larql-vindex --example vindex_bench --release  # benchmarks
cargo run -p larql-lql --example parser_demo       # parser demo
cargo run -p larql-lql --example lql_demo          # LQL spec compliance
```

## License

Apache-2.0
