# chuk-larql-rs

Knowledge graphs extracted from neural network weights. Five crates, one binary, one graph format.

LARQL extracts knowledge from language models through three complementary methods:

- **Weight walking** — reads FFN weight matrices directly from safetensors files. Zero forward passes. Extracts what each neuron feature activates on and what it produces.
- **Residual capture** — runs targeted forward passes for seed entities and captures the hidden state at specific layers. These residuals seed SurrealDB's vector store for bulk factual discovery.
- **BFS probing** — sends structured prompts to a running model endpoint, chains next-token predictions into edges. Used as a validator for high-value candidates.

The extraction pipeline produces edges. Edges flow into the runtime graph. No vectors at runtime.

## Quick start

```bash
make release

# Extract lexical graph from weights (zero forward passes)
larql weight-extract google/gemma-3-4b-it -o knowledge.larql.json

# Run inference from extracted weights (full forward pass in Rust)
larql predict google/gemma-3-4b-it --prompt "The capital of France is" -k 10

# Extract vectors for SurrealDB workshop
larql vector-extract google/gemma-3-4b-it -o vectors/ --resume

# Capture L25 residuals for seed entities (forward passes)
larql residuals capture google/gemma-3-4b-it \
    --entities "France,Germany,Japan,Mozart,Einstein" \
    --layer 25 -o residuals.vectors.ndjson

# Load into SurrealDB
larql vector-load vectors/ --ns larql --db gemma3_4b

# Query the graph
larql query --graph knowledge.larql.json France
larql stats knowledge.larql.json
```

## Architecture

Five crates. Five concerns. Clean boundaries.

```
larql-models      what the model IS     (config, traits, tensor keys)
larql-core        what the model KNOWS  (graph engine, edges, queries)
larql-inference   what the model DOES   (forward pass, extraction, capture)
larql-surreal     where you EXPLORE     (SurrealDB loading, schemas, queries)
larql-cli         how you USE it        (commands, flags, formatting)
```

Dependency flow is one direction, no cycles:

```
larql-models      <- serde, serde_json (zero compute deps)
     ^        ^
larql-core    larql-surreal     <- independent, both depend on models
     ^            ^
larql-inference   |             <- depends on models + core + ndarray + blas
     ^            |
larql-cli --------+             <- depends on all
```

### larql-models

Pure configuration. Model architecture trait, config.json parsing, tensor key mappings. Auto-detects Gemma 3, Llama, or falls back to generic. Zero compute dependencies.

```rust
let arch = larql_models::detect_architecture(model_dir)?;
println!("{}", arch.family());           // "gemma3"
println!("{}", arch.norm_weight_offset()); // 1.0 (Gemma), 0.0 (Llama)
println!("{}", arch.embed_scale());        // sqrt(hidden_size) for Gemma
println!("{}", arch.attn_q_key(5));        // "layers.5.self_attn.q_proj.weight"
```

### larql-core

Graph engine. Indexed adjacency structure with select, walk, search, subgraph, shortest path, PageRank, BFS/DFS traversal, merge, diff. JSON and MessagePack serialization.

```rust
let mut graph = Graph::new();
graph.add_edge(Edge::new("France", "capital-of", "Paris").with_confidence(0.99));
let results = graph.select("France", Some("capital-of")); // -> [Paris edge]
let (cost, path) = shortest_path(&graph, "France", "Europe").unwrap();
```

### larql-inference

Transformer inference engine. Loads safetensors weights, runs BLAS-accelerated forward passes. Also owns weight walking (edge extraction), attention walking (OV circuits), vector extraction, and residual capture.

```rust
let model = InferenceModel::load("google/gemma-3-4b-it")?;
let result = predict(model.weights(), model.tokenizer(), &token_ids, 10);
// result.predictions[0] = ("Paris", 0.9967)
```

### larql-surreal

SurrealDB integration. Reads NDJSON vector files, generates schema DDL with HNSW indexes, produces INSERT SQL, tracks load progress. No compute dependencies — just file I/O and string generation.

### larql-cli

CLI binary connecting all crates. 14 commands.

## Documentation

| Doc | Description |
|---|---|
| [docs/cli.md](docs/cli.md) | Full CLI reference — all commands, flags, examples |
| [docs/format.md](docs/format.md) | Graph file format specification — JSON and MessagePack |
| [docs/weight-extraction.md](docs/weight-extraction.md) | Weight extraction pipeline — weights to vectors to SurrealDB to graph |
| [docs/confidence.md](docs/confidence.md) | Confidence and selectivity scoring |
| [docs/circuit-types.md](docs/circuit-types.md) | Circuit type analysis — layer architecture from gate-down cosines |
| [docs/validation.md](docs/validation.md) | Graph validation — extraction faithfulness, dark space analysis |
| [docs/findings.md](docs/findings.md) | Research findings — circuit types, cross-lingual knowledge, attention routing |

## The extraction pipeline

```
weight-extract        -> lexical edges (8.2M, zero forward passes)
vector-extract     -> weight vectors to NDJSON (for SurrealDB)
vector-load        -> vectors into SurrealDB with HNSW indexes
residuals capture  -> L25 residuals for seed entities (targeted forward passes)
                     |
              SurrealDB workshop: format-adjusted queries discover factual edges
                     |
bfs                -> validate top candidates with forward passes
                     |
              merge -> knowledge.larql.json -> Rust runtime (edges only, no vectors)
```

The vectors are the microscope. The edges are the photograph. You ship the photograph.

## Extraction commands

### Weight walking

Reads safetensors directly. Zero forward passes. BLAS-accelerated.

```bash
larql weight-extract google/gemma-3-4b-it -o knowledge.larql.json
larql weight-extract google/gemma-3-4b-it --layer 26 -o L26.larql.json --stats stats.json
```

### Attention walking

Extracts routing edges from attention OV circuits.

```bash
larql attention-extract google/gemma-3-4b-it -o attention.larql.json
```

### Inference

Full forward pass from extracted safetensors weights. BLAS-accelerated. No MLX, no PyTorch.

```bash
larql predict google/gemma-3-4b-it --prompt "The capital of France is" -k 10
# 1. Paris (99.67%)
```

### Vector extraction

Extracts full weight vectors to NDJSON for SurrealDB ingestion.

```bash
# All implemented components
larql vector-extract google/gemma-3-4b-it -o vectors/ \
    --components ffn_down,ffn_gate,ffn_up,attn_ov,attn_qk,embeddings --resume

# Just factual layers
larql vector-extract google/gemma-3-4b-it -o vectors/ \
    --components ffn_down,ffn_gate --layers 25,26,27,28,29,30,31,32,33
```

| Component | What it stores | Dim | Per layer |
|---|---|---|---|
| `ffn_down` | Down projection column (output direction) | hidden | intermediate_size |
| `ffn_gate` | Gate projection row (input selectivity) | hidden | intermediate_size |
| `ffn_up` | Up projection row | hidden | intermediate_size |
| `attn_ov` | Mean OV circuit output direction | hidden | num_kv_heads |
| `attn_qk` | Q/K head projections | head_dim x hidden | num_q + num_kv |
| `embeddings` | Token embedding rows | hidden | vocab_size |

### Residual capture

Runs forward passes for seed entities and captures the hidden state at specified layers.

```bash
larql residuals capture google/gemma-3-4b-it \
    --entities "France,Germany,Japan,Mozart,Einstein" \
    --layer 25 -o residuals.vectors.ndjson

# Multiple layers
larql residuals capture google/gemma-3-4b-it \
    --entities entities.txt --layer 25 --layer 26 --layer 29 \
    -o residuals.vectors.ndjson
```

### SurrealDB loading

```bash
larql vector-load vectors/ --ns larql --db gemma3_4b
larql vector-load vectors/ --ns larql --db gemma3_4b --schema-only
```

### BFS probing

```bash
larql bfs --seeds "France,Germany" --templates templates.json \
    --endpoint http://localhost:11434/v1 --model gemma3:4b-it \
    -o knowledge.larql.json
```

## Query commands

```bash
larql query --graph knowledge.larql.json France capital-of
larql describe --graph knowledge.larql.json France
larql stats knowledge.larql.json
larql validate knowledge.larql.json
larql merge graph1.larql.json graph2.larql.json -o merged.larql.json
```

See [docs/cli.md](docs/cli.md) for full reference.

## Workspace structure

```
chuk-larql-rs/
├── crates/
│   ├── larql-models/     Model config — architecture traits, tensor keys, detection
│   ├── larql-core/       Graph engine — edges, queries, algorithms, serialization
│   ├── larql-inference/  Inference — forward pass, extraction, walkers, capture
│   ├── larql-surreal/    SurrealDB — vector loading, schemas, SQL generation
│   ├── larql-cli/        CLI binary — 14 commands over all crates
│   └── larql-python/     PyO3 binding — native Python extension
├── docs/
│   ├── cli.md            CLI reference
│   ├── format.md         Graph format specification
│   ├── confidence.md     Confidence and selectivity scoring
│   ├── weight-extraction.md  Full pipeline documentation
│   ├── circuit-types.md  Circuit type analysis
│   ├── validation.md     Extraction faithfulness
│   └── findings.md       Research findings
├── scripts/              Python analysis scripts
├── surql/                SurrealDB query templates
├── Makefile
└── README.md
```

## Python integration

The `chuk-larql` Python package uses this Rust engine as its native backend.

```python
from chuk_larql import Graph, Edge
from _larql_core import weight_walk, attention_walk, load, save

g = weight_walk("google/gemma-3-4b-it")
save(g, "knowledge.larql.json")
```

## Building

```bash
make build          # debug build (all crates)
make release        # optimized CLI binary
make test           # run all 191 tests
make lint           # clippy (zero warnings)
make ci             # fmt-check + lint + test
make demos          # run all example demos
make bench          # graph engine + SQL generation benchmarks
make bench-all      # all benchmarks including inference (needs model)
make python-build   # build Python extension (requires virtualenv)
```

## Benchmarks

### Graph engine (100K edges)

| Operation | Latency |
|---|---|
| Insert edge | 1.3 us |
| select() | 0.2 us |
| exists() | 0.1 us |
| shortest_path (1K nodes) | 14 us |
| PageRank (1K nodes) | 13 ms |
| JSON serialize 100K edges | 141 ms (9.9 MB) |
| MsgPack serialize | 132 ms (4.7 MB, 53% smaller) |

### Inference (Gemma 3 4B, Apple Silicon)

| Operation | Latency |
|---|---|
| RMS norm | 13 us |
| RoPE | 22 us |
| GQA attention | 73 us |
| FFN forward | 6 ms |
| Full predict (6 tokens, 34 layers) | 788 ms |
| Throughput | 1.3 queries/sec |

### SQL generation

| Operation | Throughput |
|---|---|
| Single insert (dim=2560) | 7.7K/sec |
| SQL output | 395 MB/sec |

## Status

### What's working

- **Weight walker** — 8.2M edges from Gemma 3-4B in 40 minutes. Confidence + selectivity scoring. Per-layer stats.
- **Attention walker** — OV circuit extraction with same scoring. 10,919 edges from 136 heads.
- **Inference engine** — Full Rust forward pass from safetensors. BLAS-accelerated. 14/14 correct on capital-of test set. No MLX, no PyTorch.
- **Model architecture** — Auto-detection from config.json. Gemma 3 (norm offset, embed scaling, QK norm, sliding window) and generic/Llama support.
- **Vector extractor** — all 6 components: ffn_down, ffn_gate, ffn_up, attn_ov, attn_qk, embeddings.
- **Residual capture** — forward pass through transformer layers, captures hidden state at any layer.
- **SurrealDB loader** — NDJSON to SurrealDB with HNSW indexes, batch insert, resume, schema-only mode.
- **Core graph engine** — full indexed graph with select, walk, search, subgraph, merge, shortest path, PageRank, BFS/DFS, diff.
- **BFS extraction** — template-based probing with multi-token chaining.
- **Serialization** — JSON and MessagePack with format auto-detection.
- **CLI** — 14 commands: weight-extract, attention-extract, predict, vector-extract, vector-import, vector-load, vector-export-surql, residuals, bfs, stats, query, describe, validate, merge.
- **PyO3 binding** — full Python API parity.
- **Tests** — 191 Rust tests across 5 crates. Zero clippy warnings.
- **Benchmarks** — component-level and end-to-end for graph engine, inference, and SQL generation.
- **Examples** — 10 runnable examples across 4 crates demonstrating all functionality.

### What's next

- CI / GitHub Actions
- Hybrid inference: swap FFN layers for graph lookups (attention from weights, knowledge from graph)
- `FfnBackend` trait: `WeightFfn`, `GraphFfn`, `SurrealFfn` — same interface, three backends
- Layer-by-layer comparison: `--ffn weights:0-25,graph:26-33`
- Additional model architectures: Llama, DeepSeek
- `larql filter` command (post-extraction confidence/selectivity filtering)
- `larql merge` improvements (max_confidence strategy at scale)
- Packed binary edge format for runtime graphs

## License

Apache-2.0
