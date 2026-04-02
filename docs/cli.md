# LARQL CLI Reference

```
larql <COMMAND> [OPTIONS]
```

## Core commands

The primary workflow: extract a vindex, launch the REPL, or build from a Vindexfile.

| Command | Description |
|---|---|
| `extract-index` | Build a .vindex from a model (safetensors/GGUF/MLX → queryable format) |
| `build` | Build a custom model from a Vindexfile (FROM + PATCH + INSERT) |
| `convert` | Convert between formats (GGUF → vindex, safetensors → vindex, gguf-info) |
| `hf` | HuggingFace Hub: download or publish vindexes |
| `verify` | Verify vindex file integrity (SHA256 checksums) |
| `repl` | Launch the LQL interactive REPL |
| `lql` | Execute a single LQL statement |
| `walk` | Walk the model as a local vector index (gate KNN + down lookup) |
| `vindex-bench` | Benchmark vindex walk: accuracy vs dense, throughput |
| `serve` | Serve a vindex over HTTP (knowledge queries, patches, multi-model) |

### `larql serve`

Serve a vindex over HTTP. Loads a vindex into memory and exposes REST endpoints for knowledge queries, feature walks, edge selection, and patch management.

```
larql serve <VINDEX_PATH> [OPTIONS]
larql serve --dir <DIR> [OPTIONS]
```

| Flag | Description | Default |
|---|---|---|
| `<VINDEX_PATH>` | Path to .vindex directory or `hf://` URL | — |
| `--dir <DIR>` | Serve all .vindex directories in folder | — |
| `--port <PORT>` | Listen port | 8080 |
| `--host <HOST>` | Bind address | 0.0.0.0 |
| `--no-infer` | Disable inference endpoint (browse-only, saves memory) | false |
| `--cors` | Enable CORS headers for browser access | false |
| `--api-key <KEY>` | Require Bearer token auth (health exempt) | — |
| `--rate-limit <SPEC>` | Per-IP rate limit (e.g., "100/min", "10/sec") | — |
| `--max-concurrent <N>` | Max concurrent requests | 100 |
| `--cache-ttl <SECS>` | Cache TTL for DESCRIBE results (0 = disabled) | 0 |
| `--grpc-port <PORT>` | Enable gRPC server on this port | — |
| `--tls-cert <PATH>` | TLS certificate for HTTPS | — |
| `--tls-key <PATH>` | TLS private key for HTTPS | — |
| `--log-level <LEVEL>` | Logging level | info |

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| GET | `/v1/describe?entity=France` | Knowledge edges (with probe relation labels) |
| GET | `/v1/walk?prompt=...&top=5` | Feature scan for a prompt |
| POST | `/v1/select` | SQL-style edge query |
| POST | `/v1/infer` | Full forward pass (walk/dense/compare) |
| GET | `/v1/relations` | List known relation types |
| GET | `/v1/stats` | Model and index statistics |
| POST | `/v1/patches/apply` | Apply a patch in-memory |
| GET | `/v1/patches` | List active patches |
| DELETE | `/v1/patches/{name}` | Remove a patch |
| POST | `/v1/walk-ffn` | Decoupled inference (client sends residual, server returns features) |
| WS | `/v1/stream` | WebSocket streaming (layer-by-layer DESCRIBE) |
| GET | `/v1/health` | Health check (auth exempt) |
| GET | `/v1/models` | List loaded models |

**Multi-model:** When using `--dir`, each model gets its own namespace: `/v1/{model_id}/describe`, etc.

**Examples:**

```bash
# Development — single model
larql serve output/gemma3-4b-v2.vindex --port 8080

# Multi-model server
larql serve --dir ./vindexes/ --port 8080

# From HuggingFace
larql serve "hf://chrishayuk/gemma-3-4b-it-vindex" --port 8080

# Browse-only with CORS for web clients
larql serve output/gemma3-4b-v2.vindex --no-infer --cors --port 8080

# With auth + HTTPS
larql serve output/gemma3-4b.vindex --api-key "sk-abc123" --tls-cert cert.pem --tls-key key.pem

# With rate limiting + DESCRIBE cache
larql serve output/gemma3-4b.vindex --rate-limit "100/min" --cache-ttl 300

# Query from the REPL
larql repl
> USE REMOTE "http://localhost:8080";
> DESCRIBE "France";
> INFER "The capital of France is" TOP 5;
> APPLY PATCH "local-facts.vlp";    -- stays client-side
```

**Sessions:** Clients can send `X-Session-Id` header to isolate patches per session. Without the header, patches apply globally.

See `crates/larql-server/README.md` for full API documentation.

### `larql repl`

Launch the LQL interactive REPL. Tab completion, history, multi-line input.

```bash
larql repl
```

```sql
larql> USE "gemma3-4b.vindex";
larql> DESCRIBE "France";
larql> WALK "The capital of France is" TOP 5;
larql> INSERT INTO EDGES (entity, relation, target) VALUES ("John", "lives-in", "London");
larql> SAVE PATCH "edits.vlp";
```

Direct model access (no extraction needed):
```sql
larql> USE MODEL "google/gemma-3-4b-it";
larql> INFER "The capital of France is" TOP 5;
-- WALK/DESCRIBE/SELECT require EXTRACT into a vindex first
```

Remote mode:
```sql
larql> USE REMOTE "http://localhost:8080";
larql> DESCRIBE "France";
```

### `larql lql`

Execute a single LQL statement from the command line.

```bash
larql lql 'USE "gemma3-4b.vindex"; DESCRIBE "France";'
larql lql 'USE "gemma3-4b.vindex"; WALK "Einstein" TOP 10;'
```

## Legacy extraction commands

These commands predate the REPL and vindex format. They remain available for low-level extraction, debugging, and research but most users should use `extract-index` + `repl` instead.

### `larql weight-extract` (legacy)

Extract edges from FFN weight matrices. Zero forward passes. Pure matrix multiplication.

```
larql weight-extract <MODEL> --output <OUTPUT> [OPTIONS]
```

| Argument/Flag | Description |
|---|---|
| `<MODEL>` | Model path or HuggingFace model ID (e.g. `google/gemma-3-4b-it`) |
| `-o, --output <OUTPUT>` | Output file (`.larql.json` or `.larql.bin`) |
| `-l, --layer <LAYER>` | Single layer to walk. Default: all layers |
| `--top-k <TOP_K>` | Top-k tokens per feature [default: 5] |
| `--min-score <MIN_SCORE>` | Minimum raw activation score for top-k selection [default: 0.02] |
| `--min-confidence <MIN_CONFIDENCE>` | Minimum normalized confidence [0-1] to keep an edge [default: 0.0] |
| `--stats <STATS>` | Write layer statistics to a separate JSON file |

**Model resolution:** Accepts a local directory path or a HuggingFace model ID. Model IDs are resolved from the HuggingFace cache at `~/.cache/huggingface/hub/` (or `$HF_HOME/hub/`).

**Resume:** If the output file already exists, completed layers are detected from edge metadata and skipped. Saves after each layer. Safe to interrupt and re-run.

**Confidence scoring:** Each edge gets a normalized confidence `c` in [0, 1], computed as `(c_in × c_out) / max(c_in × c_out)` per layer. Raw scores `c_in` (input selectivity) and `c_out` (output strength) are stored in metadata.

**Examples:**

```bash
# Full extraction
larql weight-extract google/gemma-3-4b-it -o knowledge.larql.json

# Single layer test
larql weight-extract google/gemma-3-4b-it --layer 26 -o L26.larql.json

# Filtered extraction with stats
larql weight-extract google/gemma-3-4b-it \
    -o knowledge.larql.json \
    --min-confidence 0.1 \
    --stats stats.json

# MessagePack output (smaller, faster)
larql weight-extract google/gemma-3-4b-it -o knowledge.larql.bin
```

### `larql attention-extract` (legacy)

Extract routing edges from attention OV circuits. Zero forward passes.

```
larql attention-extract <MODEL> --output <OUTPUT> [OPTIONS]
```

| Argument/Flag | Description |
|---|---|
| `<MODEL>` | Model path or HuggingFace model ID |
| `-o, --output <OUTPUT>` | Output file (`.larql.json` or `.larql.bin`) |
| `-l, --layer <LAYER>` | Single layer to walk. Default: all layers |
| `--top-k <TOP_K>` | Top-k tokens per head [default: 3] |
| `--min-score <MIN_SCORE>` | Minimum score [default: 0.0] |

**How it works:** For each attention head, computes the OV circuit (`O_h @ V_h`), projects all vocab tokens through it, finds the most amplified inputs, and decodes what output tokens each produces.

**Resume:** Same as `weight-extract` — detects completed layers and skips them.

**Examples:**

```bash
larql attention-extract google/gemma-3-4b-it -o attention.larql.json
larql attention-extract google/gemma-3-4b-it --layer 12 -o attention-L12.larql.json
```

### `larql predict` (legacy)

Run a full transformer forward pass from extracted safetensors weights and return top-k next-token predictions. Pure Rust inference — no MLX, no PyTorch.

```
larql predict <MODEL> --prompt <PROMPT> [OPTIONS]
```

| Argument/Flag | Description |
|---|---|
| `<MODEL>` | Model path or HuggingFace model ID |
| `-p, --prompt <PROMPT>` | Prompt text to predict the next token for |
| `-k, --top-k <TOP_K>` | Number of top predictions to show [default: 10] |

**How it works:** Loads all safetensors weights, tokenizes the prompt (with BOS token), runs the full forward pass through all layers (embedding, attention with RoPE, GQA, QK norm, FFN with SiLU gating, all layer norms), projects the final residual against the embedding matrix, and returns softmax probabilities.

**Architecture awareness:** Uses the `ModelArchitecture` trait to handle model-specific behavior. Gemma 3 gets +1 norm offset, sqrt(hidden) embedding scale, QK normalization, and 4 norms per layer. Llama/generic gets standard behavior.

**Performance:** ~800ms per query for Gemma 3 4B on Apple Silicon (BLAS-accelerated, 34 layers, 6 tokens).

**Examples:**

```bash
# Basic prediction
larql predict google/gemma-3-4b-it --prompt "The capital of France is" -k 5
# 1. Paris (99.67%)

# Factual queries
larql predict google/gemma-3-4b-it --prompt "The largest planet is" -k 3
# 1. Jupiter (99.86%)

# Works with any HuggingFace model in cache
larql predict google/gemma-3-4b-it -p "Water freezes at" -k 10
```

### `larql index-gates` (legacy)

Build a precomputed gate index for graph-based FFN. Offline step — run once per model. Eliminates the gate matmul at inference time.

```
larql index-gates <MODEL> --output <OUTPUT> [OPTIONS]
```

| Flag | Description |
|---|---|
| `<MODEL>` | Model path or HuggingFace model ID |
| `-o, --output <OUTPUT>` | Output index file (`.gate-index.jsonl`) |
| `--features-per-token <N>` | Features to index per token per layer [default: 100] |
| `--top-tokens <N>` | Top tokens to match at runtime [default: 10] |
| `--layers <LAYERS>` | Layers to index (e.g. `0-33` or `26,27,28`). Default: all |

**Examples:**

```bash
larql index-gates google/gemma-3-4b-it -o gates.gate-index.jsonl
larql index-gates google/gemma-3-4b-it -o gates.gate-index.jsonl --layers 24-33
```

### `larql extract-routes` (legacy)

Extract attention routing patterns from forward passes. Captures which FFN features activate for each entity/relation combination.

```
larql extract-routes <MODEL> --output <OUTPUT> [OPTIONS]
```

| Flag | Description |
|---|---|
| `<MODEL>` | Model path or HuggingFace model ID |
| `-o, --output <OUTPUT>` | Output JSON file for the routing table |
| `--top-k <N>` | Top features to capture per layer per forward pass [default: 50] |
| `--min-activation <F>` | Minimum absolute activation to record [default: 1.0] |
| `-e, --entities <ENTITIES>` | Comma-separated entities (overrides defaults) |
| `--layers <LAYERS>` | Comma-separated layers to capture. Default: all |

**Examples:**

```bash
larql extract-routes google/gemma-3-4b-it -o routes.json
larql extract-routes google/gemma-3-4b-it -o routes.json --entities "France,Germany,Japan" --layers 25,26,27
```

### `larql walk`

Walk the model as a local vector index — gate KNN followed by down token lookup. No forward pass needed when using a `.vindex`.

```
larql walk --prompt <PROMPT> [OPTIONS]
```

| Flag | Description |
|---|---|
| `-p, --prompt <PROMPT>` | Prompt text to walk through the model |
| `--index <PATH>` | Path to a `.vindex` directory (self-contained, no model needed) |
| `-m, --model <MODEL>` | Model path or HuggingFace model ID |
| `--gate-vectors <PATH>` | Path to extracted ffn_gate vectors (alternative to `--index`) |
| `--down-vectors <PATH>` | Path to extracted ffn_down vectors (alternative to `--index`) |
| `-k, --top-k <N>` | Top-K features per layer for gate KNN [default: 10] |
| `-l, --layers <LAYERS>` | Layers to walk. Comma-separated or range. Default: all |
| `--predict-top-k <N>` | Number of top predictions to show [default: 10] |
| `--predict` | Run full forward pass with walk FFN and show predictions (requires `--model`) |
| `--compare` | Compare walk FFN predictions against dense ground truth (requires `--model`) |
| `--down-top-k <N>` | Number of down tokens to show per feature [default: 5] |
| `-v, --verbose` | Show verbose loading and timing info |

**Examples:**

```bash
# Walk with a pre-built .vindex
larql walk --prompt "The capital of France is" --index model.vindex

# Walk with loose vector files
larql walk --prompt "The capital of France is" \
    --gate-vectors vectors/ffn_gate.vectors.jsonl \
    --down-vectors vectors/ffn_down.vectors.jsonl

# Walk + compare against ground truth
larql walk --prompt "The capital of France is" --index model.vindex --model google/gemma-3-4b-it --compare
```

### `larql attention-capture` (legacy)

Capture and compare attention patterns across multiple prompts. Shows which heads attend similarly or differently.

```
larql attention-capture <MODEL> --prompts <PROMPTS> [OPTIONS]
```

| Flag | Description |
|---|---|
| `<MODEL>` | Model path or HuggingFace model ID |
| `-p, --prompts <PROMPTS>` | Prompts to compare (comma-separated) |
| `-l, --layers <LAYERS>` | Layers to capture. Comma-separated or range. Default: all |
| `--threshold <F>` | Attention threshold — only show heads with max attention above this [default: 0.1] |
| `-v, --verbose` | Show verbose per-head details |

**Examples:**

```bash
larql attention-capture google/gemma-3-4b-it \
    --prompts "The capital of France is,The capital of Germany is,The capital of Japan is"

larql attention-capture google/gemma-3-4b-it \
    --prompts "The capital of France is,The language of France is" \
    --layers 20-33 --threshold 0.2
```

### `larql qk-templates` (legacy)

Extract attention template circuits from QK weight decomposition. Identifies which heads are "fixed" (same pattern regardless of entity) vs "variable".

```
larql qk-templates <MODEL> [OPTIONS]
```

| Flag | Description |
|---|---|
| `<MODEL>` | Model path or HuggingFace model ID |
| `-t, --templates <TEMPLATES>` | Template prompts (format: `relation:prompt`, comma-separated). Uses built-in templates if omitted |
| `-l, --layers <LAYERS>` | Layers to analyze. Default: all |
| `--threshold <F>` | Correlation threshold below which a head is "variable" [default: 0.95] |
| `--top-components <N>` | Number of top SVD components to show per head [default: 5] |

**Examples:**

```bash
larql qk-templates google/gemma-3-4b-it
larql qk-templates google/gemma-3-4b-it --layers 20-33 --threshold 0.90
```

### `larql ov-gate` (legacy)

Map attention OV circuits to FFN gate features. Shows what each attention head activates in the next layer's FFN.

```
larql ov-gate <MODEL> [OPTIONS]
```

| Flag | Description |
|---|---|
| `<MODEL>` | Model path or HuggingFace model ID |
| `-l, --layers <LAYERS>` | Layers to analyze. Default: all |
| `-k, --top-k <N>` | Top-K gate features to show per head [default: 10] |
| `--heads <HEADS>` | Only show specific heads (for focused analysis) |
| `-v, --verbose` | Show verbose per-feature details |

**Examples:**

```bash
larql ov-gate google/gemma-3-4b-it --layers 25,26,27
larql ov-gate google/gemma-3-4b-it --layers 26 --heads 0,1,2 -k 20 -v
```

### `larql extract-index`

Build a `.vindex` — the model decompiled to a standalone vector index. Can be used with `larql walk` without needing the original model.

```
larql extract-index [MODEL] --output <OUTPUT> [OPTIONS]
```

| Flag | Description |
|---|---|
| `<MODEL>` | Model path or HuggingFace model ID (not needed with `--from-vectors`) |
| `-o, --output <OUTPUT>` | Output path for the `.vindex` directory |
| `--level <LEVEL>` | Extract level: `browse` (default), `inference`, `all` |
| `--f16` | Store weights in f16 (half precision, halves file sizes) |
| `--from-vectors <PATH>` | Build from already-extracted NDJSON vector files instead of model weights |
| `--down-top-k <N>` | Top-K tokens per feature in down metadata [default: 10] |
| `--include-weights` | Alias for `--level all` (deprecated) |
| `--resume` | Skip stages that already have output files |

**Examples:**

```bash
# Browse-only (~3 GB at f16, enables DESCRIBE/WALK/SELECT)
larql extract-index google/gemma-3-4b-it -o model.vindex --f16

# With inference weights (~6 GB at f16, enables INFER)
larql extract-index google/gemma-3-4b-it -o model.vindex --level inference --f16

# All weights (~10 GB at f16, enables COMPILE)
larql extract-index google/gemma-3-4b-it -o model.vindex --level all --f16

# Build from pre-extracted vectors
larql extract-index -o model.vindex --from-vectors vectors/

# Resume an interrupted build
larql extract-index google/gemma-3-4b-it -o model.vindex --f16 --resume
```

### `larql build`

Build a custom model from a Vindexfile (declarative: FROM + PATCH + INSERT).

```
larql build [DIR] [OPTIONS]
```

| Flag | Description |
|---|---|
| `[DIR]` | Directory containing the Vindexfile [default: `.`] |
| `--stage <NAME>` | Build stage (e.g. `dev`, `prod`, `edge`) |
| `-o, --output <PATH>` | Output directory for the built vindex [default: `./build/vindex/`] |
| `--compile <FORMAT>` | Compile output to a model format after building |

**Examples:**

```bash
# Build from Vindexfile in current directory
larql build .

# Build a specific stage
larql build . --stage prod

# Build to a custom output path
larql build . --output custom.vindex
```

### `larql convert`

Convert between model formats.

```
larql convert <SUBCOMMAND>
```

| Subcommand | Description |
|---|---|
| `gguf-to-vindex` | Convert a GGUF model to a vindex (dequantized to f32) |
| `safetensors-to-vindex` | Convert safetensors model to a vindex |
| `gguf-info` | Show GGUF file metadata and detected architecture |

**Examples:**

```bash
# Convert GGUF to vindex
larql convert gguf-to-vindex model-Q4_K_M.gguf -o model.vindex --f16

# Show GGUF metadata
larql convert gguf-info model-Q4_K_M.gguf

# Convert safetensors to vindex
larql convert safetensors-to-vindex ./model/ -o model.vindex --level inference --f16
```

Supported GGUF quantization types for reading: F32, F16, BF16, Q4_0, Q4_1, Q8_0. All tensors are dequantized to f32 during conversion.

### `larql hf`

HuggingFace Hub: download or publish vindexes.

```
larql hf <SUBCOMMAND>
```

| Subcommand | Description |
|---|---|
| `download` | Download a vindex from HuggingFace to local cache |
| `publish` | Upload a local vindex to HuggingFace |

**Examples:**

```bash
# Download a vindex
larql hf download chrishayuk/gemma-3-4b-it-vindex

# Download to a specific directory
larql hf download chrishayuk/gemma-3-4b-it-vindex -o ./gemma3-4b.vindex

# Download a specific version
larql hf download chrishayuk/gemma-3-4b-it-vindex --revision v2.0

# Publish a vindex
larql hf publish ./gemma3-4b.vindex --repo chrishayuk/gemma-3-4b-it-vindex
```

HuggingFace vindexes can also be used directly from the REPL:

```sql
larql> USE "hf://chrishayuk/gemma-3-4b-it-vindex";
larql> DESCRIBE "France";
```

Publishing requires `HF_TOKEN` environment variable or `huggingface-cli login`.

### `larql verify`

Verify vindex file integrity against stored SHA256 checksums.

```
larql verify <VINDEX>
```

**Example:**

```bash
larql verify gemma3-4b.vindex
# gate_vectors.bin ... OK (1.66 GB)
# embeddings.bin ... OK (1.25 GB)
# down_meta.bin ... OK (2.0 MB)
# All 3 files verified.
```

### `larql vector-extract` (legacy)

Extract full weight vectors to intermediate NDJSON files.

```
larql vector-extract <MODEL> --output <OUTPUT> [OPTIONS]
```

| Flag | Description |
|---|---|
| `<MODEL>` | Model path or HuggingFace model ID |
| `-o, --output <OUTPUT>` | Output directory for `.vectors.jsonl` files |
| `--components <COMPONENTS>` | Components to extract (comma-separated): `ffn_down`, `ffn_gate`, `ffn_up`, `attn_ov`, `attn_qk`, `embeddings` |
| `--layers <LAYERS>` | Layers to extract (comma-separated). Default: all |
| `--top-k <TOP_K>` | Top-k tokens for metadata per vector [default: 10] |
| `--resume` | Resume from existing output files |

**Examples:**

```bash
# Extract all components
larql vector-extract google/gemma-3-4b-it -o vectors/

# Extract only FFN down projections from layers 25-33
larql vector-extract google/gemma-3-4b-it -o vectors/ \
    --components ffn_down --layers 25,26,27,28,29,30,31,32,33
```

### `larql residuals capture` (legacy)

Capture residual stream vectors for entities via forward passes. The residuals are the hidden state at a specific layer — the signal that the next layer's features actually see during inference.

```
larql residuals capture <MODEL> --entities <ENTITIES> --output <OUTPUT> [OPTIONS]
```

| Flag | Description |
|---|---|
| `<MODEL>` | Model path or HuggingFace model ID |
| `-e, --entities <ENTITIES>` | Comma-separated entities, or path to a text file (one per line) |
| `-l, --layer <LAYER>` | Layer(s) to capture at. Can specify multiple times. [default: 25] |
| `--all-layers` | Capture at every layer |
| `-o, --output <OUTPUT>` | Output directory for NDJSON files |
| `--template <TEMPLATE>` | Prompt template. `{entity}` is replaced. Default: bare entity name |
| `--activations` | Also capture sparse FFN activations (top-K features per layer) |
| `--activation-top-k <N>` | Number of top features to record per layer [default: 50] |

**How it works:** Tokenizes each entity, runs a full forward pass through the transformer up to the target layer(s), and saves the last-token hidden state as a vector in NDJSON format.

**Use case:** Capture real residual vectors for a small set of entities (50–100). These can be used as query vectors for vindex gate KNN lookups to discover factual edges without additional forward passes.

**Examples:**

```bash
# L25 residuals for seed entities
larql residuals capture google/gemma-3-4b-it \
    --entities "France,Germany,Japan,Mozart,Einstein" \
    --layer 25 -o residuals-L25.vectors.ndjson

# Multiple layers
larql residuals capture google/gemma-3-4b-it \
    --entities entities.txt \
    --layer 25 --layer 26 --layer 29 \
    -o residuals.vectors.ndjson

# Full trajectory (all layers)
larql residuals capture google/gemma-3-4b-it \
    --entities "France" --all-layers \
    -o residuals-full.vectors.ndjson

# With prompt template
larql residuals capture google/gemma-3-4b-it \
    --entities "France,Germany" \
    --layer 25 \
    --template "The capital of {entity} is" \
    -o residuals-capital.vectors.ndjson
```

**Output format:** Same NDJSON as `vector-extract`:

```json
{"_header": true, "component": "residuals", "model": "google/gemma-3-4b-it", "dimension": 2560}
{"id": "France_L25", "layer": 25, "feature": 0, "vector": [...], "top_token": "Paris", "c_score": 12.4, ...}
```

### `larql bfs` (legacy)

BFS extraction from a running model endpoint.

```
larql bfs --seeds <SEEDS> --templates <TEMPLATES> --output <OUTPUT> [OPTIONS]
```

| Flag | Description |
|---|---|
| `-s, --seeds <SEEDS>` | Comma-separated seed entities |
| `-t, --templates <TEMPLATES>` | Path to templates JSON file |
| `-o, --output <OUTPUT>` | Output file (`.larql.json` or `.larql.bin`) |
| `-e, --endpoint <ENDPOINT>` | Model endpoint URL [default: `http://localhost:11434/v1`] |
| `-m, --model <MODEL>` | Model name for the endpoint |
| `--mock` | Use mock provider instead of HTTP |
| `--mock-knowledge <PATH>` | Path to mock knowledge JSON (with `--mock`) |
| `--max-depth <N>` | Maximum BFS depth [default: 3] |
| `--max-entities <N>` | Maximum entities to probe [default: 1000] |
| `--min-confidence <F>` | Minimum edge confidence [default: 0.3] |
| `--resume <PATH>` | Resume from a checkpoint file |

**Requires:** Templates JSON file defining prompt templates for each relation. See [format.md](format.md) for template format.

**Examples:**

```bash
# Against Ollama
larql bfs \
    --seeds "France,Germany,Japan" \
    --templates templates.json \
    --endpoint http://localhost:11434/v1 \
    --model gemma3:4b-it \
    -o knowledge.larql.json

# With mock provider
larql bfs \
    --seeds "France,Germany" \
    --templates templates.json \
    --mock --mock-knowledge mock.json \
    -o knowledge.larql.json
```

## Query commands

### `larql query` (legacy)

Select edges from a subject, optionally filtered by relation.

```
larql query --graph <GRAPH> <SUBJECT> [RELATION]
```

```bash
larql query --graph knowledge.larql.json France
larql query --graph knowledge.larql.json France capital-of
```

### `larql describe` (legacy)

Show all outgoing and incoming edges for an entity.

```
larql describe --graph <GRAPH> <ENTITY>
```

```bash
larql describe --graph knowledge.larql.json France
```

### `larql stats` (legacy)

Show graph statistics: entity count, edge count, relation count, connected components, average degree, average confidence, source distribution.

```
larql stats <GRAPH>
```

```bash
larql stats knowledge.larql.json
```

### `larql validate` (legacy)

Check a graph file for issues: zero-confidence edges, self-loops, empty subjects/objects.

```
larql validate <GRAPH>
```

```bash
larql validate knowledge.larql.json
```

### `larql merge` (legacy)

Merge multiple graph files into one.

```
larql merge <INPUT>... --output <OUTPUT> [OPTIONS]
```

| Flag | Description |
|---|---|
| `<INPUT>...` | Input graph files to merge (at least 2) |
| `-o, --output <OUTPUT>` | Output merged graph file |
| `--strategy <STRATEGY>` | Merge strategy: `union`, `max_confidence`, `source_priority` [default: `union`] |

**Examples:**

```bash
larql merge weights.larql.json attention.larql.json -o combined.larql.json
larql merge weights.larql.json bfs.larql.json -o combined.larql.json --strategy max_confidence
```

## Templates format

Used by `larql bfs`. A JSON array of prompt templates:

```json
[
  {
    "relation": "capital-of",
    "template": "The capital of {subject} is",
    "multi_token": true,
    "stop_tokens": [".", "\n", ",", ";"]
  }
]
```

| Field | Type | Description |
|---|---|---|
| `relation` | string | Relation name for edges produced by this template |
| `template` | string | Prompt text. `{subject}` is replaced with the entity name |
| `multi_token` | bool | Chain multiple forward passes for multi-token answers |
| `reverse_template` | string? | Optional reverse probe (`{object}` placeholder) |
| `stop_tokens` | char[] | Characters that terminate multi-token chaining |

## Mock knowledge format

Used by `larql bfs --mock`. A JSON array:

```json
[
  {"prompt": "The capital of France is", "answer": "Paris", "probability": 0.89}
]
```
