# LARQL CLI Reference

```
larql <COMMAND> [OPTIONS]
```

## Extraction commands

### `larql weight-extract`

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

### `larql attention-extract`

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

### `larql predict`

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

### `larql vector-extract`

Extract full weight vectors to intermediate NDJSON files for SurrealDB ingestion.

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

### `larql vector-load`

Load extracted vectors into SurrealDB with HNSW indexes.

```
larql vector-load <INPUT> --ns <NS> --db <DB> [OPTIONS]
```

| Flag | Description |
|---|---|
| `<INPUT>` | Directory containing `.vectors.jsonl` files |
| `--surreal <URL>` | SurrealDB endpoint [default: `http://localhost:8000`] |
| `--ns <NS>` | SurrealDB namespace |
| `--db <DB>` | SurrealDB database |
| `--user <USER>` | Username [default: `root`] |
| `--pass <PASS>` | Password [default: `root`] |
| `--tables <TABLES>` | Tables to load (comma-separated). Default: all |
| `--layers <LAYERS>` | Layers to load (comma-separated). Default: all |
| `--batch-size <N>` | Batch size for INSERT transactions [default: 500] |
| `--resume` | Resume interrupted load (skips completed layers) |
| `--schema-only` | Create schema only (no data load) |

**Examples:**

```bash
larql vector-load vectors/ --ns larql --db gemma3_4b
larql vector-load vectors/ --ns larql --db gemma3_4b --layers 25,26,33 --batch-size 1000
```

### `larql vector-import`

Import vectors into SurrealDB via batched `surreal import` CLI. Handles large tables (embeddings, FFN) that exceed HTTP body limits by writing temporary `.surql` batch files and importing each one.

```
larql vector-import <INPUT> --ns <NS> --db <DB> [OPTIONS]
```

| Flag | Description |
|---|---|
| `<INPUT>` | Directory containing `.vectors.jsonl` files |
| `--ns <NS>` | SurrealDB namespace |
| `--db <DB>` | SurrealDB database |
| `--tables <TABLES>` | Tables to import (comma-separated). Default: all |
| `--layers <LAYERS>` | Layers to import (comma-separated). Default: all |
| `--endpoint <URL>` | SurrealDB endpoint [default: `http://localhost:8000`] |
| `--user <USER>` | Username [default: `root`] |
| `--pass <PASS>` | Password [default: `root`] |
| `--batch-size <N>` | Records per batch file [default: 5000] |
| `--resume` | Skip completed layers (tracks progress in `load_progress` table) |

**Requires:** `surreal` CLI installed and SurrealDB running.

**Examples:**

```bash
# Import embeddings with progress bar
larql vector-import output/vectors --tables embeddings --ns larql --db gemma3_4b --resume

# Import FFN factual layers
larql vector-import output/vectors --tables ffn_gate,ffn_down \
    --layers 25,26,27,28,29,30,31,32,33 --ns larql --db gemma3_4b --resume

# Import everything
larql vector-import output/vectors --tables ffn_gate,ffn_down,ffn_up \
    --ns larql --db gemma3_4b --resume
```

### `larql vector-export-surql`

Export vectors to `.surql` files for manual `surreal import`. Useful if you want to inspect the SQL or import on a different machine.

```
larql vector-export-surql <INPUT> --output <OUTPUT> [OPTIONS]
```

| Flag | Description |
|---|---|
| `<INPUT>` | Directory containing `.vectors.jsonl` files |
| `-o, --output <OUTPUT>` | Output directory for `.surql` files |
| `--tables <TABLES>` | Tables to export (comma-separated). Default: all |
| `--layers <LAYERS>` | Layers to export (comma-separated). Default: all |
| `--ns <NS>` | Namespace for `USE` statement [default: `larql`] |
| `--db <DB>` | Database for `USE` statement [default: `gemma3_4b`] |

**Examples:**

```bash
larql vector-export-surql output/vectors -o output/surql --tables embeddings

# Then import manually
surreal import --endpoint http://localhost:8000 \
    --namespace larql --database gemma3_4b \
    --username root --password root \
    output/surql/embeddings.surql
```

### `larql residuals capture`

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
| `-o, --output <OUTPUT>` | Output NDJSON file |
| `--template <TEMPLATE>` | Prompt template. `{entity}` is replaced. Default: bare entity name |

**How it works:** Tokenizes each entity, runs a full forward pass through the transformer up to the target layer(s), and saves the last-token hidden state as a vector in NDJSON format. The output is compatible with `vector-load` for SurrealDB ingestion.

**Use case:** Seed SurrealDB with real residuals for a small set of entities (50–100). Then use SurrealDB vector similarity queries against gate vectors to discover factual edges for thousands of entities without additional forward passes.

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

**Output format:** Same NDJSON as `vector-extract`, loadable by `vector-load`:

```json
{"_header": true, "component": "residuals", "model": "google/gemma-3-4b-it", "dimension": 2560}
{"id": "France_L25", "layer": 25, "feature": 0, "vector": [...], "top_token": "Paris", "c_score": 12.4, ...}
```

### `larql bfs`

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

### `larql query`

Select edges from a subject, optionally filtered by relation.

```
larql query --graph <GRAPH> <SUBJECT> [RELATION]
```

```bash
larql query --graph knowledge.larql.json France
larql query --graph knowledge.larql.json France capital-of
```

### `larql describe`

Show all outgoing and incoming edges for an entity.

```
larql describe --graph <GRAPH> <ENTITY>
```

```bash
larql describe --graph knowledge.larql.json France
```

### `larql stats`

Show graph statistics: entity count, edge count, relation count, connected components, average degree, average confidence, source distribution.

```
larql stats <GRAPH>
```

```bash
larql stats knowledge.larql.json
```

### `larql validate`

Check a graph file for issues: zero-confidence edges, self-loops, empty subjects/objects.

```
larql validate <GRAPH>
```

```bash
larql validate knowledge.larql.json
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
