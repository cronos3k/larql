# LARQL Weight Extraction Pipeline

End-to-end: model weights → vindex → queryable knowledge graph. No forward passes required for the bulk extraction. Residual capture uses targeted forward passes for seed entities only.

## 1. Build

```bash
make release
```

## 2. Extract a vindex

```bash
# Browse-only vindex (~3 GB at f16, enables DESCRIBE/WALK/SELECT)
larql extract-index google/gemma-3-4b-it -o output/gemma3-4b.vindex --f16

# With inference weights (~6 GB at f16, enables INFER)
larql extract-index google/gemma-3-4b-it -o output/gemma3-4b.vindex --level inference --f16

# Resume an interrupted build
larql extract-index google/gemma-3-4b-it -o output/gemma3-4b.vindex --f16 --resume
```

Accepts HuggingFace model IDs (resolved from `~/.cache/huggingface/hub/`) or local paths. Supports `--resume` on re-run.

## 3. Query the vindex

### Interactive REPL

```bash
larql repl
```

```sql
larql> USE "output/gemma3-4b.vindex";
larql> DESCRIBE "France";
larql> WALK "The capital of France is" TOP 10;
larql> INFER "The capital of France is" TOP 5;
```

### Single statement

```bash
larql lql 'USE "output/gemma3-4b.vindex"; DESCRIBE "France";'
```

## 4. Legacy extraction (NDJSON vectors)

For research and analysis, raw vectors can be extracted to NDJSON files:

```bash
# Edge graph (lexical layer, ~40 min)
larql weight-extract google/gemma-3-4b-it \
    -o output/gemma-3-4b-knowledge.larql.json \
    --stats output/gemma-3-4b-stats.json

# Vectors to NDJSON (all components, ~45 min)
larql vector-extract google/gemma-3-4b-it \
    -o output/vectors --resume
```

A vindex can also be built from these NDJSON files:

```bash
larql extract-index -o output/gemma3-4b.vindex --from-vectors output/vectors
```

## 5. Capture residuals (seed forward passes)

```bash
# L25 residuals for seed entities
larql residuals capture google/gemma-3-4b-it \
    --entities "France,Germany,Japan,Mozart,Einstein" \
    --layer 25 -o output/residuals-L25.vectors.ndjson
```

## 6. Query the edge graph (legacy)

```bash
larql query --graph output/gemma-3-4b-knowledge.larql.json France
larql describe --graph output/gemma-3-4b-knowledge.larql.json Mozart
larql stats output/gemma-3-4b-knowledge.larql.json
```

## Timing summary (Gemma 3-4B-IT on Apple Silicon Mac)

| Step | Time |
|---|---|
| Vindex extraction (browse, f16) | ~45 min |
| Weight walk (34 layers, 8.5M edges) | ~40 min |
| Vector extract (6 components, 1.29M vectors) | ~45 min |
| Residual capture (50 entities × 1 layer) | ~10 min |

## Commands used

| Command | What it does |
|---|---|
| `larql extract-index` | Build a .vindex from model weights |
| `larql repl` | Launch the LQL interactive REPL |
| `larql lql` | Execute a single LQL statement |
| `larql weight-extract` | Extract edges from FFN weights (zero forward passes) |
| `larql vector-extract` | Extract weight vectors to NDJSON |
| `larql residuals capture` | Forward passes for seed entities, capture hidden states |
| `larql attention-extract` | Extract edges from attention OV circuits |
| `larql stats` | Display graph statistics |
| `larql query` / `larql describe` | Query the edge graph |
