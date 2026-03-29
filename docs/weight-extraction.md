# LARQL Weight Extraction Pipeline

End-to-end: model weights → vectors → SurrealDB → knowledge graph. No forward passes required for the bulk extraction. Residual capture uses targeted forward passes for seed entities only.

## 1. Build

```bash
make release
```

## 2. Extract from weights

```bash
# Edge graph (lexical layer, ~40 min)
cargo run --release -p larql-cli -- weight-extract google/gemma-3-4b-it \
    -o output/gemma-3-4b-knowledge.larql.json \
    --stats output/gemma-3-4b-stats.json

# Vectors to NDJSON (all components, ~45 min)
cargo run --release -p larql-cli -- vector-extract google/gemma-3-4b-it \
    -o output/vectors --resume
```

Accepts HuggingFace model IDs (resolved from `~/.cache/huggingface/hub/`) or local paths. Both commands resume on re-run.

## 3. Start SurrealDB

```bash
# In memory (fast, volatile — good for quick exploration)
surreal start --log info --user root --pass root memory

# To disk (persistent — survives restarts)
surreal start --log info --user root --pass root surrealkv://data/surreal.db
```

## 4. Create schema

```bash
# Drop existing database (if re-running)
curl -s http://localhost:8000/sql \
    -H "surreal-ns: larql" -H "surreal-db: gemma3_4b" \
    -H "Authorization: Basic $(echo -n root:root | base64)" \
    --data "REMOVE DATABASE gemma3_4b;"

# Create schemas for all tables
cargo run --release -p larql-cli -- vector-load output/vectors \
    --ns larql --db gemma3_4b --schema-only
```

## 5. Load vectors into SurrealDB

### Small tables (seconds)

```bash
cargo run --release -p larql-cli -- vector-load output/vectors \
    --ns larql --db gemma3_4b --tables attn_ov,attn_qk
```

### Embeddings (~10 min)

```bash
cargo run --release -p larql-cli -- vector-import output/vectors \
    --tables embeddings --ns larql --db gemma3_4b --resume
```

### FFN tables — all layers (~30 min)

```bash
cargo run --release -p larql-cli -- vector-import output/vectors \
    --tables ffn_gate,ffn_down,ffn_up --ns larql --db gemma3_4b --resume
```

### Or just the factual layers (~7 min)

```bash
cargo run --release -p larql-cli -- vector-import output/vectors \
    --tables ffn_gate,ffn_down --layers 25,26,27,28,29,30,31,32,33 \
    --ns larql --db gemma3_4b --resume
```

All import commands support `--resume` — safe to interrupt and restart.

## 6. Capture residuals (seed forward passes)

```bash
# L25 residuals for seed entities
cargo run --release -p larql-cli -- residuals capture google/gemma-3-4b-it \
    --entities "France,Germany,Japan,Mozart,Einstein" \
    --layer 25 -o output/residuals-L25.vectors.ndjson

# Load into SurrealDB
cargo run --release -p larql-cli -- vector-import output/ \
    --tables residuals --ns larql --db gemma3_4b
```

## 7. Query in SurrealDB

### Interactive SQL shell

```bash
surreal sql --endpoint http://localhost:8000 --user root --pass root --ns larql --db gemma3_4b
```

This opens an interactive shell where you can run SurQL queries directly:

```sql
-- Count records per table
SELECT count() FROM embeddings GROUP ALL;
SELECT count() FROM ffn_gate GROUP ALL;
SELECT count() FROM attn_ov GROUP ALL;

-- Look up an embedding
SELECT id, token, c_score FROM embeddings WHERE token = 'France';

-- Find gate features that fire for France's embedding
LET $france = (SELECT vector FROM embeddings WHERE token = 'France' LIMIT 1);
SELECT id, layer, feature, top_token, c_score,
       vector::similarity::cosine(vector, $france[0].vector) AS similarity
FROM ffn_gate
WHERE layer = 26
ORDER BY similarity DESC
LIMIT 20;
```

### Via curl

```bash
curl -s http://localhost:8000/sql \
    -H "surreal-ns: larql" -H "surreal-db: gemma3_4b" \
    -H "Authorization: Basic $(echo -n root:root | base64)" \
    --data "SELECT count() FROM embeddings GROUP ALL;"
```

See `surql/queries/` for more example queries.

## 8. Query the edge graph

```bash
larql query --graph output/gemma-3-4b-knowledge.larql.json France
larql describe --graph output/gemma-3-4b-knowledge.larql.json Mozart
larql stats output/gemma-3-4b-knowledge.larql.json
```

## Timing summary (Gemma 3-4B-IT on Apple Silicon Mac)

| Step | Time |
|---|---|
| Weight walk (34 layers, 8.5M edges) | ~40 min |
| Vector extract (6 components, 1.29M vectors) | ~45 min |
| Load embeddings (262K vectors) | ~10 min |
| Load FFN all layers (gate + down + up, ~1M vectors) | ~30 min |
| Load FFN factual layers only (L25-L33) | ~7 min |
| Load attention tables (544 vectors) | seconds |
| Residual capture (50 entities × 1 layer) | ~10 min |

**Full pipeline:** ~2.5 hours from zero to everything in SurrealDB.
**Workshop-ready subset** (embeddings + factual FFN + attention): ~1 hour.

## Commands used

| Command | What it does |
|---|---|
| `larql weight-extract` | Extract edges from FFN weights (zero forward passes) |
| `larql vector-extract` | Extract weight vectors to NDJSON |
| `larql vector-load` | Create SurrealDB schema + load small tables via HTTP |
| `larql vector-import` | Batch import via `surreal import` CLI (for large tables) |
| `larql vector-export-surql` | Export NDJSON to `.surql` files (manual import) |
| `larql residuals capture` | Forward passes for seed entities, capture hidden states |
| `larql attention-extract` | Extract edges from attention OV circuits |
| `larql stats` | Display graph statistics |
| `larql query` / `larql describe` | Query the edge graph |
