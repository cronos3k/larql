# larql — Python Bindings

Python interface to the LARQL knowledge graph engine and vindex model format. Rust-powered via PyO3, with numpy array interop and MLX integration.

## Install

```bash
cd crates/larql-python
maturin develop --release
```

## Quickstart

```python
import larql

# Load a vindex
vindex = larql.load("output/gemma3-4b-v2.vindex")

# Knowledge queries — instant, no inference
edges = vindex.describe("France")
for e in edges[:5]:
    print(f"  {e.relation} → {e.target}  score={e.gate_score:.0f}")

# Embed + KNN
embed = vindex.embed("France")           # numpy (2560,)
hits = vindex.entity_knn("France", layer=26, top_k=10)

# Insert knowledge — no training
vindex.insert("Colchester", "country", "England")

# Bulk gate vectors for research (SVD, PCA)
gates = vindex.gate_vectors(layer=26)     # numpy (10240, 2560)
```

## MLX Integration

Load MLX models directly from a vindex. No safetensors. Weights are mmap'd from vindex binaries.

```python
import larql
import mlx_lm

model, tokenizer = larql.mlx.load("output/gemma3-4b-v2.vindex")
response = mlx_lm.generate(model, tokenizer, prompt="The capital of France is", max_tokens=20)
print(response)
```

For best performance, extract with `--f16`:
```bash
larql extract-index "google/gemma-3-4b-it" -o model.vindex --level all --f16
```

## LQL Session

Full query language access alongside direct numpy API:

```python
session = larql.session("output/gemma3-4b-v2.vindex")

# LQL queries
session.query("DESCRIBE 'France'")
session.query("WALK 'The capital of France is' TOP 10")
session.query("STATS")

# Direct numpy access on the same session
session.vindex.gate_vectors(layer=26)
```

## API Reference

### Loading

| Function | Description |
|---|---|
| `larql.load(path)` | Load vindex, returns `Vindex` |
| `larql.session(path)` | Create LQL session with `.query()` and `.vindex` |
| `larql.mlx.load(path)` | Load MLX model from vindex weights |

### Vindex — Knowledge Queries

| Method | Description |
|---|---|
| `describe(entity, band="knowledge", verbose=False)` | Find all knowledge edges for an entity |
| `has_edge(entity, relation=None)` | Check if entity has edges |
| `get_target(entity, relation)` | Get target token for entity+relation |
| `relations()` | List all relation types with counts |
| `cluster_centre(relation)` | Relation direction vector as numpy |
| `typical_layer(relation)` | Most common layer for a relation |
| `stats()` | Model metadata as dict |

### Vindex — Feature Access

| Method | Returns |
|---|---|
| `embed(text)` | `numpy (hidden_size,)` — scaled, multi-token averaged |
| `gate_vector(layer, feature)` | `numpy (hidden_size,)` |
| `gate_vectors(layer)` | `numpy (num_features, hidden_size)` |
| `embedding(token_id)` | `numpy (hidden_size,)` — unscaled |
| `embedding_matrix()` | `numpy (vocab_size, hidden_size)` |
| `feature_meta(layer, feature)` | `FeatureMeta` or `None` |
| `feature(layer, feature)` | `dict` or `None` |
| `feature_label(layer, feature)` | `str` or `None` |
| `tokenize(text)` | `list[int]` |
| `decode(ids)` | `str` |

### Vindex — KNN & Walk

| Method | Description |
|---|---|
| `gate_knn(layer, query_vector, top_k=10)` | Raw KNN with vector |
| `entity_knn(entity, layer, top_k=10)` | Embed entity then KNN |
| `walk(residual, layers=None, top_k=5)` | Walk with raw vector |
| `entity_walk(entity, layers=None, top_k=5)` | Walk with entity string |

### Vindex — Mutation

| Method | Description |
|---|---|
| `insert(entity, relation, target, layer=None, confidence=0.8)` | Insert knowledge edge |
| `delete(entity, relation=None, layer=None)` | Delete matching edges |

### Vindex — Properties

| Property | Type |
|---|---|
| `num_layers` | `int` |
| `hidden_size` | `int` |
| `vocab_size` | `int` |
| `model` | `str` — model ID |
| `family` | `str` — architecture family |
| `is_mmap` | `bool` |
| `total_gate_vectors` | `int` |
| `loaded_layers` | `list[int]` |

### Session

| Method | Description |
|---|---|
| `query(lql)` | Execute LQL, returns `list[str]` |
| `query_text(lql)` | Execute LQL, returns joined string |
| `vindex` | Access underlying `Vindex` |

### Types

| Type | Key Fields |
|---|---|
| `DescribeEdge` | `relation`, `target`, `gate_score`, `layer`, `feature`, `source`, `confidence`, `also` |
| `WalkHit` | `layer`, `feature`, `gate_score`, `top_token`, `target`, `meta` |
| `FeatureMeta` | `top_token`, `top_token_id`, `c_score`, `top_k` |
| `Relation` | `name`, `cluster_id`, `count`, `top_tokens` |

## Project Structure

```
crates/larql-python/
  src/
    lib.rs              # Module registration, graph bindings
    vindex.rs           # PyVindex, describe, insert, relations
    session.rs          # PySession (LQL queries)
  python/larql/
    __init__.py         # Clean Python API
    mlx.py              # MLX model loading from vindex (mmap, zero-copy)
  tests/
    test_bindings.py    # 41 tests, synthetic vindex (no model dependency)
  examples/
    knowledge.py        # Describe, relations, steering
    insert.py           # Insert knowledge, no training
    session.py          # LQL session + numpy access
    mlx_vindex.py       # MLX generation from vindex weights
  bench/
    bench_bindings.py   # Load, KNN, walk, describe, MLX benchmarks
```

### Running Tests

```bash
# Synthetic vindex tests (run anywhere, no model files needed)
pytest crates/larql-python/tests/ -v

# With a real vindex (optional, more thorough)
REAL_VINDEX_PATH=output/gemma3-4b-v2.vindex pytest crates/larql-python/tests/ -v
```

## Not Yet Implemented

These are accessible via `session.query()` (LQL) or the CLI:

- HuggingFace loading (`hf://...`)
- Remote server connection
- `select()` queries
- `update()` mutation
- Patch API (begin/save/apply)
- Compile to safetensors/GGUF/MLX
