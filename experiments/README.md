# Experiments

Hypothesis-driven experiments using the vindex Python bindings.
Each directory tests one idea. Results go in `results/`.

## Setup

```bash
cd crates/larql-python
maturin develop --release
```

```python
import larql
vindex = larql.load_vindex("output/gemma3-4b-v2.vindex")
```

## Experiments

### 01 — Gate Synthesis
Can you synthesise a gate vector from scratch and have it match a forward pass residual?
Compare heuristic synthesis (entity_embed * scale + relation_centre * weight) vs captured residual.

### 02 — Manifold Dimensionality
What's the true rank of the knowledge manifold? SVD of all gate vectors from knowledge layers.
If 99% variance in 15D, compress 71 GB to 416 MB.

### 03 — Build Knowledge Layer
Can you construct L14-27 from Wikidata triples? Embed entities, assign to layers by relation type,
write gate+down vectors. Run INFER — does "The capital of France is" produce Paris?
