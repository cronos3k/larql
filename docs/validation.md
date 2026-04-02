# Graph Validation

## Extraction faithfulness

The knowledge graph is extracted directly from model weight vectors. Validation is not "does the graph match inference" but "does the graph faithfully represent what's in the weights."

This is proven by embedding KNN queries. Each clean edge was verified by:

1. **Gate KNN** — the gate vector's nearest embedding neighbor IS the trigger entity
2. **Down KNN** — the down vector's nearest embedding neighbors ARE the output concept

### Verified examples (Gemma 3-4B-IT, L26)

```
Feature   Gate KNN        Down KNN (top 5)                              Verified
────────────────────────────────────────────────────────────────────────────────────
F5040     Toulouse        French, French, french, FRENCH, France         ✓
F943      €               euros, €, Euros, EU, 欧盟, Spain, EUR          ✓
F918      Rome            Roman, ROM, Rom, Roma, Rome, Romano            ✓
F2230     Dutch           Dutch, Netherlands, dutch, Amsterdam, 荷兰     ✓
```

Each result is a cosine similarity search against 262,208 token embeddings. The down vector genuinely points toward the output concept in embedding space.

### What the graph captures vs what inference produces

| Graph edge | Graph answer | Model inference | Same knowledge? |
|---|---|---|---|
| Toulouse → French | French | "a city in southern France" | Yes — different expression |
| Rome → Roman | Roman | "Italian" | Yes — different facet |
| Dutch → Dutch | Dutch, Netherlands, Amsterdam | "Dutch" | Yes — exact match |
| € → euros | euros, EU, Spain | "the euro" | Yes — same concept |

The graph stores **associations** (what's connected in the weights). The model generates **text** (fluent continuation). Both encode the same underlying knowledge, expressed differently.

## Graph statistics

```
Total edges:           346,698 (all features, all layers)
Clean edges (< 0.8):   21,894 (down vector resolves to embedding space)
Factual entities:       ~14,000 unique subjects
Cross-lingual:         フランス, 荷兰, француз, Francia, Frankreich (multilingual)
```

## What the dark space IS

85% of features have `down_dist > 0.85` — their output direction doesn't align with any single token embedding. Analysis of activation traces shows these are **structural features** that fire for any input:

- Articles, formatting, syntax processing
- Scale normalization, routing
- Features that fire for EVERY entity, not specific ones

The dark space is the model's inference engine, not missing knowledge. The 15% that resolves cleanly IS the factual knowledge — a small fraction of total computation, which is expected.

## Circuit type validation

The 34-layer circuit type profile independently confirms the model architecture:

```
L0-L6:   Passive (97% projector) — embedding transformation
L7-L18:  Active (40% transform+suppress) — computation
L19-L29: Knowledge (85-95% projector) — factual bridges
L30-L33: Format gate (11% identity+inverter) — output control
```

This matches known transformer architecture findings without using any forward passes.

## Reproduction

```bash
# Build a vindex
larql extract-index google/gemma-3-4b-it -o output/gemma3-4b.vindex --f16

# Verify edges via REPL
larql repl
> USE "output/gemma3-4b.vindex";
> DESCRIBE "France";
> SELECT * FROM EDGES NEAREST TO "France" AT LAYER 26 LIMIT 20;

# Or extract raw vectors for batch analysis
larql vector-extract google/gemma-3-4b-it -o output/vectors --resume
python scripts/edge_discover_fast.py --vectors output/vectors --output output/edges --layers 0-33
```
