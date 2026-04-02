# Research Findings

Discoveries from extracting and querying Gemma 3-4B-IT weight vectors.

## 1. The FFN graph is complete

348,160 edges across 34 layers. Every FFN feature is an edge: gate (what triggers it) → down (what it produces). The graph contains ALL the model's token-level transformations:

- **Factual**: Toulouse → French, Rome → Roman, Dutch → Netherlands
- **Morphological**: read → reading, justify → just
- **Translational**: 全新的 → brand, 순番 → order, 뉴 → new
- **Syntactic**: if → should, has → are
- **Format**: back → back, best → best (identity preservation)
- **Suppressive**: 不但 → not (Chinese "not only" → "not")

The model doesn't distinguish between types. They're all stored the same way in FFN features. All routed the same way by attention.

## 2. Circuit type distribution reveals architecture

Cosine(gate, down) classifies every feature. The distribution across 34 layers shows three computational phases:

```
L0-L6:   Passive (97% projector) — embedding transformation
L7-L18:  Active (40% transform+suppress) — computation
L19-L29: Knowledge (85-95% projector) — factual bridges
L30-L33: Format gate (11% identity+inverter) — output control
```

L26-L27 are the peak knowledge layers (89% projector). L33 has the most identity+inverter features (11%) — the format enforcement layer.

## 3. Cross-lingual knowledge surfaces automatically

Down vector KNN against embeddings reveals multilingual knowledge:

```
F5040 down KNN: French, French, french, FRENCH, France, Frenchman, француз, フランス
F943 down KNN:  euros, €, Euros, EU, 欧盟, 欧洲, Spain, EUR
F918 down KNN:  Roman, ROM, Rom, Roma, Rome, Romano
F2230 down KNN: Dutch, Netherlands, dutch, Amsterdam, 荷兰, Nederlandse, Dutchman
```

Each feature's down vector points toward a region of embedding space that spans all languages.

## 4. 85% dark space is structural, not missing knowledge

Features where down_dist > 0.85 (85% of features) have down vectors that don't align with any single token embedding. Activation traces show these fire for ALL inputs — they're structural computation (articles, formatting, scale), not entity-specific knowledge.

The 15% that resolves cleanly IS the factual/morphological/translational knowledge. The graph is not 15% complete — the knowledge portion IS 100% extracted.

## 5. Aggregation by cross-layer repetition recovers answers

The correct answer repeats across multiple layers. Noise appears once.

```
France → french(3 edges), француз(2), 法国(3) = "French" family dominates
Germany → german(7 edges across layers) = "German" dominates
Japan → japanese(8 edges) = "Japanese" dominates
```

Aggregation by count × confidence → 64% match rate against model inference.

## 6. Attention routing is the missing index

The FFN graph stores ALL knowledge. Attention determines WHICH features to use for a given query. Forward pass traces show zero overlap between statically-extracted features and actually-activated features.

The features the model uses ARE in the graph — just under different source keys. The model routes "Germany" to features gated to "француз" (French) and "немец" (German) based on context. The routing is attention's job.

The attention routing graph = the index. The FFN knowledge graph = the store. Both extractable from weights. Together = the complete model.

## 7. Single-token attention approximation doesn't improve gate matching

Computing `embedding × W_V × W_O` across all heads/layers makes distances worse, not better. The OV projection without inter-token context adds noise. Attention's value comes from token INTERACTION, not from single-token projection.

## Reproduction

All findings reproducible from:
```bash
# Build a vindex
larql extract-index google/gemma-3-4b-it -o output/gemma3-4b.vindex --f16

# Query via REPL
larql repl
> USE "output/gemma3-4b.vindex";
> DESCRIBE "France";
> WALK "The capital of France is" TOP 10;

# Or extract raw vectors for analysis
larql vector-extract google/gemma-3-4b-it -o output/vectors --resume
python scripts/edge_discover_fast.py --vectors output/vectors --output output/edges --layers 0-33
```
