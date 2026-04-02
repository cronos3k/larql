# WalkFfn — Vindex Gate KNN with Sparse FFN

**File:** `crates/larql-inference/src/vindex/walk_ffn.rs`
**Status:** Production
**Speed:** Lossless at K=8092 (97.91% on France→Paris)
**Accuracy:** Proven equivalent to dense for factual queries

## Description

The production FFN backend for LARQL inference. Uses the vindex gate KNN for feature selection,
then runs sparse FFN computation on only the selected features. Captures a walk trace showing
which features activated and what they mean.

This is the backend used by the LQL `INFER` statement.

## Architecture

```
Input x (post-attention residual)
  │
  ├─► GateIndex::gate_knn(layer, x_last, top_k)  →  feature selection
  │     Uses VectorIndex or PatchedVindex (both implement GateIndex)
  │
  └─► sparse_ffn_forward(weights, layer, x, features)  →  sparse FFN output
        Only computes gate/up/down for selected features
```

The `GateIndex` trait abstracts over both `VectorIndex` (base, readonly) and `PatchedVindex`
(with overlay). This means INSERT/DELETE/UPDATE to the vindex immediately affect inference
output — patched gate vectors are used for feature selection.

## Walk Trace

Each layer's trace contains:
- **Feature ID** — which FFN feature activated
- **Gate score** — how strongly it activated
- **Down meta** — what token this feature predicts (from the vindex)

Example for "The capital of France is":
```
L27: F9515  gate=+9.247  hears="Paris"   c=0.05
L26: F5040  gate=+7.880  hears="French"  c=0.08
L28: F8200  gate=-5.297  hears="France"  c=0.08
```

## Usage

```rust
use larql_inference::vindex::WalkFfn;

// Works with VectorIndex (unpatched)
let walk_ffn = WalkFfn::new(weights, &index, top_k);

// Works with PatchedVindex (mutations visible)
let walk_ffn = WalkFfn::new(weights, &patched, top_k);

let result = predict_with_ffn(weights, tokenizer, &token_ids, 5, &walk_ffn);
let trace = walk_ffn.take_trace(); // interpretability layer
```

## LQL

```sql
INFER "The capital of France is" TOP 5;
EXPLAIN INFER "The capital of France is" TOP 5;
```
