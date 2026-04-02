# Circuit Type Analysis

Every FFN feature has a gate vector (what activates it) and a down vector (what it produces). The cosine similarity between them classifies the feature's circuit role.

## Circuit types

| Type | Cosine range | Behaviour |
|---|---|---|
| **Identity** | > 0.5 | Reads X, writes X back (self-reinforcement) |
| **Transform** | 0.2 – 0.5 | Reads X, writes a related form (morphological, syntactic) |
| **Projector** | -0.2 – 0.2 | Reads X, writes something unrelated (factual bridge) |
| **Suppressor** | -0.5 – -0.2 | Weak direction flip (gating, interference) |
| **Inverter** | < -0.5 | Strong direction flip (format enforcement, suppression) |

## Layer architecture (Gemma 3-4B-IT)

Discovered from cosine(gate, down) on extracted weight vectors. No forward passes. Full 34-layer profile computed in ~5 minutes.

```
Layer  Proj    Trans   Supp    Ident   Inv     Role
──────────────────────────────────────────────────────
L0     97.2%   1.4%    1.4%    0.0%    0.0%    passive
L1     96.1%   1.7%    1.7%    0.3%    0.2%    passive
L2     94.9%   2.3%    2.6%    0.1%    0.1%    ↓
L3     86.6%   6.5%    6.7%    0.1%    0.1%    rising
L4     84.0%   7.6%    7.8%    0.3%    0.4%
L5     74.1%   12.8%   12.6%   0.3%    0.3%    ↓
L6     76.3%   11.0%   11.1%   0.8%    0.8%
L7     63.8%   17.6%   17.0%   0.9%    0.7%    ACTIVE
L8     59.1%   19.8%   19.7%   0.6%    0.8%    ACTIVE ←peak1
L9     56.2%   21.3%   20.8%   0.9%    0.8%    ACTIVE ←peak1
L10    57.1%   20.4%   19.5%   1.6%    1.4%    ACTIVE
L11    58.2%   20.1%   20.0%   0.8%    0.8%    ACTIVE
L12    62.9%   17.6%   18.4%   0.6%    0.5%    ACTIVE
L13    62.7%   18.1%   17.3%   1.0%    1.0%    ACTIVE
L14    65.0%   16.1%   16.7%   1.1%    1.1%    ACTIVE
L15    56.4%   20.6%   20.6%   1.2%    1.2%    ACTIVE ←peak2
L16    55.1%   22.3%   21.2%   0.6%    0.8%    ACTIVE ←peak2
L17    62.3%   18.5%   17.5%   0.9%    0.8%    ACTIVE
L18    64.7%   16.9%   16.8%   0.9%    0.7%    ↓
L19    65.2%   16.6%   17.0%   0.7%    0.6%    winding down
L20    69.9%   13.9%   14.7%   0.7%    0.8%
L21    75.4%   11.4%   11.9%   0.7%    0.6%
L22    74.9%   11.4%   11.4%   1.1%    1.2%
L23    82.3%   7.9%    8.2%    0.7%    0.9%    ↓
L24    84.5%   6.7%    6.7%    1.0%    1.1%    knowledge
L25    83.1%   6.8%    7.4%    1.3%    1.4%    knowledge
L26    88.5%   4.8%    4.9%    0.9%    0.9%    KNOWLEDGE ←peak
L27    89.7%   4.1%    4.6%    0.6%    1.0%    KNOWLEDGE
L28    92.6%   3.4%    3.3%    0.3%    0.3%    knowledge
L29    95.3%   2.1%    2.4%    0.1%    0.1%    passive
L30    87.7%   5.6%    5.7%    0.4%    0.5%    ↑
L31    80.2%   7.9%    8.2%    1.9%    1.8%    rising
L32    79.6%   7.3%    7.2%    2.8%    3.1%    id+inv rising
L33    59.6%   14.4%   15.0%   5.7%    5.3%    FORMAT GATE
```

### Three phases

**Phase 1: Computation (L7–L18)** — Two activity peaks at L8-9 and L15-16. Transform and suppressor both exceed 20%. This is where the model actively processes inputs — classifying, routing, gating representations.

**Phase 2: Knowledge (L23–L29)** — Projector rises to 85-95%. Minimal active computation. Features bridge between entity and attribute subspaces. L26-27 are the peak knowledge layers (89-90% projector).

**Phase 3: Format gate (L30–L33)** — Identity and inverter spike together. L33 has 5.7% identity + 5.3% inverter = 11% active allow/suppress. Features preserve approved outputs and invert suppressed alternatives.

### L26 — Knowledge bridge peak
88.5% projector. Gate-side queries show clean geographic/semantic clusters (France → Toulouse, París, Italia, €). Features connect entity subspaces to attribute subspaces.

### L33 — Format enforcement
59.6% projector, but 11% identity+inverter — the highest of any layer. Top inverters show character-level suppression (Y→S, sixteen→7). The allow/suppress pair confirms format enforcement from forward pass experiments.

### L33 — Format enforcement
35% active. Highest identity (5.7%) AND inverter (5.3%) of any layer. The allow/suppress pair: identity features preserve approved outputs, inverter features flip suppressed alternatives. Top inverters show character-level suppression (Y→S, sixteen→7, Pherson→M).

## Discovery method

```bash
# Build a vindex
larql extract-index google/gemma-3-4b-it -o output/gemma3-4b.vindex --f16

# Query via REPL
larql repl
> USE "output/gemma3-4b.vindex";
> DESCRIBE "France";
> SHOW FEATURES AT LAYER 26;
```

Or extract raw vectors for analysis:

```bash
larql vector-extract google/gemma-3-4b-it -o output/vectors --resume
python scripts/edge_discover_fast.py --vectors output/vectors --output output/edges --layers 0-33
```

## Key insight

Projector (cosine ≈ 0) is the default — most features at every layer have near-orthogonal gate and down vectors. The model stores input and output directions independently.

The signal is in the **non-projector** features:
- **Identity** features are deliberately preserved directions — what the model chose to keep intact
- **Inverter** features are deliberately flipped — what the model chose to suppress
- **Transform** features are active computation — morphological, syntactic, semantic transforms

The ratio of identity to inverter at each layer reveals the layer's computational role. L33's 5.7% identity + 5.3% inverter = 11% active allow/suppress — the format gate.
