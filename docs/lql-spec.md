# LQL — Lazarus Query Language Specification

**Version:** 0.3  
**Author:** Chris Hay  
**Date:** 2026-03-31  
**Status:** Draft  
**Implementation target:** `larql-lql` crate (Rust)  
**Companion:** `larql-knowledge` spec (data pipeline)

---

## 1. Design Principles

LQL is a query language for neural network weights treated as a graph database. It is not SQL. It is not SPARQL. It borrows from both but serves a different purpose: decompiling, inspecting, editing, and recompiling neural networks.

**Principles:**

1. **Weights are rows.** Every W_gate row is a record. Every W_down column is a record. Every embedding vector is a record. The model IS the database.
2. **Two backends, one language.** LQL operates on either a `.vindex` (pre-extracted, fast) or directly on model weights via safetensors (live, no extraction needed). The vindex is preferred for production — sub-millisecond lookups from a pre-built index. Direct weight access is for exploration — point at any model, start querying immediately. Same statements, same results, different performance.
3. **Statements, not scripts.** Each LQL statement is self-contained. No variables that persist across statements (except `USE` context). Pipe results with `|>`.
4. **Three verbs for the demo.** The video needs exactly: `EXTRACT`, `DESCRIBE`, `INSERT`, `COMPILE`. Everything else is power-user.
5. **Rust-native.** The parser lives in `larql-lql`. No Python dependency. No runtime. One binary.
6. **Labels are external.** Relation labels come from the `larql-knowledge` project (probes, Wikidata triples, WordNet, AST pairs). The engine reads label files. It does not contain ingestion or probing code.

---

## 2. Statement Categories

### 2.1 Model Lifecycle

| Statement | Purpose |
|---|---|
| `EXTRACT` | Decompile model weights → vindex |
| `COMPILE` | Recompile vindex → model weights |
| `DIFF` | Compare two vindexes |
| `USE` | Set active vindex / model context |

### 2.2 Knowledge Browser (pure vindex, no model needed)

| Statement | Purpose |
|---|---|
| `WALK` | Feature scan — what gate features fire for a token's embedding |
| `SELECT` | Query edges by entity, relation, layer |
| `DESCRIBE` | Show all knowledge for an entity, grouped by layer band |
| `EXPLAIN WALK` | Feature trace without attention (pure vindex) |

### 2.3 Inference (requires model weights in vindex)

| Statement | Purpose |
|---|---|
| `INFER` | Full forward pass with attention — actual next-token prediction |
| `EXPLAIN INFER` | Full inference with per-layer feature trace |

### 2.4 Knowledge Mutation

| Statement | Purpose |
|---|---|
| `INSERT` | Add edge(s) to the vindex |
| `DELETE` | Remove edge(s) from the vindex |
| `UPDATE` | Modify existing edge(s) |
| `MERGE` | Merge edges from another vindex |

### 2.5 Patches

| Statement | Purpose |
|---|---|
| `BEGIN PATCH` | Start recording edits into a patch file |
| `SAVE PATCH` | Save and close the current patch |
| `APPLY PATCH` | Apply a patch to the current vindex (stacks) |
| `REMOVE PATCH` | Remove a patch from the stack |
| `SHOW PATCHES` | List active patches |
| `DIFF ... INTO PATCH` | Extract diff between two vindexes as a patch |
| `COMPILE ... INTO VINDEX` | Bake patches into a new clean vindex |

### 2.6 Schema Introspection

| Statement | Purpose |
|---|---|
| `SHOW RELATIONS` | List discovered relation types |
| `SHOW LAYERS` | Layer-by-layer summary |
| `SHOW FEATURES` | Feature details at a layer |
| `SHOW MODELS` | List available vindexes |
| `STATS` | Counts, coverage, size |

---

## 3. Grammar

### 3.1 Notation

```
UPPERCASE  = keyword (case-insensitive in parser)
<name>     = required parameter
[name]     = optional parameter
{a | b}    = choice
...        = repeatable
```

### 3.2 Model Lifecycle Statements

```
EXTRACT MODEL <model_id> INTO <vindex_path>
    [COMPONENTS <component_list>]
    [LAYERS <range>]
    [WITH {INFERENCE | ALL}]

-- Decompile a HuggingFace model (or local path) into a vindex.
-- 
-- Default (no WITH clause): builds the browse-only vindex.
--   Extracts: gate_vectors, embeddings, down_meta, labels
--   Enables: WALK, DESCRIBE, SELECT, EXPLAIN WALK
--   Size: ~3 GB (f16)
--
-- WITH INFERENCE: adds attention weights for INFER.
--   Adds: attn_weights (Q, K, V, O per layer)
--   Enables: + INFER, EXPLAIN INFER
--   Size: ~6 GB (f16)
--
-- WITH ALL: adds all weights for COMPILE.
--   Adds: + up_weights, norms, lm_head
--   Enables: + COMPILE
--   Size: ~10 GB (f16)
--
-- Layers: 0-33 (default: all)
-- No data is duplicated — gate_vectors IS W_gate, embeddings IS W_embed.

EXTRACT MODEL "google/gemma-3-4b-it"
    INTO "gemma3-4b.vindex";
-- Browse-only: ~3 GB

EXTRACT MODEL "google/gemma-3-4b-it"
    INTO "gemma3-4b.vindex"
    WITH INFERENCE;
-- Browse + inference: ~6 GB

EXTRACT MODEL "google/gemma-3-4b-it"
    INTO "gemma3-4b.vindex"
    WITH ALL;
-- Full: ~10 GB, supports COMPILE
```

```
COMPILE {<vindex_path> | CURRENT} INTO MODEL <output_path>
    [FORMAT {safetensors | gguf}]

-- Recompile a vindex back to model weights.
-- Round-trip: EXTRACT then COMPILE should produce identical weights.

COMPILE CURRENT
    INTO MODEL "gemma3-4b-edited/"
    FORMAT safetensors;
```

```
DIFF <vindex_a> {<vindex_b> | CURRENT}
    [LAYER <n>]
    [RELATION <type>]
    [LIMIT <n>]

-- Show edges that differ between two vindexes.

DIFF "gemma3-4b.vindex" CURRENT;

DIFF "gemma3-4b.vindex" "gemma3-4b-edited.vindex"
    RELATION "capital"
    LIMIT 20;
```

```
USE <vindex_path>;
USE MODEL <model_id> [AUTO_EXTRACT];

-- Set the active backend for subsequent statements.
-- USE with a .vindex path: fast, pre-extracted, all statements available.
-- USE MODEL: live weight access, reads safetensors directly. Slower but
--   zero setup — point at any model, start querying immediately.
-- AUTO_EXTRACT: create vindex automatically on first mutation.
--
-- Mutation (INSERT/DELETE/UPDATE) and COMPILE require a vindex.
-- INFER requires model weights (either via --include-weights or USE MODEL).

USE "gemma3-4b.vindex";

USE MODEL "google/gemma-3-4b-it";

USE MODEL "google/gemma-3-4b-it" AUTO_EXTRACT;
```

### 3.3 Knowledge Browser Statements

```
WALK <prompt>
    [TOP <n>]
    [LAYERS {<range> | ALL}]
    [MODE {hybrid | pure | dense}]
    [COMPARE]

-- Feature scan: what gate features fire for the last token's embedding.
-- This is a knowledge browser operation, NOT inference.
-- No attention is used. The query is the raw token embedding.
-- Returns per-layer top features with relation labels and gate scores.

WALK "France" TOP 10;

WALK "The capital of France is"
    TOP 10
    LAYERS 24-33;
```

```
DESCRIBE <entity>
    [{ALL LAYERS | SYNTAX | KNOWLEDGE | OUTPUT}]
    [AT LAYER <n>]
    [RELATIONS ONLY]

-- Show all knowledge for an entity. Groups by layer band:
--   SYNTAX:     morphological, syntactic, code structure features
--   KNOWLEDGE:  semantic/factual features (default view)
--   OUTPUT:     formatting/output features
--   ALL LAYERS: show all three bands
--
-- Layer band boundaries are model-specific, stored in index.json:
--   Gemma 3 4B:  syntax=0-13, knowledge=14-27, output=28-33
--   Llama 3 8B:  syntax=0-15, knowledge=16-28, output=29-31
--   (boundaries discoverable via SHOW LAYERS)
--
-- Labels come from multiple sources (highest priority first):
--   1. Probe-confirmed (model inference verified this feature)
--   2. Wikidata output matching (L14-27 only)
--   3. WordNet output matching (L0-13 only)
--   4. AST output matching (L0-13 only)
--   5. Entity pattern detection (country, language, month, number)
--   6. Morphological detection (short suffixes/prefixes)
--   7. TF-IDF top tokens (fallback)
--
-- Each edge shows: relation label, target token, gate score,
-- layer range, occurrence count, and "also:" variant tokens.

DESCRIBE "France";
-- Shows knowledge edges (L14-27) by default

DESCRIBE "France" ALL LAYERS;
-- Shows syntax (L0-13) + knowledge (L14-27) + output (L28-33)

DESCRIBE "def" SYNTAX;
-- Shows L0-13 features only — code/linguistic structure

DESCRIBE "France" KNOWLEDGE;
-- Explicit: same as default

DESCRIBE "France" OUTPUT;
-- Shows L28-33 only — output formatting features

DESCRIBE "Mozart" AT LAYER 26;
-- Single layer

DESCRIBE "France" RELATIONS ONLY;
-- Only show edges with a confirmed relation label (probe or matched)
```

```
SELECT [<fields>]
    FROM EDGES
    [NEAREST TO <entity> AT LAYER <n>]
    [WHERE <conditions>]
    [ORDER BY <field> {ASC | DESC}]
    [LIMIT <n>]

-- Query edges in the vindex.
-- Fields: entity, relation, target, layer, feature, confidence, gate

SELECT entity, relation, target, confidence
    FROM EDGES
    WHERE entity = "France"
    ORDER BY confidence DESC
    LIMIT 10;

SELECT entity, target
    FROM EDGES
    WHERE relation = "capital"
    AND confidence > 0.5;

SELECT *
    FROM EDGES
    WHERE layer = 27
    AND feature = 9515;

SELECT entity, target, distance
    FROM EDGES
    NEAREST TO "Mozart"
    AT LAYER 26
    LIMIT 20;
```

```
EXPLAIN WALK <prompt>
    [LAYERS <range>]
    [VERBOSE]
    [TOP <n>]

-- Show the per-layer feature trace from a pure vindex walk.
-- No attention. Each layer shows top-K features that fire,
-- with relation labels, gate scores, and output tokens.

EXPLAIN WALK "The capital of France is";

EXPLAIN WALK "The capital of France is"
    LAYERS 24-33 TOP 3 VERBOSE;
```

### 3.4 Inference Statements

```
INFER <prompt>
    [TOP <n>]
    [COMPARE]

-- Full forward pass with attention. Requires attention weights.
-- Vindex must be built with WITH INFERENCE or WITH ALL.
-- Or use USE MODEL for live weight access.
-- Uses walk FFN (gate KNN from vindex) as the FFN backend,
-- but runs real attention for token routing.
-- COMPARE: also run dense inference and show both.

INFER "The capital of France is" TOP 5;

INFER "The capital of France is" TOP 5 COMPARE;
```

```
EXPLAIN INFER <prompt>
    [LAYERS <range>]
    [VERBOSE]
    [TOP <n>]

-- Full inference with per-layer feature trace.
-- Shows which features fire WITH attention context.
-- Requires model weights.

EXPLAIN INFER "The capital of France is" TOP 5;
```

### 3.5 Knowledge Mutation Statements

```
INSERT INTO EDGES
    (entity, relation, target)
    VALUES (<entity>, <relation>, <target>)
    [AT LAYER <n>]
    [CONFIDENCE <float>]

-- Add an edge to the vindex.
-- The relation should be one of the known types (from probe/cluster labels).
-- AT LAYER: which layer to write to (default: auto-select based on relation).
-- CONFIDENCE: injection strength (default: auto-calibrate).

INSERT INTO EDGES
    (entity, relation, target)
    VALUES ("John Coyle", "lives-in", "Colchester");

INSERT INTO EDGES
    (entity, relation, target)
    VALUES ("John Coyle", "occupation", "engineer")
    AT LAYER 26 CONFIDENCE 0.8;
```

```
DELETE FROM EDGES
    WHERE <conditions>

DELETE FROM EDGES
    WHERE entity = "John Coyle"
    AND relation = "lives-in";
```

```
UPDATE EDGES
    SET <field> = <value> [, <field> = <value>]...
    WHERE <conditions>

UPDATE EDGES
    SET target = "London"
    WHERE entity = "John Coyle"
    AND relation = "lives-in";
```

```
MERGE <source_vindex>
    [INTO <target_vindex>]
    [ON CONFLICT {KEEP_SOURCE | KEEP_TARGET | HIGHEST_CONFIDENCE}]

MERGE "medical-knowledge.vindex"
    INTO "gemma3-4b.vindex"
    ON CONFLICT HIGHEST_CONFIDENCE;
```

```
BEGIN PATCH <patch_path>

-- Start a patch session. All subsequent INSERT/DELETE/UPDATE operations
-- are captured into the patch file. The base vindex is NOT modified.
-- End with SAVE PATCH.

BEGIN PATCH "medical-knowledge.vlp";
```

```
SAVE PATCH

-- Save the current patch session to disk and end patch mode.
-- The base vindex remains unchanged.

SAVE PATCH;
-- Saved: medical-knowledge.vlp (3 operations, 30 KB)
```

```
APPLY PATCH <patch_path>

-- Apply a patch to the current vindex. Patches stack in order.
-- The base vindex files are not modified — patches are an overlay.

APPLY PATCH "medical-knowledge.vlp";
APPLY PATCH "fix-hallucinations.vlp";
```

```
REMOVE PATCH <patch_path>

-- Remove a previously applied patch from the stack.

REMOVE PATCH "fix-hallucinations.vlp";
```

```
SHOW PATCHES

-- List all currently applied patches.

SHOW PATCHES;
-- 1. medical-knowledge.vlp     (3 inserts, 30 KB)
-- 2. company-facts.vlp         (200 inserts, 2 MB)
```

```
DIFF <vindex_a> <vindex_b>
    INTO PATCH <patch_path>

-- Extract the difference between two vindexes as a portable patch file.

DIFF "gemma3-4b.vindex" "gemma3-4b-medical.vindex"
    INTO PATCH "medical-changes.vlp";
```

```
COMPILE CURRENT INTO VINDEX <output_path>
    [ON CONFLICT {LAST_WINS | HIGHEST_CONFIDENCE | FAIL}]

-- Flatten all applied patches into a new clean vindex.
-- The result is a standalone vindex with no patch dependencies.

COMPILE CURRENT INTO VINDEX "gemma3-4b-medical.vindex";
```

### 3.6 Schema Introspection Statements

```
SHOW RELATIONS
    [AT LAYER <n>]
    [WITH EXAMPLES]

SHOW LAYERS
    [RANGE <start>-<end>]
    [{<start>-<end>}]

SHOW FEATURES <layer>
    [WHERE <conditions>]
    [LIMIT <n>]

SHOW MODELS;

SHOW PATCHES;

STATS [<vindex_path>];
```

### 3.7 Comparison Operators

```
=    -- equal
!=   -- not equal
>    -- greater than
<    -- less than
>=   -- greater than or equal
<=   -- less than or equal
LIKE -- pattern match (% wildcard)
IN   -- set membership

WHERE entity = "France"
WHERE confidence > 0.5
WHERE entity LIKE "Fran%"
WHERE entity IN ("France", "Germany")
WHERE layer >= 20 AND layer <= 30
```

---

## 4. Backend Architecture

LQL abstracts over two backends through a common trait. Every query statement works against either backend.

### 4.1 The Two Backends

```
┌──────────────────────────────────────────────────────────────┐
│                        LQL Parser                            │
│         (same AST, same statements, same output)             │
└───────────────────────┬──────────────────────────────────────┘
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
   ┌──────────────────┐  ┌──────────────────┐
   │  VindexBackend   │  │  WeightBackend   │
   │                  │  │                  │
   │  Pre-extracted   │  │  Live safetensors│
   │  KNN index       │  │  Dense matmul    │
   │  0.98ms/layer    │  │  ~6ms/layer      │
   │  Read + write    │  │  Read only       │
   │  No model needed │  │  Model in memory │
   │  (model optional │  │                  │
   │   for INFER)     │  │                  │
   └──────────────────┘  └──────────────────┘
```

### 4.2 Backend Capabilities

| Statement | Vindex | Direct Weights |
|---|---|---|
| WALK (feature scan) | ✅ KNN (0.98ms/layer) | ✅ Dense matmul (~6ms/layer) |
| DESCRIBE | ✅ Pre-computed edges + labels | ✅ On-the-fly per entity |
| SELECT | ✅ Index lookup | ✅ Live gate×embedding scan |
| EXPLAIN WALK | ✅ Walk trace from index | ✅ Walk trace from matmul |
| INFER | ✅ With `--include-weights` | ✅ Full forward pass |
| EXPLAIN INFER | ✅ With `--include-weights` | ✅ Full forward pass + trace |
| SHOW RELATIONS | ✅ From label cache | ✅ Cluster on-the-fly (slow) |
| SHOW LAYERS | ✅ From metadata | ✅ Computed from weights |
| SHOW FEATURES | ✅ Index lookup | ✅ Dense scan per layer |
| STATS | ✅ Instant | ✅ Computed |
| INSERT | ✅ | ❌ Error: "requires vindex" |
| DELETE | ✅ | ❌ Error: "requires vindex" |
| UPDATE | ✅ | ❌ Error: "requires vindex" |
| BEGIN/SAVE/APPLY PATCH | ✅ | ❌ Error: "requires vindex" |
| SHOW PATCHES | ✅ | ❌ |
| COMPILE | ✅ | ❌ Error: "requires vindex" |
| DIFF | ✅ | ⚠️ One side can be weights |
| MERGE | ✅ | ❌ Error: "requires vindex" |

### 4.3 Promotion: Weights → Vindex

Direct weight access is the on-ramp. When someone hits a mutation, LQL nudges:

```
larql> USE MODEL "google/gemma-3-4b-it";
Using model: google/gemma-3-4b-it (8.1 GB, live weights)

larql> INSERT INTO EDGES (entity, relation, target)
   ...   VALUES ("John Coyle", "lives-in", "Colchester");
Error: INSERT requires a vindex. Run:
  EXTRACT MODEL "google/gemma-3-4b-it" INTO "gemma3-4b.vindex";
  USE "gemma3-4b.vindex";
```

---

## 5. Label Architecture

Feature labels come from the `larql-knowledge` project and are stored in the vindex as JSON files. The engine reads them; it does not produce them.

### 5.1 Label Sources (Priority Order)

| Priority | Source | Confidence | Layer Band | Description |
|----------|--------|------------|------------|-------------|
| 1 | Probe-confirmed | Highest | Knowledge | Model inference confirmed this feature encodes this relation |
| 2 | Wikidata output matching | High | Knowledge | Cluster outputs match Wikidata objects |
| 3 | WordNet output matching | High | Syntax | Cluster outputs match WordNet pairs |
| 4 | AST output matching | High | Syntax | Cluster outputs match code AST pairs |
| 5 | Entity pattern detection | Medium | Any | Cluster members match known lists (country, language, month, number) |
| 6 | Morphological detection | Medium | Syntax | Cluster members are short suffixes/prefixes |
| 7 | TF-IDF top tokens | Low | Any | Fallback: most distinctive tokens in the cluster |

### 5.2 Vindex File Layout

The vindex IS the model, reorganised for queryability. No data is duplicated — each weight matrix is stored once in its optimal format.

```
gemma3-4b.vindex/

  # ═══ Query Index (the vindex core) ═══
  # These are the model's FFN + embedding weights in a queryable format.
  # WALK, DESCRIBE, SELECT, EXPLAIN WALK use only these files.
  
  gate_vectors.bin            # W_gate rows per layer (KNN index)
                              # ~3.3 GB (f32) or ~1.7 GB (f16)
  
  embeddings.bin              # W_embed matrix (token lookup)
                              # ~2.5 GB (f32) or ~1.3 GB (f16)
  
  down_meta.bin               # W_down decoded: per-feature top token IDs + scores
                              # ~2 MB (binary) — replaces 160 MB JSONL
  
  # ═══ Inference Weights (for INFER, not duplicated) ═══
  # Only the weights NOT already in the query index.
  # INFER loads these in addition to the query index.
  
  attn_weights.bin            # Q, K, V, O attention matrices per layer
                              # ~3 GB (f16)
  
  # ═══ Compile Weights (for COMPILE, not duplicated) ═══
  # Additional weights needed to reconstruct full safetensors.
  
  up_weights.bin              # W_up per layer
                              # ~3.3 GB (f16)
  
  norms.bin                   # LayerNorm parameters per layer
                              # ~1 MB
  
  lm_head.bin                 # Output projection (unembed)
                              # ~1.3 GB (f16) — or shared with embeddings if tied
  
  # ═══ Metadata & Labels ═══
  
  index.json                  # Config: layers, hidden_size, vocab_size,
                              # component manifest, build info
  tokenizer.json              # Tokenizer (~32 MB)
  relation_clusters.json      # Cluster centres, labels, counts
  feature_clusters.jsonl      # Per-feature cluster assignments
  feature_labels.json         # Probe-confirmed labels (from larql-knowledge)
```

**Size by use case:**

| Use Case | Files Loaded | Size (f16) | Size (f32 current) |
|----------|-------------|------------|-------------------|
| Browse only (WALK, DESCRIBE, SELECT) | gate + embed + down_meta + labels | ~3 GB | ~6 GB |
| Browse + Inference (+ INFER) | Above + attn_weights | ~6 GB | ~9 GB |
| Full (+ COMPILE) | All files | ~10 GB | ~16 GB |

**Compared to original model:**

```
Original HuggingFace model (f16):    ~8 GB
Vindex browse-only (f16):            ~3 GB  (62% smaller)
Vindex browse + infer (f16):         ~6 GB  (25% smaller)
Vindex full (f16):                   ~10 GB (25% larger, but queryable + compilable)
```

Each component loads lazily — DESCRIBE never touches attention weights, INFER never touches up_weights. The vindex grows from 3GB to 10GB as you need more capabilities, but the browse-only core is always small.

**Deduplication principle:** gate_vectors.bin IS W_gate. embeddings.bin IS W_embed. They are not copies — they are the canonical storage. COMPILE reads gate_vectors.bin to reconstruct W_gate in safetensors format. No data is stored twice.

### 5.3 Layer-Aware Matching

| Layer Band | Name | Reference Databases | Typical Relations |
|------------|------|--------------------|--------------------|
| Syntax (early layers) | Morphological + Syntactic | Morphological lexicon, WordNet, English grammar, AST pairs | plural, gerund, synonym, determiner→noun, py:function_def |
| Knowledge (middle layers) | Factual + Relational | Wikidata triples, probe labels | capital, language, continent, occupation, genre |
| Output (late layers) | Formatting | None (formatting) | TF-IDF fallback only |

Layer band boundaries are model-specific and stored in `index.json`:

```json
{
  "layer_bands": {
    "syntax": [0, 13],
    "knowledge": [14, 27],
    "output": [28, 33]
  }
}
```

Default boundaries are computed during EXTRACT by analysing feature distributions. Models with more layers have wider bands. The boundaries can be overridden manually.

**Example boundaries by model:**

| Model | Layers | Syntax | Knowledge | Output |
|-------|--------|--------|-----------|--------|
| Gemma 3 4B | 34 | 0-13 | 14-27 | 28-33 |
| Llama 3 8B | 32 | 0-12 | 13-25 | 26-31 |
| Llama 3 70B | 80 | 0-30 | 31-65 | 66-79 |
| Mistral 7B | 32 | 0-12 | 13-25 | 26-31 |
| GPT-2 | 12 | 0-4 | 5-9 | 10-11 |

These are estimates. The actual boundaries are discoverable via `SHOW LAYERS` — the layer where factual features start appearing marks the syntax/knowledge boundary.

### 5.4 DESCRIBE Output Format

```
larql> DESCRIBE "France";
France
  Edges (L14-27):
    capital        → Paris           gate=1436.9  L27-27  1x  (probe)
    language       → French          gate=35.2    L24-32  4x  (probe)
    continent      → Europe          gate=14.4    L25-25  1x  (probe)
    borders        → Spain           gate=13.3    L18-18  1x  (probe)  also: España, Europe, Germany
    country        → Australia       gate=25.1    L26-26  1x  (cluster) also: Italy, Germany, Spain
  Output (L28-33):
                   → German          gate=15.0    L30-30  1x  also: Dutch, Dutch, Italian
                   → European        gate=11.9    L33-33  1x  also: Europe, Europe, Europeans

larql> DESCRIBE "def" SYNTAX;
def
  Syntax (L0-13):
    py:function_def → init           gate=12.3    L4-8   3x  (ast)
    py:function_def → forward        gate=11.1    L6-6   1x  (ast)
    py:function_def → main           gate=9.8     L5-5   1x  (ast)

larql> DESCRIBE "France" ALL LAYERS;
France
  Syntax (L0-13):
    morphological  → français        gate=8.2     L3-3   1x
    derivation     → French          gate=6.1     L7-7   1x  (wordnet)
  Edges (L14-27):
    capital        → Paris           gate=1436.9  L27-27  1x  (probe)
    language       → French          gate=35.2    L24-32  4x  (probe)
    continent      → Europe          gate=14.4    L25-25  1x  (probe)
    borders        → Spain           gate=13.3    L18-18  1x  (probe)
  Output (L28-33):
                   → German          gate=15.0    L30-30  1x
                   → European        gate=11.9    L33-33  1x
```

The `(probe)`, `(cluster)`, `(wordnet)`, `(ast)` tags show the label source. Edges with no tag use TF-IDF fallback labels.

---

## 6. Pipe Operator

LQL supports `|>` to chain statements. The output of the left statement becomes context for the right.

```sql
WALK "The capital of France is" TOP 5
    |> EXPLAIN WALK "The capital of France is";

DESCRIBE "France"
    |> DIFF WITH "llama3-8b.vindex";
```

---

## 7. The Demo Script

One terminal. One language. The full loop.

```sql
-- ═══════════════════════════════════════════════════════
-- ACT 1: DECOMPILE
-- ═══════════════════════════════════════════════════════

EXTRACT MODEL "google/gemma-3-4b-it"
    INTO "gemma3-4b.vindex"
    WITH ALL;
-- Extracts browse index + inference weights + compile weights (~10 GB)

USE "gemma3-4b.vindex";
STATS;

-- ═══════════════════════════════════════════════════════
-- ACT 2: INSPECT
-- ═══════════════════════════════════════════════════════

SHOW RELATIONS WITH EXAMPLES;

DESCRIBE "France";
-- France
--   capital   → Paris      (probe, 0.97)
--   language  → French     (probe, 0.95)
--   continent → Europe     (probe, 0.92)
--   borders   → Spain      (probe, 0.89)

DESCRIBE "Einstein";
-- Einstein
--   occupation  → physicist    (probe, 0.94)
--   birthplace  → Ulm          (probe, 0.88)
--   nationality → German       (probe, 0.91)

DESCRIBE "def" SYNTAX;
-- def
--   py:function_def → init, forward, main  (ast)

-- ═══════════════════════════════════════════════════════
-- ACT 3: WALK + INFER
-- ═══════════════════════════════════════════════════════

WALK "France" TOP 10;

EXPLAIN WALK "The capital of France is";

INFER "The capital of France is" TOP 5 COMPARE;
-- Walk prediction:  Is (99.88%)  — no attention, wrong
-- Infer prediction: Paris (97.91%) — with attention, correct

-- ═══════════════════════════════════════════════════════
-- ACT 4: EDIT
-- ═══════════════════════════════════════════════════════

DESCRIBE "John Coyle";
-- John Coyle
--   (no edges found)

INSERT INTO EDGES
    (entity, relation, target)
    VALUES ("John Coyle", "lives-in", "Colchester");

DESCRIBE "John Coyle";
-- John Coyle
--   lives-in → Colchester  (inserted)

-- ═══════════════════════════════════════════════════════
-- ACT 5: RECOMPILE
-- ═══════════════════════════════════════════════════════

DIFF "gemma3-4b.vindex" CURRENT;
-- 1 edge added: John Coyle → lives-in → Colchester

COMPILE CURRENT
    INTO MODEL "gemma3-4b-edited/"
    FORMAT safetensors;
-- Compiling 348,161 features across 34 layers...
-- Written: gemma3-4b-edited/model.safetensors
```

---

## 8. Implementation Notes

### 8.1 Parser Architecture

```
Input string
    → Lexer (tokenise keywords, strings, numbers, operators)
    → Parser (recursive descent, one statement at a time)
    → AST (Statement enum with variants per statement type)
    → Executor (dispatches to larql-core / larql-inference / larql-models)
```

The parser lives in `larql-lql`, organized as modular subfiles:

```
src/
  lib.rs                    Module exports
  ast.rs                    AST definitions
  error.rs                  Error types
  lexer.rs                  Tokenizer (90+ keywords)
  relations.rs              Relation classifier + label loader
  repl.rs                   REPL & batch runner
  parser/
    mod.rs                  Parser struct, dispatch, parse()
    lifecycle.rs            EXTRACT, COMPILE, DIFF, USE
    query.rs                WALK, INFER, SELECT, DESCRIBE, EXPLAIN
    mutation.rs             INSERT, DELETE, UPDATE, MERGE
    introspection.rs        SHOW, STATS
    helpers.rs              Token utilities, value/field/condition parsers
    tests.rs                Parser tests (70+)
  executor/
    mod.rs                  Session struct, dispatch, backend accessors
    lifecycle.rs            USE, STATS, EXTRACT/COMPILE/DIFF (stubs)
    query.rs                WALK, INFER, SELECT, DESCRIBE, EXPLAIN
    introspection.rs        SHOW RELATIONS/LAYERS/FEATURES/MODELS
    mutation.rs             INSERT/DELETE/UPDATE/MERGE (stubs)
    helpers.rs              Formatting, token filtering
    tests.rs                Executor tests (35+)
```

### 8.2 AST

```rust
pub enum Statement {
    // Lifecycle
    Extract { model: String, output: String, components: Option<Vec<Component>>,
              layers: Option<Range>, extract_level: ExtractLevel },
    Compile { vindex: VindexRef, output: String, format: OutputFormat },
    Diff { a: VindexRef, b: VindexRef, layer: Option<u32>,
           relation: Option<String>, limit: Option<u32> },
    Use { target: UseTarget },

    // Knowledge browser (pure vindex)
    Walk { prompt: String, top: Option<u32>, layers: Option<Range>,
           mode: Option<WalkMode>, compare: bool },
    Select { fields: Vec<Field>, conditions: Vec<Condition>,
             nearest: Option<NearestClause>, order: Option<OrderBy>,
             limit: Option<u32> },
    Describe { entity: String, band: Option<LayerBand>, layer: Option<u32>,
               relations_only: bool },
    Explain { prompt: String, mode: ExplainMode, layers: Option<Range>,
              verbose: bool, top: Option<u32> },

    // Inference (requires model weights)
    Infer { prompt: String, top: Option<u32>, compare: bool },

    // Mutation
    Insert { entity: String, relation: String, target: String,
             layer: Option<u32>, confidence: Option<f32> },
    Delete { conditions: Vec<Condition> },
    Update { set: Vec<Assignment>, conditions: Vec<Condition> },
    Merge { source: String, target: Option<String>,
            conflict: Option<ConflictStrategy> },

    // Introspection
    ShowRelations { layer: Option<u32>, with_examples: bool },
    ShowLayers { range: Option<Range> },
    ShowFeatures { layer: u32, conditions: Vec<Condition>, limit: Option<u32> },
    ShowModels,
    Stats { vindex: Option<String> },

    // Pipe
    Pipe { left: Box<Statement>, right: Box<Statement> },
}

pub enum UseTarget {
    Vindex(String),
    Model { id: String, auto_extract: bool },
}

pub enum VindexRef {
    Path(String),
    Current,
}

pub enum LayerBand {
    Syntax,        // Early layers: DESCRIBE "def" SYNTAX
    Knowledge,     // Middle layers: DESCRIBE "France" KNOWLEDGE (default)
    Output,        // Late layers: DESCRIBE "France" OUTPUT
    All,           // All layers: DESCRIBE "France" ALL LAYERS
}

pub enum ExplainMode {
    Walk,          // EXPLAIN WALK — pure vindex, no attention
    Infer,         // EXPLAIN INFER — full inference trace
}

pub enum WalkMode { Hybrid, Pure, Dense }
pub enum OutputFormat { Safetensors, Gguf }
pub enum ConflictStrategy { KeepSource, KeepTarget, HighestConfidence }
pub enum Component { FfnGate, FfnDown, FfnUp, Embeddings, AttnOv, AttnQk }

pub enum ExtractLevel {
    Browse,     // Default: gate + embed + down_meta (~3 GB)
    Inference,  // + attention weights (~6 GB)
    All,        // + up, norms, lm_head (~10 GB, enables COMPILE)
}
```

### 8.3 Crate Mapping

| Statement | Crate | Function |
|---|---|---|
| EXTRACT | `larql-models` | Read safetensors → write vindex |
| COMPILE | `larql-models` | Read vindex → write safetensors |
| WALK | `larql-inference` | Gate KNN on VectorIndex |
| INFER | `larql-inference` | predict_with_ffn (attention + walk FFN) |
| SELECT | `larql-core` | Edge query on graph |
| INSERT/DELETE/UPDATE | `larql-core` | Graph mutation |
| DESCRIBE | `larql-inference` | Multi-layer gate KNN + label lookup |
| EXPLAIN | `larql-inference` | Walk/infer with trace capture |
| MERGE | `larql-core` | Graph union |
| DIFF | `larql-core` | Graph comparison |
| SHOW/STATS | `larql-core` + `larql-models` | Metadata queries |
| USE | `larql-lql` | Session state |

### 8.4 Implementation Status

| Component | Status |
|---|---|
| LQL Parser | ✅ Done — recursive descent, 90+ keywords, modular subfiles |
| REPL | ✅ Done — rustyline, history, multi-line, help |
| USE / STATS | ✅ Done — vindex loading, stats display |
| SHOW (RELATIONS, LAYERS, FEATURES, MODELS) | ✅ Done |
| SELECT / DESCRIBE | ✅ Done — vindex edge query, layer band grouping |
| DESCRIBE layer bands (SYNTAX/KNOWLEDGE/OUTPUT/ALL) | ✅ Done — all bands, per-family boundaries from config |
| WALK / EXPLAIN WALK | ✅ Done — gate KNN, per-layer feature trace |
| INFER | ✅ Done — full forward pass with walk FFN (requires `--include-weights`) |
| EXPLAIN INFER | ✅ Done — inference trace with relation labels |
| Label loading (feature_labels.json) | ✅ Done — probe-confirmed labels override cluster labels |
| Cluster-based labels (relation_clusters.json) | ✅ Done — k=512, offset clustering, Wikidata + WordNet + pattern matching |
| EXTRACT | ✅ Done — full pipeline: gate, embed, down_meta, clustering, split weights |
| INSERT | ✅ Done — cluster-centre gate synthesis, auto-layer, patch overlay (base readonly) |
| DELETE | ✅ Done — by layer+feature or entity match, patch overlay |
| UPDATE | ✅ Done — target/confidence update, patch overlay |
| COMPILE INTO VINDEX | ✅ Done — bake_down patches into clean vindex |
| COMPILE INTO MODEL | ✅ Done — reconstructs safetensors from split weight files |
| DIFF | ✅ Done — feature-level comparison, INTO PATCH export |
| MERGE | ✅ Done — graph union with KeepSource/KeepTarget/HighestConfidence strategies |
| BEGIN/SAVE/APPLY/SHOW/REMOVE PATCH | ✅ Done — full patch lifecycle |
| Auto-patch on mutation | ✅ Done — INSERT/DELETE/UPDATE auto-start anonymous patch session |
| Readonly base | ✅ Done — base vindex files never modified, all edits via PatchedVindex overlay |
| Split weight files | ✅ Done — attn, up, down, norms, lm_head (no gate duplication) |
| f16 storage | ✅ Done — `--f16` flag, halves file sizes |
| Vindexfile | ✅ Done — declarative builds (FROM + PATCH + INSERT), `larql build` CLI |
| USE REMOTE | ✅ Done — HTTP client to larql-server, all queries forwarded, local patch overlay |
| `larql serve` | ✅ Done — HTTP/gRPC server, all endpoints, multi-model, per-session patches |
| WeightBackend (USE MODEL) | ✅ Done — direct safetensors, INFER/EXPLAIN INFER/STATS; browse ops guide to EXTRACT |
| GGUF output format | 🔴 Planned — COMPILE INTO MODEL FORMAT gguf |
| MXFP4 browse quality | 🟡 Known limitation — gate KNN noisy for 4-bit quantized MoE; INFER works correctly |
| Gated KNN for MoE | 🔴 Planned — use SiLU(gate)×up instead of raw gate dot product for MXFP4 models |
| Residual-based DESCRIBE | 🔴 Planned — capture actual residuals for accurate MoE knowledge browse |

### 8.5 INSERT Semantics — How Edge Becomes Vector

When you `INSERT ("John Coyle", "lives-in", "Colchester")`:

1. **Find the relation direction.** Look up the "lives-in" cluster centre from schema discovery. The probe labels provide the geometric direction for known relations.
2. **Find the entity embedding.** Look up "John" and "Coyle" token embeddings. Combine to get the entity vector.
3. **Find the target embedding.** Look up "Colchester" token embedding.
4. **Synthesise the gate vector.** Gate direction ≈ entity embedding, scaled to match existing gate magnitudes at the target layer.
5. **Synthesise the down vector.** Down direction ≈ target embedding, scaled to match existing down magnitudes.
6. **Synthesise the up vector.** Up direction ≈ gate direction (for simple facts).
7. **Find a free feature slot.** Use an unused feature (low activation across all entities).
8. **Write the vectors.** Update gate_vectors.bin, down metadata, and up vectors in the vindex.

The relation type determines which layer to write to. Probe data shows which layers each relation type occupies (e.g., capital features cluster at L24-27, language features at L22-32).

### 8.6 Priority Order for Implementation

1. ~~LQL Parser + REPL~~ — **done**
2. ~~USE / STATS / SHOW~~ — **done**
3. ~~SELECT / DESCRIBE~~ — **done**
4. ~~WALK / EXPLAIN WALK~~ — **done**
5. ~~INFER / EXPLAIN INFER~~ — **done**
6. ~~Label loading (probe + cluster)~~ — **done**
7. ~~DESCRIBE SYNTAX / ALL LAYERS~~ — **done**
8. ~~EXTRACT~~ — **done**
9. ~~W_up extraction~~ — **done**
10. ~~INSERT~~ — **done**
11. ~~COMPILE~~ — **done**
12. ~~DIFF / DELETE / UPDATE / MERGE~~ — **done**
13. ~~WeightBackend (USE MODEL)~~ — **done**

---

## 9. The REPL

```
$ larql repl

   ╦   ╔═╗ ╦═╗ ╔═╗ ╦
   ║   ╠═╣ ╠╦╝ ║═╬╗║
   ╩═╝ ╩ ╩ ╩╚═ ╚═╝╚╩═╝
   Lazarus Query Language v0.1

larql> USE "output/gemma3-4b-full.vindex";
Using: output/gemma3-4b-full.vindex (34 layers, 348.2K features,
  model: google/gemma-3-4b-it, relations: 512 types, 143 probe-confirmed)

larql> DESCRIBE "France";
France
  Edges (L14-27):
    capital        → Paris           gate=1436.9  L27-27  1x  (probe)
    language       → French          gate=35.2    L24-32  4x  (probe)
    continent      → Europe          gate=14.4    L25-25  1x  (probe)
    borders        → Spain           gate=13.3    L18-18  1x  (probe)
    country        → Australia       gate=25.1    L26-26  1x  also: Italy, Germany, Spain
  Output (L28-33):
                   → German          gate=15.0    L30-30  1x  also: Dutch, Italian
                   → European        gate=11.9    L33-33  1x  also: Europe, Europeans

larql> INFER "The capital of France is" TOP 3;
  1. Paris                (97.91%)
  2. the                  (0.42%)
  3. a                    (0.31%)

larql> INSERT INTO EDGES (entity, relation, target)
   ...   VALUES ("John Coyle", "lives-in", "Colchester");
Inserted 1 edge. Feature F8821@L26 allocated.

larql> COMPILE CURRENT INTO MODEL "edited/" FORMAT safetensors;
Compiling 348,161 features across 34 layers...
Written: edited/model.safetensors (8.1 GB)
```

---

## 10. Integration with larql-knowledge

The `larql-knowledge` project produces all label artifacts. The engine consumes them.

### 10.1 Label Merge Command

```bash
# After running probes in larql-knowledge:
larql label <vindex_path> \
  --probes <path_to_feature_labels.json> \
  --triples <path_to_wikidata_triples.json> \
  --wordnet <path_to_wordnet_relations.json> \
  --ast <path_to_ast_dir>
```

### 10.2 What the Engine Reads

| File | Produced by | Used for |
|------|-------------|----------|
| `feature_labels.json` | `larql-knowledge` probe pipeline | Probe-confirmed per-feature labels |
| `relation_clusters.json` | `larql` vindex build + `larql-knowledge` triples/WordNet | Cluster-based labels |
| `feature_clusters.jsonl` | `larql` vindex build | Per-feature cluster assignments |

### 10.3 Additive Updates

Labels are additive. New probes add new feature labels without removing existing ones. New triples improve cluster labels for unprobed features. The `larql label` command merges incrementally:

```bash
# Add new probe results — keeps existing, adds new
larql label gemma3-4b.vindex \
  --probes probes/batch2/feature_labels.json

# Re-cluster with new triples — improves cluster labels only
larql label gemma3-4b.vindex \
  --triples data/wikidata_triples_v2.json
```

---

## 11. Future Extensions (Not for V1)

### 11.1 Cross-Model DIFF

Compare what one model knows versus another about the same entity or relation. The Procrustes alignment work (0.946 cosine across models) means knowledge coordinates map between architectures.

```sql
-- What does Llama know about France that Gemma doesn't?
DIFF "gemma3-4b.vindex" "llama3-8b.vindex"
    ENTITY "France";
-- Gemma has:  capital→Paris, language→French, continent→Europe
-- Llama adds: population→67M, anthem→La Marseillaise, motto→Liberté

-- Which model has more knowledge about music?
DIFF "gemma3-4b.vindex" "llama3-8b.vindex"
    RELATION "instrument";
-- Gemma: 4 features (guitar, piano, saxophone, drums)
-- Llama: 11 features (+ violin, cello, flute, trumpet, bass, harmonica, banjo)

-- Full knowledge comparison report
DIFF "gemma3-4b.vindex" "llama3-70b.vindex"
    INTO REPORT "model-comparison.md";
```

This answers a question nobody can answer today: "what does this model know that that model doesn't?" The vindex makes it a database query instead of a research project.

### 11.2 Search Across Entities

Reverse DESCRIBE — instead of "what does the model know about France?" ask "which entities have this relation?"

```sql
-- Which entities have a capital?
SELECT entity, target FROM EDGES
    WHERE relation = "capital"
    ORDER BY confidence DESC;
-- France→Paris, Germany→Berlin, Japan→Tokyo, UK→London, ...

-- What does the model think is in Europe?
SELECT entity FROM EDGES
    WHERE relation = "continent"
    AND target = "Europe";
-- France, Germany, Spain, Italy, Poland, ...

-- Find all composers the model knows
SELECT entity, target FROM EDGES
    WHERE relation = "occupation"
    AND target LIKE "%composer%";
-- Mozart→composer, Beethoven→composer, Bach→composer, ...
```

This is full-text search across the model's entire knowledge graph. No inference, no prompting — just index lookups against the gate vectors and labels.

### 11.3 Export to Standard Knowledge Graph Formats

Dump the model's knowledge as standard graph formats, making it queryable by every graph database, SPARQL endpoint, and visualisation tool in the world.

```sql
EXPORT CURRENT
    INTO "gemma3-4b-knowledge.ttl"
    FORMAT turtle;
-- RDF triples: <France> <capital> <Paris> .
--              <France> <language> <French> .

EXPORT CURRENT
    INTO "gemma3-4b-knowledge.csv"
    FORMAT neo4j;
-- CSV import files for Neo4j: nodes.csv, relationships.csv

EXPORT CURRENT
    INTO "gemma3-4b-knowledge.jsonld"
    FORMAT json-ld;
-- JSON-LD linked data format

EXPORT CURRENT
    INTO "gemma3-4b-knowledge.graphml"
    FORMAT graphml;
-- GraphML for visualisation in Gephi, yEd, Cytoscape
```

The export includes only probe-confirmed and high-confidence cluster labels — not raw TF-IDF fallbacks. The resulting knowledge graph is a curated extraction of what the model actually knows, in formats the entire graph database ecosystem can consume.

### 11.4 Temporal Snapshots

Compare vindexes built from different training checkpoints to see what the model learned between runs.

```sql
-- Build vindexes from two checkpoints of the same model
-- (or two different versions: Llama 2 vs Llama 3)

DIFF "llama2-7b.vindex" "llama3-8b.vindex"
    INTO REPORT "llama-evolution.md";
-- New relations in Llama 3: 47 types not in Llama 2
-- Removed relations: 3 types (deprecated knowledge)
-- Strengthened: capital edges +40% confidence
-- New entities: 12,000 entities Llama 3 knows that Llama 2 doesn't

DIFF "gemma2-2b.vindex" "gemma3-4b.vindex"
    RELATION "capital";
-- Gemma 2: 80 countries with capital edges
-- Gemma 3: 180 countries with capital edges (100 new)
```

Model archaeology. Track how knowledge accumulates across training runs, model families, and parameter scales. See which facts are stable across versions and which are volatile.

### 11.5 Streaming Queries

For large models with hundreds of edges per entity, return results progressively:

```sql
-- Stream edges as they're found, layer by layer
DESCRIBE "France" STREAM;
-- L14: (nothing yet)
-- L15: language → French (probe)
-- L18: borders → Spain (probe)
-- ...results appear as each layer is scanned

-- Useful for remote vindexes — start seeing results before the full scan completes
USE REMOTE "https://models.example.com/llama-70b.vindex";
DESCRIBE "Einstein" STREAM LIMIT 20;
```

The layer-level byte offsets in gate_vectors.bin enable this — each layer can be fetched and scanned independently. For remote vindexes, the client sees results from L14 while L15-27 are still downloading.

### 11.6 Additional Future Work

- **STEER** — relation steering via ±Δ vectors at a layer. Proven (8/8) but complex to expose cleanly.
- **CHAIN** — multi-hop via L24 residual injection. Proven (KL<0.06) but requires model loaded.
- **LIFT** — analogy operation (same edge, different node). Proven but needs model.
- **GRAPH ALGORITHMS** — PageRank, community detection, shortest path across the knowledge graph.
- **INGEST** — document → context graph extraction (Mode 5, the Apollo demo).
- **Training from graph** — compile an edited knowledge graph back to weights, bypassing gradient descent entirely.

### 11.7 Attention Template Cache (Research, Not Yet Proven at Scale)

Early research suggests that 99% of attention heads produce fixed patterns across entity substitutions for the same query template. If validated, attention templates could be cached in the vindex — a small lookup table (~50 KB) replacing 3GB of attention weights.

This would collapse the inference tier into the knowledge tier: full INFER from the 3GB browse-only vindex, no attention weights needed. The knowledge server becomes the inference server.

**Status:** The template fixedness has been demonstrated for single relations. The remaining research question is whether the per-layer mode count is small enough for full attention caching — the final blocker before inference with zero matrix multiplication. This is an active area of research, not a planned feature.