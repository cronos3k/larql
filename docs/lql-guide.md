# LQL Quick Start Guide

**LQL** (Lazarus Query Language) is the query language for neural network weights treated as a graph database. One binary, no Python, no GPU.

## Launch the REPL

```bash
cargo run -p larql-cli -- repl
```

Features: arrow keys, command history (`~/.larql_history`), Ctrl-R search, Ctrl-C cancel, Ctrl-D exit.

Single statement:

```bash
cargo run -p larql-cli -- lql 'SHOW MODELS;'
```

## Getting Started

### 1. Extract a model

```sql
-- Browse-only (~3 GB f16 / ~6 GB f32, fast queries, no inference)
EXTRACT MODEL "google/gemma-3-4b-it" INTO "gemma3-4b.vindex";

-- With inference support (~6 GB f16 / ~12 GB f32, enables INFER)
EXTRACT MODEL "google/gemma-3-4b-it" INTO "gemma3-4b.vindex" WITH INFERENCE;

-- Full (~10 GB f16 / ~18 GB f32, enables COMPILE for recompilation)
EXTRACT MODEL "google/gemma-3-4b-it" INTO "gemma3-4b.vindex" WITH ALL;
```

CLI equivalent: `larql extract-index google/gemma-3-4b-it -o gemma3-4b.vindex --level inference --f16`

### 2. Connect

```sql
-- Use a pre-extracted vindex (fast, all operations)
USE "gemma3-4b.vindex";
STATS;

-- Or point directly at model weights (no extraction needed)
USE MODEL "google/gemma-3-4b-it";
STATS;
-- Supports: INFER, EXPLAIN INFER, STATS
-- For WALK/DESCRIBE/SELECT/INSERT: extract into a vindex first
```

### 3. Browse knowledge

```sql
-- What does the model know about France?
DESCRIBE "France";

-- Show all layer bands (syntax + knowledge + output)
DESCRIBE "France" ALL LAYERS;

-- Verbose mode: raw cluster labels, layer ranges, also-tokens
DESCRIBE "France" VERBOSE;

-- Single layer
DESCRIBE "Mozart" AT LAYER 26;

-- Feature scan: which features fire for a prompt?
WALK "The capital of France is" TOP 10;

-- Per-layer trace
EXPLAIN WALK "The capital of France is" LAYERS 24-33;

-- SQL-style edge queries
SELECT entity, target FROM EDGES WHERE relation = "capital" LIMIT 10;
```

### 4. Run inference

Requires model weights: either a vindex built with `WITH INFERENCE` / `WITH ALL`,
or a `USE MODEL` session (direct weight access).

```sql
-- Next-token prediction with attention
INFER "The capital of France is" TOP 5;

-- Compare walk (no attention) vs infer (with attention)
INFER "The capital of France is" TOP 5 COMPARE;

-- Full inference trace
EXPLAIN INFER "The capital of France is" TOP 5;
```

### 5. Edit knowledge

```sql
-- Insert a fact
INSERT INTO EDGES (entity, relation, target)
    VALUES ("John Coyle", "lives-in", "Colchester");

-- Verify
DESCRIBE "John Coyle";

-- Delete
DELETE FROM EDGES WHERE entity = "John Coyle" AND relation = "lives-in";

-- Update
UPDATE EDGES SET target = "London"
    WHERE entity = "John Coyle" AND relation = "lives-in";
```

### 6. Patches

Patches are lightweight knowledge diffs — portable JSON files that modify a vindex without touching the base files.

```sql
-- Start recording a patch
BEGIN PATCH "medical-knowledge.vlp";

INSERT INTO EDGES (entity, relation, target)
    VALUES ("aspirin", "side_effect", "bleeding");
INSERT INTO EDGES (entity, relation, target)
    VALUES ("aspirin", "treats", "headache");

-- Save (base vindex NOT modified)
SAVE PATCH;

-- Apply a patch
APPLY PATCH "medical-knowledge.vlp";

-- Stack multiple patches
APPLY PATCH "fix-hallucinations.vlp";

-- See active patches
SHOW PATCHES;

-- Remove a patch (instantly reverts)
REMOVE PATCH "fix-hallucinations.vlp";

-- Extract diff as a patch
DIFF "base.vindex" "edited.vindex" INTO PATCH "changes.vlp";
```

### 7. Recompile

```sql
-- See what changed
DIFF "gemma3-4b.vindex" CURRENT;

-- Compile back to HuggingFace format
COMPILE CURRENT INTO MODEL "gemma3-4b-edited/" FORMAT safetensors;
```

## Introspection

```sql
-- Discovered relation types
SHOW RELATIONS WITH EXAMPLES;

-- Layer summary
SHOW LAYERS;
SHOW LAYERS RANGE 14-27;

-- Feature details
SHOW FEATURES 26 LIMIT 20;

-- Available vindexes in current directory
SHOW MODELS;

-- Active patches
SHOW PATCHES;

-- Knowledge graph coverage
STATS;
```

## Layer Bands

DESCRIBE groups features into three bands based on the model's layer structure:

| Band | Gemma 3 4B | Llama 3 8B | What it contains |
|------|-----------|-----------|-----------------|
| Syntax | L0-13 | L0-12 | Morphological, syntactic, code |
| Knowledge | L14-27 | L13-25 | Factual relations (default view) |
| Output | L28-33 | L26-31 | Formatting, token selection |

```sql
DESCRIBE "France";              -- Knowledge band (default)
DESCRIBE "France" SYNTAX;       -- Syntax band only
DESCRIBE "France" OUTPUT;       -- Output band only
DESCRIBE "France" ALL LAYERS;   -- All three bands
```

Bands are model-specific — computed automatically during EXTRACT from known architecture boundaries.

## Statement Reference

| Category | Statements |
|----------|-----------|
| Lifecycle | EXTRACT, COMPILE, DIFF, USE |
| Browse | WALK, DESCRIBE, SELECT, EXPLAIN WALK |
| Inference | INFER, EXPLAIN INFER |
| Mutation | INSERT, DELETE, UPDATE, MERGE |
| Patches | BEGIN PATCH, SAVE PATCH, APPLY PATCH, SHOW PATCHES, REMOVE PATCH |
| Introspection | SHOW RELATIONS/LAYERS/FEATURES/MODELS/PATCHES, STATS |
| Pipe | `\|>` chains two statements |

See [lql-spec.md](lql-spec.md) for the full language specification.
