# Vindex — The Queryable Model Format

**Version:** 0.1  
**Author:** Chris Hay  
**Date:** 2026-03-31  
**Status:** Draft  
**Implementation:** `larql-vindex` crate (Rust)

---

## 1. What is a Vindex?

A vindex (vector index) is a directory containing a neural network's weights reorganised for queryability. The model IS the database — no data is duplicated, no copies are made. Each weight matrix is stored once in its optimal format for the operations it supports.

**Key principle:** `gate_vectors.bin` IS W_gate. `embeddings.bin` IS W_embed. They are the canonical storage, not copies. COMPILE reads `gate_vectors.bin` to reconstruct W_gate in safetensors format.

---

## 2. File Layout

```
gemma3-4b.vindex/
│
│  # ═══ Query Index (browse-only core) ═══
│  # WALK, DESCRIBE, SELECT, EXPLAIN WALK use only these files.
│
├── gate_vectors.bin          # W_gate rows per layer (KNN index)
├── embeddings.bin            # W_embed matrix (token lookup)
├── down_meta.jsonl           # Per-feature output metadata (top tokens + scores)
│
│  # ═══ Inference Weights (for INFER) ═══
│  # Only the weights NOT already in the query index.
│
├── model_weights.bin         # Attention + FFN weights (Q, K, V, O, up, down per layer)
├── weight_manifest.json      # Key → offset/length mapping into model_weights.bin
│
│  # ═══ Metadata & Labels ═══
│
├── index.json                # VindexConfig: layers, sizes, component manifest
├── tokenizer.json            # HuggingFace tokenizer
├── relation_clusters.json    # Cluster centres, labels, counts (from build)
├── feature_clusters.jsonl    # Per-feature cluster assignments (from build)
└── feature_labels.json       # Probe-confirmed labels (from larql-knowledge)
```

---

## 3. Binary Formats

### 3.1 gate_vectors.bin

Raw f32 floats, contiguous, no headers. Layer-by-layer concatenation.

**Layout:**
```
[Layer 0: num_features × hidden_size × f32]
[Layer 1: num_features × hidden_size × f32]
...
[Layer N: num_features × hidden_size × f32]
```

**Per-layer shape:** `(intermediate_size, hidden_size)` — one row per FFN feature.

**Byte order:** Native endian (platform-dependent, typically little-endian).

**Index:** `VindexLayerInfo` in `index.json` stores byte offset and length for each layer, enabling random access without reading the entire file.

**Usage:** Gate KNN — `residual × gate_vectors^T` finds which features fire. This is both the gate computation and the similarity search: same operation, different framing.

### 3.2 embeddings.bin

Raw f32 floats, no headers. Single contiguous matrix.

**Shape:** `(vocab_size, hidden_size)` in row-major order.

**Usage:** Token embedding lookup. Multiply by `embed_scale` (from config) to match gate vector magnitudes. For multi-token entities, average the scaled embeddings.

### 3.3 down_meta.jsonl

NDJSON (newline-delimited JSON). One record per feature per layer.

**Record format:**
```json
{"l":0,"f":512,"t":"the","i":278,"c":3.45,"k":[{"t":"the","i":278,"s":3.45},{"t":"and","i":345,"s":2.12}]}
```

| Field | Type | Description |
|-------|------|-------------|
| `l` | usize | Layer index |
| `f` | usize | Feature index within layer |
| `t` | string | Top output token (decoded) |
| `i` | u32 | Top output token ID |
| `c` | f32 | C-score (selectivity of top token) |
| `k` | array | Top-K output tokens with scores |

Each `k` entry: `{"t": token_string, "i": token_id, "s": logit_score}`

**Usage:** After gate KNN finds which features fire, down_meta provides what each feature outputs. This is the edge label: gate fires on entity → down outputs target.

### 3.4 model_weights.bin

Binary container for all model weights not already in the query index. Used by INFER and COMPILE.

**Format:** Sequential f32 tensors, no headers. Layout described by `weight_manifest.json`.

**Contents (per layer):**
- Attention: Q, K, V, O projection matrices
- FFN: gate, up, down weight matrices
- Norms: input_layernorm, post_attention_layernorm vectors

**Plus global:**
- Final norm vector
- LM head (output projection) matrix

### 3.5 weight_manifest.json

JSON array mapping tensor keys to byte offsets in `model_weights.bin`.

```json
[
  {
    "key": "layers.0.self_attn.q_proj.weight",
    "kind": "tensor",
    "shape": [2048, 2560],
    "offset": 0,
    "length": 20971520
  },
  {
    "key": "norm.weight",
    "kind": "vector",
    "shape": [2560],
    "offset": 1234567890,
    "length": 10240
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `key` | string | Tensor key (architecture-specific naming) |
| `kind` | string | `"tensor"` (2D) or `"vector"` (1D) |
| `shape` | [usize] | Dimensions |
| `offset` | u64 | Byte offset into model_weights.bin |
| `length` | u64 | Byte length (element_count × 4) |

### 3.6 index.json (VindexConfig)

```json
{
  "version": 1,
  "model": "google/gemma-3-4b-it",
  "family": "gemma3",
  "num_layers": 34,
  "hidden_size": 2560,
  "intermediate_size": 10240,
  "vocab_size": 262144,
  "embed_scale": 50.596,
  "layers": [
    {"layer": 0, "num_features": 10240, "offset": 0, "length": 104857600},
    {"layer": 1, "num_features": 10240, "offset": 104857600, "length": 104857600}
  ],
  "down_top_k": 10,
  "has_model_weights": false,
  "model_config": {
    "model_type": "gemma3",
    "head_dim": 256,
    "num_q_heads": 8,
    "num_kv_heads": 4,
    "rope_base": 10000.0,
    "sliding_window": 1024
  }
}
```

### 3.7 Label Files

**relation_clusters.json** — Discovered relation clusters from offset-direction clustering:
```json
{
  "k": 512,
  "labels": ["capital", "language", "morphological", ...],
  "counts": [142, 89, 1203, ...],
  "top_tokens": [["Paris", "Berlin", "Tokyo"], ["French", "English", "German"], ...]
}
```

**feature_clusters.jsonl** — Per-feature cluster assignments:
```json
{"l":26,"f":9515,"c":42}
```
Maps (layer, feature) → cluster_id.

**feature_labels.json** — Probe-confirmed labels (from larql-knowledge pipeline):
```json
{
  "26:9515": "capital",
  "24:4532": "language",
  "25:3877": "continent"
}
```
Key format: `"layer:feature"`. These override cluster labels.

---

## 4. Extract Levels

A vindex can be built at three levels, each adding more weight components:

| Level | LQL Syntax | Files Added | Size (f16 est.) | Enables |
|-------|-----------|-------------|-----------------|---------|
| Browse | `EXTRACT MODEL ... INTO ...` | gate + embed + down_meta | ~3 GB | WALK, DESCRIBE, SELECT |
| Inference | `... WITH INFERENCE` | + model_weights.bin | ~6 GB | + INFER, EXPLAIN INFER |
| All | `... WITH ALL` | + model_weights.bin (full) | ~10 GB | + COMPILE |

Currently implemented as a single `has_model_weights: bool` flag. The tiered split (separate attn_weights.bin, up_weights.bin, etc.) is planned but not yet implemented.

---

## 5. Core Operations

### 5.1 Load

```rust
let config = load_vindex_config(&path)?;
let mut cb = SilentLoadCallbacks;
let index = VectorIndex::load_vindex(&path, &mut cb)?;
```

Loading reads:
1. `index.json` → `VindexConfig` (layer offsets, model metadata)
2. `gate_vectors.bin` → per-layer `Array2<f32>` matrices (via offset lookup)
3. `down_meta.jsonl` → per-feature `FeatureMeta` (top token, c_score, top_k)

Embeddings and tokenizer are loaded separately on demand:
```rust
let (embed, embed_scale) = load_vindex_embeddings(&path)?;
let tokenizer = load_vindex_tokenizer(&path)?;
```

### 5.2 Gate KNN

```rust
let hits: Vec<(usize, f32)> = index.gate_knn(layer, &residual, top_k);
```

Computes `gate_matrix @ residual` via BLAS matmul, returns top-K feature indices sorted by absolute dot product. This is both the gate computation and the nearest-neighbor search.

### 5.3 Walk

```rust
let trace: WalkTrace = index.walk(&query, &layers, top_k);
```

Runs gate KNN at each layer, annotates hits with down_meta (what each feature outputs). Returns a `WalkTrace` with per-layer `WalkHit` entries:

```rust
pub struct WalkHit {
    pub layer: usize,
    pub feature: usize,
    pub gate_score: f32,
    pub meta: FeatureMeta,
}
```

### 5.4 Mutate

```rust
// Insert: set gate vector + metadata for a feature
index.set_gate_vector(layer, feature, &gate_vec);
index.set_feature_meta(layer, feature, meta);

// Delete: clear metadata
index.delete_feature_meta(layer, feature);

// Find unused slot
let slot = index.find_free_feature(layer);

// Save changes to disk
index.save_down_meta(&path)?;
index.save_gate_vectors(&path)?;
VectorIndex::save_config(&config, &path)?;
```

### 5.5 Feature Lookup

```rust
let meta: Option<&FeatureMeta> = index.feature_meta(layer, feature);
let n: usize = index.num_features(layer);
let layers: Vec<usize> = index.loaded_layers();
```

---

## 6. Rust API

### 6.1 Core Types

```rust
// The index — owns gate vectors + down metadata in memory
pub struct VectorIndex { ... }

// Per-feature output metadata
pub struct FeatureMeta {
    pub top_token: String,
    pub top_token_id: u32,
    pub c_score: f32,
    pub top_k: Vec<TopKEntry>,
}

// Walk result
pub struct WalkTrace {
    pub layers: Vec<(usize, Vec<WalkHit>)>,
}

// Config from index.json
pub struct VindexConfig {
    pub version: u32,
    pub model: String,
    pub family: String,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub embed_scale: f32,
    pub layers: Vec<VindexLayerInfo>,
    pub down_top_k: usize,
    pub has_model_weights: bool,
    pub model_config: Option<VindexModelConfig>,
}

// Error type
pub enum VindexError {
    NotADirectory(PathBuf),
    NoSafetensors(PathBuf),
    MissingTensor(String),
    Parse(String),
    UnsupportedDtype(String),
    Io(std::io::Error),
}
```

### 6.2 Load Functions

```rust
pub fn load_vindex_config(dir: &Path) -> Result<VindexConfig, VindexError>;
pub fn load_vindex_embeddings(dir: &Path) -> Result<(Array2<f32>, f32), VindexError>;
pub fn load_vindex_tokenizer(dir: &Path) -> Result<Tokenizer, VindexError>;
pub fn load_feature_labels(path: &Path) -> Result<HashMap<(usize, usize), String>, VindexError>;
```

### 6.3 Callbacks

```rust
pub trait IndexLoadCallbacks {
    fn on_file_start(&mut self, component: &str, path: &str) {}
    fn on_progress(&mut self, records: usize) {}
    fn on_file_done(&mut self, component: &str, records: usize, elapsed_ms: f64) {}
}

pub struct SilentLoadCallbacks;
```

---

## 7. Crate Structure

```
larql-vindex/
├── Cargo.toml
└── src/
    ├── lib.rs          Module exports + crate docs
    ├── config.rs       VindexConfig, VindexLayerInfo, DownMetaRecord
    ├── error.rs        VindexError
    ├── index.rs        VectorIndex, FeatureMeta, KNN, WalkTrace
    ├── load.rs         Load gate vectors, down_meta, embeddings, config
    └── mutate.rs       Set/delete features, save to disk
```

**Dependencies:** `larql-models` (TopKEntry), `ndarray` (BLAS), `serde`/`serde_json`, `tokenizers`, `thiserror`

**Build pipeline** (EXTRACT) lives in `larql-inference/vindex/` because it needs `ModelWeights`.

---

## 8. Size Reference (Gemma 3 4B)

| File | Size | Description |
|------|------|-------------|
| gate_vectors.bin | 3.32 GB | 34 layers × 10,240 features × 2,560 dim × 4 bytes |
| embeddings.bin | 2.50 GB | 262,144 vocab × 2,560 dim × 4 bytes (f32, could be f16) |
| down_meta.jsonl | ~160 MB | 348,160 features × ~460 bytes avg per record |
| model_weights.bin | ~12 GB | Full model weights (when `--include-weights`) |
| index.json | ~8 KB | Config + layer offsets |
| tokenizer.json | ~32 MB | HuggingFace tokenizer |
| relation_clusters.json | ~2 MB | 512 clusters with labels + top tokens |
| feature_clusters.jsonl | ~5 MB | 348,160 feature → cluster assignments |
| feature_labels.json | ~10 KB | 157 probe-confirmed labels |

**Browse-only total:** ~6 GB (f32), ~3 GB projected (f16)  
**With inference weights:** ~18 GB (f32), ~9 GB projected (f16)

---

## 9. Future Format Changes

Planned improvements (not yet implemented):

1. **Binary down_meta** — Replace JSONL with compact binary format (~2 MB vs 160 MB). Store token IDs only, resolve strings via tokenizer at read time.
2. **Split weight files** — Replace single `model_weights.bin` with component files (`attn_weights.bin`, `up_weights.bin`, `norms.bin`, `lm_head.bin`) for lazy loading by capability.
3. **f16 storage** — Half-precision for gate vectors and embeddings, halving index size.
4. **Memory-mapped loading** — Use `mmap` for gate vectors to avoid copying into heap.
5. **Incremental down_meta** — Append-only format for mutation without full rewrite.
