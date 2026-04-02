# Vindex Format Specification

**Version:** 0.3  
**Author:** Chris Hay  
**Date:** 2026-04-01  
**Status:** Implemented (~98%)  
**Implementation:** `larql-vindex` crate (Rust)  
**Companion specs:** [Operations](vindex-operations-spec.md), [Ecosystem](vindex-ecosystem-spec.md), [LQL](lql-spec.md)

**Implementation coverage:** File layout, binary formats, extract levels, f16 storage, checksums, mmap loading, streaming extraction, `larql verify` — all implemented. Remaining: int8/int4 quantisation (future).

---

## 1. What is a Vindex?

A vindex (vector index) is a directory containing a neural network's weights reorganised for queryability. The model IS the database — each weight matrix is stored once in its optimal format for the operations it supports.

**Key principle:** `gate_vectors.bin` IS W_gate. `embeddings.bin` IS W_embed. They are the canonical storage, not copies. COMPILE reads `gate_vectors.bin` to reconstruct W_gate in safetensors format. No data is stored twice.

### Weights separated by function, not by file size

Existing model formats — safetensors, GGUF, ONNX — store all weights together. Sharded formats split by file size, not by purpose. Every format assumes you want all the weights or none of them.

Vindex separates weights by what you want to do with the model:

| Intent | Weights needed | Size (4B, f16) | What you skip |
|--------|---------------|-----------------|---------------|
| Browse knowledge | Gate + embeddings + down metadata | ~3 GB | Attention, up projections, norms, LM head |
| Run inference | + attention weights + norms | ~6 GB | Up projections, LM head |
| Edit and recompile | All weights | ~10 GB | Nothing |

This separation exists because these operations touch fundamentally different weight matrices. Querying what a model knows about France uses gate vectors and embeddings — a dot product against pre-extracted rows. It never touches attention weights. Running inference additionally needs attention for token routing. Recompiling needs everything to reconstruct the full safetensors.

---

## 2. Architecture Support

The vindex format is model-agnostic. It stores weights by function (gate, down, embed, attention) not by architecture name. Any transformer model with a gated FFN (gate + up + down projections) and standard attention (Q, K, V, O) can be extracted into a vindex.

### Supported model families

| Family | Models | FFN Type | Notes |
|--------|--------|----------|-------|
| Gemma | Gemma 2/3 (2B-27B) | Gated (gate + up + down) | GeGLU activation, GQA attention |
| Llama | Llama 2/3 (7B-405B) | Gated (gate + up + down) | SiLU activation, GQA attention |
| Mistral | Mistral 7B | Gated (gate + up + down) | Sliding window attention |
| Mixtral | Mixtral 8x7B, 8x22B | MoE (8 experts × gate + up + down) | Sparse MoE, top-2 routing |
| Qwen | Qwen 2/2.5 (0.5B-72B) | Gated (gate + up + down) | SiLU activation |
| Phi | Phi 2/3 (2.7B-14B) | Gated (gate + up + down) | Partial attention |
| DeepSeek | DeepSeek V2/V3 | MoE (shared + routed experts) | Fine-grained MoE, shared expert |
| GPT-2 | GPT-2 (117M-1.5B) | Dense (W_in + W_out) | Soft gating via GELU |

### MoE layout

Mixture-of-Experts models have multiple FFN "experts" per layer. In the vindex, each expert's features are separate entries in the same `gate_vectors.bin`. A dense model has `intermediate_size` features per layer. An MoE model with N experts has `N × intermediate_size` features per layer.

```
Dense (Gemma 4B):    10,240 features per layer → 348,160 total
MoE (Mixtral 8x7B):  8 × 14,336 = 114,688 features per layer → 3,670,016 total
```

Gate KNN naturally selects features across all experts — no router needed for browse operations.

### Dense FFN support (GPT-2)

Dense FFN models have no explicit gate matrix — the FFN is `W_out @ GELU(W_in @ x)`. `W_in` rows serve the same functional role as gate vectors: each row defines a direction in residual space that activates a feature. Extraction uses `W_in` rows as `gate_vectors.bin` and `W_out` columns as down projections. All query operations work unchanged.

### Architecture variations handled

- **GQA (Grouped Query Attention):** Fewer KV heads than Q heads. Stored as-is in attention weights.
- **Sliding window attention:** Window size stored in `model_config`. Browse operations unaffected.
- **Tied embeddings:** LM head shares embedding matrix. `lm_head.bin` omitted, `embeddings.bin` used for both.
- **RoPE variants:** Base frequency and scaling stored in `model_config`. Used by INFER only.
- **MoE shared expert (DeepSeek):** Stored as Expert 0 with a `shared: true` flag.
- **Fine-grained MoE (DeepSeek V3):** 256 experts with top-8 routing. Same structure, more features per layer.

---

## 3. File Layout

```
model.vindex/
│
│  # ═══ Query Index (browse-only core) ═══
│  # WALK, DESCRIBE, SELECT, EXPLAIN WALK use only these files.
│
├── gate_vectors.bin          # W_gate rows per layer (KNN index)
├── embeddings.bin            # W_embed matrix (token lookup)
├── down_meta.bin             # Per-feature output metadata (binary)
│
│  # ═══ Inference Weights (for INFER) ═══
│
├── attn_weights.bin          # Q, K, V, O projection matrices per layer
├── norms.bin                 # LayerNorm parameters per layer
│
│  # ═══ Compile Weights (for COMPILE) ═══
│
├── up_weights.bin            # W_up per layer
├── down_weights.bin          # W_down per layer (full vectors)
├── lm_head.bin               # Output projection (omitted if tied to embeddings)
│
│  # ═══ Metadata & Labels ═══
│
├── index.json                # VindexConfig: layers, sizes, checksums, provenance
├── tokenizer.json            # HuggingFace tokenizer
├── relation_clusters.json    # Cluster centres, labels, counts
├── feature_labels.json       # Probe-confirmed labels
└── weight_manifest.json      # Weight file → offset mapping
```

Gate vectors are NOT duplicated — `gate_vectors.bin` IS the W_gate weight matrix. COMPILE reads it directly to reconstruct the safetensors gate tensor.

---

## 4. Extract Levels

A vindex can be built at three levels, each adding more weight components:

| Level | LQL Syntax | Components | Size (f16, 4B) | Enables |
|-------|-----------|------------|-----------------|---------|
| Browse | `EXTRACT MODEL ... INTO ...` | gate + embed + down_meta | ~3 GB | WALK, DESCRIBE, SELECT |
| Inference | `... WITH INFERENCE` | + attn_weights + norms | ~6 GB | + INFER, EXPLAIN INFER |
| All | `... WITH ALL` | + up, down, lm_head | ~10 GB | + COMPILE |

```rust
pub enum ExtractLevel {
    Browse,      // Default: gate + embed + down_meta
    Inference,   // + attention weights + norms
    All,         // + all remaining weights for COMPILE
}
```

The `index.json` stores the extract level. Operations that require a higher level than available return `VindexError::InsufficientExtractLevel`.

---

## 5. Binary Formats

### 5.1 gate_vectors.bin

Raw floats (f32 or f16 per `dtype` in config), contiguous, no headers. Layer-by-layer concatenation.

**Layout:**
```
[Layer 0: num_features × hidden_size × sizeof(float)]
[Layer 1: num_features × hidden_size × sizeof(float)]
...
[Layer N: num_features × hidden_size × sizeof(float)]
```

**Per-layer shape:** `(intermediate_size, hidden_size)` — one row per FFN feature.

**Byte order:** Little-endian.

**Index:** `VindexLayerInfo` in `index.json` stores byte offset and length for each layer, enabling random access without reading the entire file.

**MoE layout:** Experts are contiguous within each layer:
```
[Layer 0, Expert 0: intermediate_size × hidden_size]
[Layer 0, Expert 1: intermediate_size × hidden_size]
...
[Layer 0, Expert N: intermediate_size × hidden_size]
[Layer 1, Expert 0: ...]
```

### 5.2 embeddings.bin

Raw floats (f32 or f16), no headers. Single contiguous matrix.

**Shape:** `(vocab_size, hidden_size)` in row-major order.

**Usage:** Token embedding lookup. Multiply by `embed_scale` (from config) to match gate vector magnitudes.

### 5.3 down_meta.bin (binary, primary)

Compact binary format. Preferred over JSONL — ~80x smaller.

**Header:**
```
magic:       [u8; 4]   "DMET"
version:     u32
num_layers:  u32
top_k:       u32       (entries per feature)
```

**Per feature (fixed size):**
```
top_token_id:  u32  (4 bytes)
c_score:       f32  (4 bytes)
num_top_k:     u8   (1 byte)
top_k entries:
  token_id:    u32  (4 bytes)
  score:       f32  (4 bytes)
```

Token strings resolved at read time via tokenizer. No string storage needed.

Only `down_meta.bin` is written during extraction. Loading falls back to `down_meta.jsonl` if binary is absent (v1 compatibility).

### 5.4 down_meta.jsonl (v1 legacy)

NDJSON format from v1 vindexes. No longer written during extraction. Supported for loading only (backward compatibility). ~5.5x larger than binary.

### 5.5 attn_weights.bin

Sequential float tensors (f32 or f16), no headers. Per-layer Q, K, V, O projection matrices. Layout described by `weight_manifest.json`.

Only present at Inference or All extract level.

### 5.6 up_weights.bin / down_weights.bin

Sequential float tensors (f32 or f16). W_up and W_down per layer respectively.

Only present at All extract level. Gate vectors are NOT stored here — they are in `gate_vectors.bin`.

### 5.7 norms.bin

LayerNorm parameters per layer (input_layernorm, post_attention_layernorm) plus final norm. Small file (~1 MB).

Only present at Inference or All extract level.

### 5.8 lm_head.bin

Output projection matrix. Omitted when `tie_word_embeddings` is true in `model_config` (embeddings.bin used instead).

Only present at All extract level.

### 5.9 weight_manifest.json

JSON array mapping tensor keys to byte offsets in the weight files.

```json
[
  {
    "key": "layers.0.self_attn.q_proj.weight",
    "kind": "tensor",
    "shape": [2048, 2560],
    "offset": 0,
    "length": 20971520
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `key` | string | Tensor key (architecture-specific naming) |
| `kind` | string | `"tensor"` (2D) or `"vector"` (1D) |
| `shape` | [usize] | Dimensions |
| `offset` | u64 | Byte offset into the relevant weight file |
| `length` | u64 | Byte length |

---

## 6. index.json (VindexConfig)

The central configuration file. Version 2 is the current format.

```json
{
  "version": 2,
  "model": "google/gemma-3-4b-it",
  "family": "gemma3",
  "dtype": "f16",

  "source": {
    "huggingface_repo": "google/gemma-3-4b-it",
    "huggingface_revision": "a1b2c3d4e5f6",
    "safetensors_sha256": "e3b0c44298fc1c149afb...",
    "extracted_at": "2026-04-01T14:22:00Z",
    "larql_version": "0.1.0"
  },

  "checksums": {
    "gate_vectors.bin": "a1b2c3d4...",
    "embeddings.bin": "e5f6a7b8...",
    "down_meta.bin": "c9d0e1f2...",
    "attn_weights.bin": "13141516..."
  },

  "num_layers": 34,
  "hidden_size": 2560,
  "intermediate_size": 10240,
  "vocab_size": 262144,
  "embed_scale": 50.596,
  "extract_level": "inference",
  "has_model_weights": true,          // deprecated — use extract_level instead
  "down_top_k": 10,

  "layer_bands": {
    "syntax": [0, 13],
    "knowledge": [14, 27],
    "output": [28, 33]
  },

  "layers": [
    {"layer": 0, "num_features": 10240, "offset": 0, "length": 52428800},
    {"layer": 1, "num_features": 10240, "offset": 52428800, "length": 52428800}
  ],

  "model_config": {
    "model_type": "gemma3",
    "head_dim": 256,
    "num_q_heads": 8,
    "num_kv_heads": 4,
    "rope_base": 10000.0,
    "rope_scaling_factor": 1.0,
    "sliding_window": 1024,
    "attention_type": "gqa",
    "activation": "geglu",
    "tie_word_embeddings": true
  }
}
```

### Key fields

**`version`** — Config format version. Current: 2.

**`dtype`** — Storage precision for all binary files. `"f32"` or `"f16"`. Cast to f32 at load time.

**`source`** — Provenance tracking. Records exactly which model checkpoint was extracted, when, and by which version of LARQL.

**`checksums`** — SHA256 hash of each binary file. Enables integrity verification after download.

**`layer_bands`** — Model-specific boundaries for DESCRIBE layer grouping. Per-family values auto-detected during EXTRACT:

| Family | Syntax | Knowledge | Output |
|--------|--------|-----------|--------|
| Gemma 3 (34L) | 0-13 | 14-27 | 28-33 |
| Llama 3 (32L) | 0-7 | 8-24 | 25-31 |
| Qwen 2.5 (28L) | 0-7 | 8-21 | 22-27 |
| Mixtral (32L) | 0-7 | 8-24 | 25-31 |
| GPT-2 (12L) | 0-3 | 4-9 | 10-11 |

**`layers`** — Per-layer byte offset and length into `gate_vectors.bin`. For MoE models, includes `num_experts` and `num_features_per_expert`.

**MoE config** (when applicable):
```json
"moe_config": {
  "num_experts": 8,
  "top_k_experts": 2,
  "has_shared_expert": false,
  "router_type": "top_k_softmax"
}
```

---

## 7. Label Files

### relation_clusters.json

Discovered relation clusters from offset-direction clustering:
```json
{
  "k": 512,
  "labels": ["capital", "language", "morphological"],
  "counts": [142, 89, 1203],
  "top_tokens": [["Paris", "Berlin", "Tokyo"], ["French", "English", "German"]]
}
```

### feature_labels.json

Probe-confirmed labels (from the larql-knowledge pipeline):
```json
{
  "26:9515": "capital",
  "24:4532": "language",
  "25:3877": "continent"
}
```

Key format: `"layer:feature"`. These override cluster labels at query time.

---

## 8. Storage Precision

The `dtype` field in `index.json` controls storage precision for all binary files.

| Dtype | Bytes/float | gate_vectors (4B) | embeddings (4B) | Total browse |
|-------|-------------|-------------------|-----------------|--------------|
| f32 | 4 | 3.32 GB | 2.50 GB | ~6 GB |
| f16 | 2 | 1.66 GB | 1.25 GB | ~3 GB |

All data is cast to f32 at load time. Gate KNN accuracy at f16 is effectively identical to f32 — the top-K results don't change because ranking is preserved.

Controlled by `StorageDtype` enum in the implementation:
```rust
pub enum StorageDtype {
    F32,
    F16,
}
```

---

## 9. Size Reference (Gemma 3 4B)

### f32

| File | Size | Description |
|------|------|-------------|
| gate_vectors.bin | 3.32 GB | 34 × 10,240 × 2,560 × 4 bytes |
| embeddings.bin | 2.50 GB | 262,144 × 2,560 × 4 bytes |
| down_meta.bin | ~2 MB | Binary token IDs + scores |
| attn_weights.bin | ~6 GB | Q, K, V, O per layer |
| up_weights.bin | ~3.4 GB | W_up per layer |
| down_weights.bin | ~3.4 GB | W_down per layer |
| norms.bin | ~1 MB | LayerNorm parameters |
| lm_head.bin | ~2.6 GB | Output projection |
| **Browse total** | **~6 GB** | |
| **Inference total** | **~12 GB** | |
| **All total** | **~18 GB** | |

### f16

| File | Size | Description |
|------|------|-------------|
| gate_vectors.bin | 1.66 GB | Half precision |
| embeddings.bin | 1.25 GB | Half precision |
| down_meta.bin | ~2 MB | Same (integer token IDs) |
| attn_weights.bin | ~3 GB | Half precision |
| up_weights.bin | ~1.7 GB | Half precision |
| down_weights.bin | ~1.7 GB | Half precision |
| norms.bin | ~1 MB | Small regardless |
| lm_head.bin | ~1.3 GB | Half precision |
| **Browse total** | **~3 GB** | |
| **Inference total** | **~6 GB** | |
| **All total** | **~10 GB** | |

---

## 10. Version History

| Version | Changes |
|---------|---------|
| 1 | Original: gate + embed + down_meta JSONL + model_weights.bin |
| 2 | Added extract_level, layer_bands, model_config, source, checksums, dtype. Binary down_meta. Split weight files (attn, up, down, norms, lm_head). f16 storage. |

**Compatibility:** v1 vindexes load with sensible defaults for missing fields:
- Missing `layer_bands` → auto-computed from layer count
- Missing `source` → empty (provenance unknown)
- Missing `checksums` → skip verification
- Missing `extract_level` → inferred from `has_model_weights`
- Missing `dtype` → assumed f32

Legacy `model_weights.bin` is still supported for loading. The engine checks for split weight files first, falls back to `model_weights.bin` + `weight_manifest.json`.

---

## 11. Implemented Format Features

### 11.1 Memory-Mapped Loading — IMPLEMENTED

`gate_vectors.bin` and `embeddings.bin` are loaded via `mmap` (memmap2). For f32 files, gate KNN reinterprets the mmap'd bytes directly as `&[f32]` — zero heap allocation, zero copy. The OS pages data in on demand. Only queried layers consume physical RAM.

Saves use atomic write-to-temp + rename to avoid invalidating active mmaps. Multiple vindexes can share physical memory for overlapping pages.

### 11.2 Checksums and Verification — IMPLEMENTED

SHA256 hashes of all binary files, stored in `index.json` under `checksums`. Computed during EXTRACT. Verified via CLI:

```bash
larql verify gemma3-4b.vindex
# gate_vectors.bin ... OK (1.66 GB)
# embeddings.bin ... OK (1.25 GB)
# down_meta.bin ... OK (29 MB)
# All 3 files verified.
```

---

## 12. Future Format Changes

### 12.1 Quantised Browse (Priority: LOW)

Store gate vectors at int8 or int4 precision. KNN accuracy is nearly identical — ranking is preserved.

```
Gate vectors at f32:  3.32 GB
Gate vectors at f16:  1.66 GB
Gate vectors at int8: 0.83 GB
Gate vectors at int4: 0.42 GB — a 4B model's knowledge in 400 MB
```

### 12.2 MXFP4 Quantized Models

Models distributed with MXFP4 block quantization (e.g., GPT-OSS-120B) can be extracted to vindex format, but gate KNN produces noisy results due to 4-bit weight precision. The model works correctly at inference time because the full forward pass (SiLU gating × up projection, transformed residuals) compensates for quantization noise. Isolated gate dot products cannot.

See [Operations Spec Section 6](vindex-operations-spec.md) for strategies.

### 12.3 Streaming Build — IMPLEMENTED

Extracts vindex from safetensors without loading the full model into memory. Mmaps safetensors shards, processes one layer at a time. Peak memory = embeddings + 1 layer's weights, not the full model. Enables 120B+ MoE extraction on machines with 16 GB RAM.

---

## License

Apache-2.0
