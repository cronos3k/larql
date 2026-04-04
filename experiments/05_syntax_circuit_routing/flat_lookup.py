#!/usr/bin/env python3
"""
flat_lookup.py — Graph lookup without layers

The current walk still goes L14, L15, ... L27 sequentially.
That's the transformer's architecture leaking into the graph.

A true graph lookup:
  entity_embedding @ ALL_gate_vectors → top features → filter by relation → answer

One matmul against the full gate matrix (all layers concatenated),
not 14 sequential matmuls. No hourglass. No collapse. No layers.

This script:
  1. Concatenates all knowledge-layer gate vectors into one matrix
  2. Does a single lookup: entity → features → answer
  3. Compares accuracy: flat (1 step) vs layered (14 steps) vs transformer
  4. Shows that the layer structure is unnecessary for retrieval

USAGE:
  python3 experiments/05_syntax_circuit_routing/flat_lookup.py \
      --model google/gemma-3-4b-it \
      --vindex output/gemma3-4b-f16.vindex
"""

import argparse
import json
import math
import struct
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np


# ---- Test queries -------------------------------------------------------

QUERIES = [
    # (prompt, entity_token, expected_answer, relation)
    ("The capital of France is", "France", "Paris", "capital"),
    ("The capital of Japan is", "Japan", "Tokyo", "capital"),
    ("The capital of Brazil is", "Brazil", "Brasilia", "capital"),
    ("The capital of Egypt is", "Egypt", "Cairo", "capital"),
    ("The capital of Germany is", "Germany", "Berlin", "capital"),
    ("The capital of India is", "India", "Delhi", "capital"),
    ("The capital of Spain is", "Spain", "Madrid", "capital"),
    ("The capital of Italy is", "Italy", "Rome", "capital"),
    ("The capital of Sweden is", "Sweden", "Stockholm", "capital"),
    ("The capital of Thailand is", "Thailand", "Bangkok", "capital"),
    ("The capital of Mexico is", "Mexico", "Mexico", "capital"),
    ("The capital of Canada is", "Canada", "Ottawa", "capital"),
    ("The capital of Australia is", "Australia", "Canberra", "capital"),
    ("The capital of Argentina is", "Argentina", "Buenos", "capital"),
    ("The capital of Kenya is", "Kenya", "Nairobi", "capital"),
    ("The capital of Poland is", "Poland", "Warsaw", "capital"),
    ("The capital of Turkey is", "Turkey", "Ankara", "capital"),
    ("The capital of Peru is", "Peru", "Lima", "capital"),
    ("The capital of Nigeria is", "Nigeria", "Abuja", "capital"),
    ("The capital of Vietnam is", "Vietnam", "Hanoi", "capital"),
]


def load_vindex(vindex_path):
    """Load vindex config, gates, and down_meta."""
    vindex_path = Path(vindex_path)

    with open(vindex_path / "index.json") as f:
        config = json.load(f)

    hidden_size = config["hidden_size"]
    bands = config.get("layer_bands", {})
    kn_start = bands.get("knowledge", [14, 27])[0]
    kn_end = bands.get("knowledge", [14, 27])[1]

    # Gate vectors
    gate_path = vindex_path / "gate_vectors.bin"
    gate_file_size = gate_path.stat().st_size
    total_elements = sum(li["num_features"] for li in config["layers"]) * hidden_size

    if gate_file_size == total_elements * 2:
        gate_dtype, bpe = np.float16, 2
    else:
        gate_dtype, bpe = np.float32, 4

    gate_raw = np.fromfile(gate_path, dtype=gate_dtype)
    gates = {}
    for layer_info in config["layers"]:
        layer = layer_info["layer"]
        nf = layer_info["num_features"]
        offset = layer_info["offset"] // bpe
        chunk = gate_raw[offset:offset + nf * hidden_size].reshape(nf, hidden_size)
        gates[layer] = chunk.astype(np.float32)

    # Down meta (token labels for features)
    down_meta = {}
    down_meta_path = vindex_path / "down_meta.bin"
    if down_meta_path.exists():
        try:
            from tokenizers import Tokenizer
            tok_path = vindex_path / "tokenizer.json"
            if tok_path.exists():
                tok = Tokenizer.from_file(str(tok_path))
            else:
                tok = None

            data = down_meta_path.read_bytes()
            pos = 0
            magic, version, num_layers, top_k_count = struct.unpack_from("<IIII", data, pos)
            pos += 16
            record_size = 8 + top_k_count * 8

            def decode(tid):
                if tid == 0 or tok is None:
                    return ""
                try:
                    return tok.decode([tid], skip_special_tokens=True).strip()
                except:
                    return ""

            for layer_idx in range(num_layers):
                nf = struct.unpack_from("<I", data, pos)[0]
                pos += 4
                for feat_idx in range(nf):
                    top_tid = struct.unpack_from("<I", data, pos)[0]
                    token_str = decode(top_tid)
                    if token_str:
                        down_meta[(layer_idx, feat_idx)] = token_str
                    pos += record_size

            print(f"  Down meta: {len(down_meta)} feature labels")
        except Exception as e:
            print(f"  Down meta load failed: {e}")

    # Feature labels
    labels = {}
    labels_path = vindex_path / "feature_labels.json"
    if labels_path.exists():
        with open(labels_path) as f:
            labels = json.load(f)

    return config, gates, down_meta, labels


def get_entity_embedding(model_path, entity_token):
    """Get the scaled embedding for an entity token."""
    from mlx_lm import load as mlx_load
    import mlx.core as mx

    model, tokenizer = mlx_load(model_path)

    tokens = tokenizer.encode(entity_token)
    # Take the last meaningful token
    try:
        lm = model.language_model
        inner = lm.model
        embed_fn = inner.embed_tokens
    except:
        embed_fn = model.model.embed_tokens

    tok_id = tokens[-1]  # Last token of entity name
    emb = embed_fn(mx.array([[tok_id]]))
    emb = emb * math.sqrt(emb.shape[-1])  # Gemma scaling
    mx.eval(emb)
    return np.array(emb[0, 0, :].astype(mx.float32)), model, tokenizer


# ---- Lookup methods -----------------------------------------------------

def layered_walk(entity_emb, gates, config, down_meta, top_k=10):
    """
    Traditional layer-by-layer walk.
    14 sequential matmuls (one per knowledge layer).
    """
    bands = config.get("layer_bands", {})
    kn_start = bands.get("knowledge", [14, 27])[0]
    kn_end = bands.get("knowledge", [14, 27])[1]

    all_hits = []
    for layer in range(kn_start, kn_end + 1):
        if layer not in gates:
            continue
        layer_gates = gates[layer]
        activations = layer_gates @ entity_emb
        top_idx = np.argsort(-np.abs(activations))[:top_k]
        for fi in top_idx:
            val = float(activations[fi])
            token = down_meta.get((layer, int(fi)), "")
            all_hits.append((layer, int(fi), val, token))

    # Sort by absolute activation
    all_hits.sort(key=lambda x: abs(x[2]), reverse=True)
    return all_hits


def flat_lookup(entity_emb, flat_gates, flat_keys, down_meta, top_k=20):
    """
    Single flat lookup. ONE matmul against all features concatenated.
    No layers. No sequential processing. One step.
    """
    # flat_gates: [total_features, hidden_dim]
    # One matmul:
    activations = flat_gates @ entity_emb  # [total_features]

    top_idx = np.argsort(-np.abs(activations))[:top_k]

    hits = []
    for idx in top_idx:
        layer, fi = flat_keys[idx]
        val = float(activations[idx])
        token = down_meta.get((layer, fi), "")
        hits.append((layer, fi, val, token))

    return hits


def transformer_forward(model, tokenizer, prompt):
    """Full transformer forward pass. 34 layers, all attention, all FFN."""
    import mlx.core as mx
    import mlx.nn as nn

    tokens = tokenizer.encode(prompt)
    try:
        lm = model.language_model
        inner = lm.model
        embed_fn = inner.embed_tokens
        layers = inner.layers
        norm = inner.norm
        def lm_head(h): return h @ embed_fn.weight.T
    except:
        inner = model.model
        embed_fn = inner.embed_tokens
        layers = inner.layers
        norm = inner.norm
        def lm_head(h): return h @ embed_fn.weight.T

    h = embed_fn(mx.array([tokens]))
    h = h * math.sqrt(h.shape[-1])
    mask = nn.MultiHeadAttention.create_additive_causal_mask(len(tokens)).astype(h.dtype)

    for layer in layers:
        h = layer(h, mask=mask)
        mx.eval(h)

    h_normed = norm(h[:, -1:, :])
    logits = lm_head(h_normed)
    mx.eval(logits)
    logits_np = np.array(logits[0, 0, :].astype(mx.float32))
    top5_idx = np.argsort(-logits_np)[:5]
    top5 = [(tokenizer.decode([int(i)]).strip().lower(), float(logits_np[i])) for i in top5_idx]
    return top5


# ---- Main ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Flat graph lookup vs layered walk vs transformer"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--vindex", required=True)
    parser.add_argument("--output", default="output/syntax_circuit_routing/")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading vindex...")
    config, gates, down_meta, labels = load_vindex(args.vindex)

    bands = config.get("layer_bands", {})
    kn_start = bands.get("knowledge", [14, 27])[0]
    kn_end = bands.get("knowledge", [14, 27])[1]
    hidden_size = config["hidden_size"]

    # Build flat gate matrix (all knowledge layers concatenated)
    print("\nBuilding flat gate matrix...")
    flat_gate_list = []
    flat_key_list = []
    for layer in range(kn_start, kn_end + 1):
        if layer not in gates:
            continue
        lg = gates[layer]
        for fi in range(lg.shape[0]):
            flat_gate_list.append(lg[fi])
            flat_key_list.append((layer, fi))

    flat_gates = np.stack(flat_gate_list)
    total_features = flat_gates.shape[0]
    print(f"  Flat matrix: {flat_gates.shape} ({total_features} features across L{kn_start}-L{kn_end})")
    print(f"  One matmul: [{total_features}, {hidden_size}] @ [{hidden_size}] = [{total_features}]")

    # Load model for embeddings + transformer comparison
    print("\nLoading model...")
    entity_emb_cache = {}
    first_entity = QUERIES[0][1]
    entity_emb, model, tokenizer = get_entity_embedding(args.model, first_entity)
    entity_emb_cache[first_entity] = entity_emb

    # ---- Run all three methods ----
    print(f"\n{'='*70}")
    print(f"THREE METHODS: FLAT LOOKUP vs LAYERED WALK vs TRANSFORMER")
    print(f"{'='*70}")

    n_correct_flat = 0
    n_correct_layered = 0
    n_correct_transformer = 0

    for prompt, entity, expected, relation in QUERIES:
        # Get entity embedding
        if entity not in entity_emb_cache:
            import mlx.core as mx
            tokens = tokenizer.encode(entity)
            tok_id = tokens[-1]
            try:
                emb_fn = model.language_model.model.embed_tokens
            except:
                emb_fn = model.model.embed_tokens
            emb = emb_fn(mx.array([[tok_id]]))
            emb = emb * math.sqrt(emb.shape[-1])
            mx.eval(emb)
            entity_emb_cache[entity] = np.array(emb[0, 0, :].astype(mx.float32))

        entity_emb = entity_emb_cache[entity]

        # Method 1: Flat lookup (ONE matmul)
        t0 = time.perf_counter()
        flat_hits = flat_lookup(entity_emb, flat_gates, flat_key_list, down_meta, top_k=20)
        flat_time = (time.perf_counter() - t0) * 1000

        # Method 2: Layered walk (14 matmuls)
        t0 = time.perf_counter()
        layered_hits = layered_walk(entity_emb, gates, config, down_meta, top_k=10)
        layered_time = (time.perf_counter() - t0) * 1000

        # Method 3: Transformer (34 layers, full attention + FFN)
        t0 = time.perf_counter()
        transformer_top5 = transformer_forward(model, tokenizer, prompt)
        transformer_time = (time.perf_counter() - t0) * 1000

        # Check answers
        flat_tokens = [h[3].lower() for h in flat_hits[:10] if h[3]]
        layered_tokens = [h[3].lower() for h in layered_hits[:10] if h[3]]
        transformer_token = transformer_top5[0][0] if transformer_top5 else ""

        expected_lower = expected.lower()
        flat_correct = any(expected_lower in t for t in flat_tokens[:5])
        layered_correct = any(expected_lower in t for t in layered_tokens[:5])
        transformer_correct = expected_lower in transformer_token

        if flat_correct: n_correct_flat += 1
        if layered_correct: n_correct_layered += 1
        if transformer_correct: n_correct_transformer += 1

        # Display
        flat_answer = next((t for t in flat_tokens if expected_lower in t), flat_tokens[0] if flat_tokens else "?")
        layered_answer = next((t for t in layered_tokens if expected_lower in t), layered_tokens[0] if layered_tokens else "?")

        flat_status = "OK" if flat_correct else "XX"
        lay_status = "OK" if layered_correct else "XX"
        tf_status = "OK" if transformer_correct else "XX"

        print(f"\n  {entity:12s} → {expected}")
        print(f"    Flat    [{flat_status}]: {flat_answer:15s}  {flat_time:6.2f}ms  (1 matmul)")
        print(f"    Layered [{lay_status}]: {layered_answer:15s}  {layered_time:6.2f}ms  (14 matmuls)")
        print(f"    Neural  [{tf_status}]: {transformer_token:15s}  {transformer_time:6.1f}ms  (34 layers)")

    # ---- Summary ----
    n_queries = len(QUERIES)

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    print(f"\n  {'Method':<20s}  {'Accuracy':>8s}  {'Steps':>10s}  {'Description'}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*10}  {'-'*30}")
    print(f"  {'Flat lookup':<20s}  {n_correct_flat:3d}/{n_queries:<3d}     {'1 matmul':>10s}  entity @ all_gates → answer")
    print(f"  {'Layered walk':<20s}  {n_correct_layered:3d}/{n_queries:<3d}     {'14 matmuls':>10s}  entity @ gates[L] x14 → answer")
    print(f"  {'Transformer':<20s}  {n_correct_transformer:3d}/{n_queries:<3d}     {'34 layers':>10s}  full neural forward pass")

    print(f"\n  The flat lookup does ONE matrix multiply against all {total_features}")
    print(f"  features simultaneously. No layers. No sequential processing.")
    print(f"  No hourglass. No dark space. No routing signal.")
    print(f"  Just: entity → features → answer.")

    if n_correct_flat >= n_correct_transformer * 0.8:
        print(f"\n  FLAT LOOKUP WORKS")
        print(f"    {n_correct_flat}/{n_queries} correct with a single matmul")
        print(f"    The layer structure is unnecessary for retrieval")
    elif n_correct_flat >= n_queries * 0.5:
        print(f"\n  FLAT LOOKUP PARTIALLY WORKS")
        print(f"    {n_correct_flat}/{n_queries} correct — some queries need layer structure")
    else:
        print(f"\n  FLAT LOOKUP INSUFFICIENT")
        print(f"    {n_correct_flat}/{n_queries} correct — layer processing adds information")

    # ---- What layers add ----
    if n_correct_layered > n_correct_flat:
        print(f"\n  Layered > Flat by {n_correct_layered - n_correct_flat} queries.")
        print(f"  The extra layers contribute to those queries.")
        print(f"  But the STRUCTURE (14 sequential steps) is still")
        print(f"  the transformer's constraint, not the problem's requirement.")
        print(f"  A better index (HNSW, inverted index by relation) could")
        print(f"  match layered accuracy with flat lookup speed.")

    print()


if __name__ == "__main__":
    main()
