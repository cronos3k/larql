#!/usr/bin/env python3
"""Fast edge discovery: batch cosine similarity in NumPy.

Reads gate/down/embedding NDJSON files directly.
All 34 layers in under an hour.

Usage:
    python scripts/edge_discover_fast.py \
        --vectors output/vectors \
        --output output/edges \
        --layers 0-33

    # Single layer
    python scripts/edge_discover_fast.py --vectors output/vectors --output output/edges --layers 26
"""

import argparse
import json
import os
import sys
import time

import numpy as np


def load_ndjson_vectors(path, layer_filter=None):
    """Load vectors from NDJSON file, optionally filtered by layer.
    Returns dict of {feature_id: (vector, top_token, c_score)}
    """
    vectors = {}
    scanned = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Quick layer check before full parse (avoid parsing 300K+ lines)
            if layer_filter is not None:
                layer_key = f'"layer":{layer_filter},'
                layer_key2 = f'"layer": {layer_filter},'
                if layer_key not in line and layer_key2 not in line:
                    scanned += 1
                    if scanned % 100000 == 0:
                        print(f"\r    Scanning: {scanned} lines, {len(vectors)} matched...", end="", file=sys.stderr)
                    continue
            obj = json.loads(line)
            if "_header" in obj:
                continue
            if layer_filter is not None and obj.get("layer") != layer_filter:
                scanned += 1
                continue
            feat = obj["feature"]
            vectors[feat] = (
                np.array(obj["vector"], dtype=np.float32),
                obj.get("top_token", ""),
                obj.get("c_score", 0.0),
            )
            scanned += 1
            if scanned % 100000 == 0:
                print(f"\r    Scanning: {scanned} lines, {len(vectors)} matched...", end="", file=sys.stderr)
    if scanned > 50000:
        print(f"\r    Scanned {scanned} lines, loaded {len(vectors)} vectors", file=sys.stderr)
    return vectors


def load_embeddings(path):
    """Load all embeddings. Returns (matrix, token_list).
    matrix: (vocab_size, hidden_dim)
    token_list: list of token strings indexed by position
    """
    vecs = []
    tokens = []
    count = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "_header" in obj:
                continue
            vecs.append(obj["vector"])
            tokens.append(obj.get("top_token", ""))
            count += 1
            if count % 50000 == 0:
                print(f"\r  Loading embeddings: {count}...", end="", file=sys.stderr)

    print(f"\r  Loading embeddings: {count} done, building matrix...", file=sys.stderr)
    matrix = np.array(vecs, dtype=np.float32)
    return matrix, tokens


def cosine_distances(queries, targets, target_norms):
    """Batch cosine distance: 1 - cosine_similarity.
    queries: (N, D), targets: (M, D), target_norms: (M,)
    Returns: (N, M) distance matrix
    """
    query_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    query_norms = np.maximum(query_norms, 1e-8)
    queries_normed = queries / query_norms

    # targets already normalized
    similarities = queries_normed @ targets.T  # (N, M)
    return 1.0 - similarities


def top_k_per_row(distances, k):
    """For each row, find the k smallest distances. Returns (indices, values)."""
    if k >= distances.shape[1]:
        indices = np.argsort(distances, axis=1)
        return indices, np.take_along_axis(distances, indices, axis=1)
    indices = np.argpartition(distances, k, axis=1)[:, :k]
    values = np.take_along_axis(distances, indices, axis=1)
    # Sort within top-k
    sort_idx = np.argsort(values, axis=1)
    indices = np.take_along_axis(indices, sort_idx, axis=1)
    values = np.take_along_axis(values, sort_idx, axis=1)
    return indices, values


def process_layer(
    layer, gate_path, down_path, embed_matrix, embed_norms, embed_tokens,
    circuits_dir, top_k=3,
):
    """Process one layer. Returns list of edge dicts."""

    print(f"  Loading gate vectors for L{layer}...", file=sys.stderr)
    gates = load_ndjson_vectors(gate_path, layer_filter=layer)
    print(f"  Loading down vectors for L{layer}...", file=sys.stderr)
    downs = load_ndjson_vectors(down_path, layer_filter=layer)

    n_features = max(max(gates.keys(), default=0), max(downs.keys(), default=0)) + 1
    if n_features == 0:
        return []

    # Build matrices
    hidden_dim = embed_matrix.shape[1]
    gate_matrix = np.zeros((n_features, hidden_dim), dtype=np.float32)
    down_matrix = np.zeros((n_features, hidden_dim), dtype=np.float32)
    gate_tokens = [""] * n_features
    down_tokens = [""] * n_features

    for feat, (vec, tok, _) in gates.items():
        if feat < n_features:
            gate_matrix[feat] = vec
            gate_tokens[feat] = tok

    for feat, (vec, tok, _) in downs.items():
        if feat < n_features:
            down_matrix[feat] = vec
            down_tokens[feat] = tok

    # Normalize embeddings once
    embed_normed = embed_matrix / np.maximum(embed_norms[:, None], 1e-8)

    # Batch cosine distance
    print(f"  Computing gate KNN ({n_features} × {embed_matrix.shape[0]})...", file=sys.stderr)
    gate_dists = cosine_distances(gate_matrix, embed_normed, embed_norms)
    print(f"  Computing down KNN ({n_features} × {embed_matrix.shape[0]})...", file=sys.stderr)
    down_dists = cosine_distances(down_matrix, embed_normed, embed_norms)

    # Top-k
    gate_top_idx, gate_top_dist = top_k_per_row(gate_dists, top_k)
    down_top_idx, down_top_dist = top_k_per_row(down_dists, top_k)

    # Load circuit classifications if available
    circuit_data = {}
    circuit_path = os.path.join(circuits_dir, f"L{layer}_circuits.json")
    if os.path.exists(circuit_path):
        data = json.load(open(circuit_path))
        circuit_data = {f["feature"]: f["circuit_type"] for f in data["features"]}

    # Emit edges
    edges = []
    for feat in range(n_features):
        gate_nearest_token = embed_tokens[gate_top_idx[feat, 0]]
        gate_nearest_dist = float(gate_top_dist[feat, 0])
        down_nearest_token = embed_tokens[down_top_idx[feat, 0]]
        down_nearest_dist = float(down_top_dist[feat, 0])

        # Top-k lists
        gate_top = [
            {"token": embed_tokens[gate_top_idx[feat, k]], "dist": round(float(gate_top_dist[feat, k]), 6)}
            for k in range(min(top_k, gate_top_idx.shape[1]))
        ]
        down_top = [
            {"token": embed_tokens[down_top_idx[feat, k]], "dist": round(float(down_top_dist[feat, k]), 6)}
            for k in range(min(top_k, down_top_idx.shape[1]))
        ]

        edge = {
            "source": gate_nearest_token,
            "target": down_nearest_token,
            "relation": f"L{layer}-F{feat}",
            "layer": layer,
            "feature": feat,
            "gate_dist": round(gate_nearest_dist, 6),
            "down_dist": round(down_nearest_dist, 6),
            "gate_top": gate_top,
            "down_top": down_top,
            "circuit_type": circuit_data.get(feat, "unknown"),
            "gate_token_original": gate_tokens[feat],
            "down_token_original": down_tokens[feat],
        }
        edges.append(edge)

    return edges


def parse_layers(s):
    """Parse '26' or '0-33' or '23,24,25,26'."""
    layers = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(part))
    return layers


def main():
    parser = argparse.ArgumentParser(description="Fast edge discovery via batch cosine similarity")
    parser.add_argument("--vectors", required=True, help="Directory with .vectors.jsonl files")
    parser.add_argument("--output", required=True, help="Output directory for edge JSONL files")
    parser.add_argument("--layers", default="26", help="Layers: '26' or '0-33' or '23,24,25'")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k embedding matches per feature")
    parser.add_argument("--circuits-dir", default="output/circuits", help="Circuit classification dir")
    args = parser.parse_args()

    layers = parse_layers(args.layers)
    os.makedirs(args.output, exist_ok=True)

    # Load embeddings once
    embed_path = os.path.join(args.vectors, "embeddings.vectors.jsonl")
    print(f"Loading embeddings from {embed_path}...", file=sys.stderr)
    embed_start = time.time()
    embed_matrix, embed_tokens = load_embeddings(embed_path)
    embed_norms = np.linalg.norm(embed_matrix, axis=1)
    print(
        f"  {embed_matrix.shape[0]} tokens, dim={embed_matrix.shape[1]} "
        f"({time.time() - embed_start:.1f}s, {embed_matrix.nbytes / 1024**3:.1f} GB)",
        file=sys.stderr,
    )

    gate_path = os.path.join(args.vectors, "ffn_gate.vectors.jsonl")
    down_path = os.path.join(args.vectors, "ffn_down.vectors.jsonl")

    overall_start = time.time()
    total_edges = 0
    total_clean = 0

    for layer in layers:
        layer_start = time.time()
        print(f"\nLayer {layer}:", file=sys.stderr)

        edges = process_layer(
            layer, gate_path, down_path,
            embed_matrix, embed_norms, embed_tokens,
            args.circuits_dir, args.top_k,
        )

        out_path = os.path.join(args.output, f"L{layer}_edges.jsonl")
        with open(out_path, "w") as f:
            for edge in edges:
                f.write(json.dumps(edge) + "\n")

        clean = sum(1 for e in edges if e["down_dist"] < 0.8)
        total_edges += len(edges)
        total_clean += clean

        elapsed = time.time() - layer_start
        print(
            f"  {len(edges)} edges, {clean} clean (down_dist < 0.8), "
            f"{elapsed:.1f}s → {out_path}",
            file=sys.stderr,
        )

    elapsed = time.time() - overall_start
    print(f"\nTotal: {total_edges} edges ({total_clean} clean) in {elapsed:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
