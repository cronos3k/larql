#!/usr/bin/env python3
"""Compute attention-approximated residuals for entities.

For a single token, self-attention reduces to a linear projection:
  output = embedding × W_V × W_O (per head, per layer)
  approximate_residual = embedding + Σ(OV outputs across layers)

Then project the approximate residual against gate vectors to see
if the factual features activate.

Usage:
    python scripts/attention_residual.py \
        --vectors output/vectors \
        --entities "Germany,Spain,Paris,Mozart,Amsterdam" \
        --layers-ov 0-25 \
        --layers-gate 26
"""

import argparse
import json
import os
import sys
import time

import numpy as np


def load_embeddings(path):
    """Load embedding matrix and token list."""
    vecs = []
    tokens = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line.strip())
            if '_header' in obj:
                continue
            vecs.append(obj['vector'])
            tokens.append(obj.get('top_token', ''))
    return np.array(vecs, dtype=np.float32), tokens


_weight_cache = {}

def load_weight_matrix(safetensors_dir, key):
    """Load a specific weight matrix from safetensors files via MLX (handles bf16)."""
    import mlx.core as mx

    if key in _weight_cache:
        return _weight_cache[key]

    st_files = sorted(f for f in os.listdir(safetensors_dir) if f.endswith('.safetensors'))
    for st_file in st_files:
        path = os.path.join(safetensors_dir, st_file)
        tensors = mx.load(path)
        for prefix in ['language_model.model.', 'model.', '']:
            full_key = prefix + key
            if full_key in tensors:
                arr = tensors[full_key].astype(mx.float32)
                mx.eval(arr)
                result = np.array(arr)
                _weight_cache[key] = result
                return result
    return None


def compute_ov_projection(embedding, w_v, w_o, head_dim, num_kv_heads):
    """Compute OV projection for all heads: embedding × W_V × W_O.

    For single-token self-attention, attention weight = 1.0,
    so output = V × W_O where V = embedding × W_V.
    """
    # V = embedding × W_V.T → (num_kv_heads * head_dim,)
    v = embedding @ w_v.T

    # For each head, extract V_h and multiply by O_h
    hidden = w_o.shape[0]
    output = np.zeros(hidden, dtype=np.float32)

    # Assuming GQA: num_q_heads = w_o.shape[1] / head_dim
    num_q_heads = w_o.shape[1] // head_dim
    reps = num_q_heads // num_kv_heads

    for kv_h in range(num_kv_heads):
        v_h = v[kv_h * head_dim:(kv_h + 1) * head_dim]
        # Each KV head maps to `reps` Q heads
        for r in range(reps):
            q_h = kv_h * reps + r
            o_h = w_o[:, q_h * head_dim:(q_h + 1) * head_dim]  # (hidden, head_dim)
            output += o_h @ v_h

    return output


def main():
    parser = argparse.ArgumentParser(description="Compute attention-approximated residuals")
    parser.add_argument("--vectors", required=True, help="Vectors directory")
    parser.add_argument("--model-dir", default=None, help="Model safetensors directory (auto-detected from HF cache)")
    parser.add_argument("--model", default="google/gemma-3-4b-it", help="Model name for HF cache lookup")
    parser.add_argument("--entities", required=True)
    parser.add_argument("--layers-ov", default="0-25", help="Layers to accumulate OV projections")
    parser.add_argument("--layers-gate", default="26", help="Layer to query gate vectors against")
    parser.add_argument("--edges-dir", default="output/edges")
    args = parser.parse_args()

    entities = [e.strip() for e in args.entities.split(",")]

    # Parse layer ranges
    ov_layers = []
    for part in args.layers_ov.split(","):
        if "-" in part:
            a, b = part.split("-")
            ov_layers.extend(range(int(a), int(b) + 1))
        else:
            ov_layers.append(int(part))

    gate_layer = int(args.layers_gate)

    # Find model directory
    model_dir = args.model_dir
    if model_dir is None:
        cache_name = f"models--{args.model.replace('/', '--')}"
        hf_cache = os.path.expanduser(f"~/.cache/huggingface/hub/{cache_name}/snapshots")
        if os.path.isdir(hf_cache):
            model_dir = next(
                (os.path.join(hf_cache, d) for d in os.listdir(hf_cache)
                 if os.path.isdir(os.path.join(hf_cache, d))),
                None
            )
    if not model_dir:
        print("ERROR: Could not find model directory", file=sys.stderr)
        sys.exit(1)

    # Load config
    config = json.load(open(os.path.join(model_dir, 'config.json')))
    tc = config.get('text_config', config)
    hidden_size = tc.get('hidden_size', 2560)
    head_dim = tc.get('head_dim', 256)
    num_kv_heads = tc.get('num_key_value_heads', 4)

    print(f"Model: {model_dir}", file=sys.stderr)
    print(f"  hidden={hidden_size}, head_dim={head_dim}, kv_heads={num_kv_heads}", file=sys.stderr)

    # Load embeddings
    print("Loading embeddings...", file=sys.stderr)
    embed_matrix, embed_tokens = load_embeddings(os.path.join(args.vectors, "embeddings.vectors.jsonl"))
    embed_norms = np.linalg.norm(embed_matrix, axis=1, keepdims=True)
    embed_normed = embed_matrix / np.maximum(embed_norms, 1e-8)
    token_to_idx = {t: i for i, t in enumerate(embed_tokens)}

    # Load gate vectors for the target layer
    print(f"Loading gate vectors for L{gate_layer}...", file=sys.stderr)
    gate_vecs = {}
    with open(os.path.join(args.vectors, "ffn_gate.vectors.jsonl")) as f:
        for line in f:
            obj = json.loads(line.strip())
            if '_header' in obj: continue
            if obj.get('layer') == gate_layer:
                gate_vecs[obj['feature']] = (
                    np.array(obj['vector'], dtype=np.float32),
                    obj.get('top_token', '')
                )
    print(f"  {len(gate_vecs)} gate vectors", file=sys.stderr)

    # Load edge data for comparison
    edge_features = {}
    try:
        with open(f"{args.edges_dir}/L{gate_layer}_edges.jsonl") as f:
            for line in f:
                e = json.loads(line)
                edge_features[e['feature']] = e
    except FileNotFoundError:
        pass

    # For each entity, compute attention-approximated residual
    print(f"\nComputing approximate residuals ({len(ov_layers)} OV layers → L{gate_layer} gates)...", file=sys.stderr)

    for entity in entities:
        if entity not in token_to_idx:
            print(f"\n{entity}: not found in vocabulary", file=sys.stderr)
            continue

        idx = token_to_idx[entity]
        embedding = embed_matrix[idx] * np.sqrt(hidden_size)  # Gemma scaling

        # Accumulate OV projections across layers
        residual = embedding.copy()
        for layer in ov_layers:
            w_v = load_weight_matrix(model_dir, f"layers.{layer}.self_attn.v_proj.weight")
            w_o = load_weight_matrix(model_dir, f"layers.{layer}.self_attn.o_proj.weight")
            if w_v is None or w_o is None:
                continue
            ov_out = compute_ov_projection(embedding, w_v, w_o, head_dim, num_kv_heads)
            residual += ov_out

        # Normalize
        res_norm = np.linalg.norm(residual)
        res_normed = residual / max(res_norm, 1e-8)

        # Project against gate vectors
        gate_matrix = np.zeros((len(gate_vecs), hidden_size), dtype=np.float32)
        gate_features = sorted(gate_vecs.keys())
        for i, feat in enumerate(gate_features):
            gate_matrix[i] = gate_vecs[feat][0]
        gate_norms = np.linalg.norm(gate_matrix, axis=1, keepdims=True)
        gate_normed = gate_matrix / np.maximum(gate_norms, 1e-8)

        # Cosine distances
        sims = res_normed @ gate_normed.T
        dists = 1.0 - sims

        # Top-10 nearest gates
        top_idx = np.argpartition(dists, 10)[:10]
        top_idx = top_idx[np.argsort(dists[top_idx])]

        # Also compute raw embedding distance for comparison
        raw_normed = (embedding / max(np.linalg.norm(embedding), 1e-8))
        raw_sims = raw_normed @ gate_normed.T
        raw_dists = 1.0 - raw_sims
        raw_top = np.argpartition(raw_dists, 3)[:3]
        raw_top = raw_top[np.argsort(raw_dists[raw_top])]

        print(f"\n{'='*80}")
        print(f"{entity} (embedding idx={idx})")
        print(f"  Raw embedding → gate (no attention):")
        for i in raw_top:
            feat = gate_features[i]
            gt = gate_vecs[feat][1]
            edge = edge_features.get(feat, {})
            dt = edge.get('target', '?')
            print(f"    F{feat:5d}: gate={gt:15s} → {dt:15s}  dist={raw_dists[i]:.3f}")

        print(f"  Attention-approximated residual → gate ({len(ov_layers)} OV layers):")
        for i in top_idx:
            feat = gate_features[i]
            gt = gate_vecs[feat][1]
            edge = edge_features.get(feat, {})
            dt = edge.get('target', '?')
            print(f"    F{feat:5d}: gate={gt:15s} → {dt:15s}  dist={dists[i]:.3f}")


if __name__ == "__main__":
    main()
