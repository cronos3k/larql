#!/usr/bin/env python3
"""
syntax_relation_detect.py — Can syntax-layer features detect the relation?

The syntax layers (L0-13) have 1,500+ labeled features including:
  wn:hypernym, wn:synonym, capital, birthplace, occupation, currency...

If "The capital of France is" activates the "capital" labeled features
in syntax layers, we can read the relation directly from gate activations.
No q_proj. No centroids. No L14+ processing.

Just: tokenize → embed → project through L0-13 gates → read relation label
→ filtered walk(entity, relation) → answer.

USAGE:
  python3 experiments/05_syntax_circuit_routing/syntax_relation_detect.py \
      --model google/gemma-3-4b-it \
      --vindex output/gemma3-4b-f16.vindex
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np

import mlx.core as mx
import mlx.nn as nn


# ---- Test prompts with known relations ----------------------------------

QUERIES = [
    # (prompt, entity, expected_relation, expected_answer)
    ("The capital of France is", "France", "capital", "Paris"),
    ("The capital of Japan is", "Japan", "capital", "Tokyo"),
    ("The capital of Germany is", "Germany", "capital", "Berlin"),
    ("The capital of Egypt is", "Egypt", "capital", "Cairo"),
    ("The capital of India is", "India", "capital", "Delhi"),

    ("The official language of France is", "France", "language", "French"),
    ("The official language of Japan is", "Japan", "language", "Japanese"),
    ("The official language of Brazil is", "Brazil", "language", "Portuguese"),

    ("The currency of Japan is the", "Japan", "currency", "yen"),
    ("The currency of India is the", "India", "currency", "rupee"),
    ("The currency of Sweden is the", "Sweden", "currency", "krona"),

    ("Einstein was born in", "Einstein", "birthplace", "Ulm"),
    ("Shakespeare was born in", "Shakespeare", "birthplace", "Stratford"),
    ("Mozart was born in", "Mozart", "birthplace", "Salzburg"),

    ("The occupation of Einstein was", "Einstein", "occupation", "physicist"),
    ("The occupation of Shakespeare was", "Shakespeare", "occupation", "playwright"),
    ("The occupation of Mozart was", "Mozart", "occupation", "composer"),

    ("A dog is a type of", "dog", "wn:hypernym", "animal"),
    ("A rose is a type of", "rose", "wn:hypernym", "flower"),
    ("A piano is a type of", "piano", "wn:hypernym", "instrument"),

    ("Happy means", "Happy", "wn:synonym", "joyful"),
    ("Sad means", "Sad", "wn:synonym", "unhappy"),
    ("Big means", "Big", "wn:synonym", "large"),

    ("France is located in", "France", "continent", "Europe"),
    ("Japan is located in", "Japan", "continent", "Asia"),
    ("Brazil is located in", "Brazil", "continent", "South America"),

    # Natural language variants
    ("What is the capital of France?", "France", "capital", "Paris"),
    ("What language do people speak in Japan?", "Japan", "language", "Japanese"),
    ("Where was Einstein born?", "Einstein", "birthplace", "Ulm"),
    ("What did Mozart do?", "Mozart", "occupation", "composer"),
]


# ---- Loading ------------------------------------------------------------

def find_model_parts(model):
    try:
        lm = model.language_model
        inner = lm.model
        if hasattr(inner, 'embed_tokens') and hasattr(inner, 'layers'):
            embed_fn = inner.embed_tokens
            def lm_head(h): return h @ embed_fn.weight.T
            return embed_fn, inner.layers, inner.norm, lm_head, True
    except AttributeError:
        pass
    inner = getattr(model, 'model', None)
    if inner and hasattr(inner, 'embed_tokens') and hasattr(inner, 'layers'):
        embed_fn = inner.embed_tokens
        if hasattr(model, 'lm_head'):
            f = model.lm_head
            def lm_head(h): return f(h)
        else:
            def lm_head(h): return h @ embed_fn.weight.T
        model_type = getattr(getattr(model, 'config', None), 'model_type', '')
        needs_scale = 'gemma' in str(model_type).lower()
        return embed_fn, inner.layers, inner.norm, lm_head, needs_scale
    raise RuntimeError("Could not detect model structure.")


def load_vindex(vindex_path):
    vindex_path = Path(vindex_path)
    with open(vindex_path / "index.json") as f:
        config = json.load(f)
    hidden_size = config["hidden_size"]

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

    labels = {}
    labels_path = vindex_path / "feature_labels.json"
    if labels_path.exists():
        with open(labels_path) as f:
            labels = json.load(f)

    return config, gates, labels


def capture_residuals(model, tokenizer, prompt, max_layer=13):
    """Capture residuals at syntax layers only."""
    embed_fn, layers, norm, lm_head, needs_scale = find_model_parts(model)

    tokens = tokenizer.encode(prompt)
    h = embed_fn(mx.array([tokens]))
    if needs_scale:
        h = h * math.sqrt(h.shape[-1])
    mask = nn.MultiHeadAttention.create_additive_causal_mask(len(tokens)).astype(h.dtype)

    residuals = {}
    mx.eval(h)
    residuals[-1] = np.array(h[0, -1, :].astype(mx.float32))

    for i, layer in enumerate(layers):
        h = layer(h, mask=mask)
        mx.eval(h)
        residuals[i] = np.array(h[0, -1, :].astype(mx.float32))
        if i >= max_layer:
            break

    return residuals


# ---- Syntax relation detection -----------------------------------------

def detect_relation_from_syntax(residuals, gates, labels, config,
                                 syntax_range=range(0, 14)):
    """
    Project residuals through syntax-layer gates.
    Find which LABELED features fire.
    The labels ARE the relation type.
    """

    # Collect all labeled feature activations
    label_activations = defaultdict(list)  # relation_label -> [(layer, feat, activation)]

    for layer in syntax_range:
        if layer not in residuals or layer not in gates:
            continue

        res = residuals[layer]
        layer_gates = gates[layer]
        acts = layer_gates @ res  # [n_features]

        # Check every labeled feature
        for feat_id in range(len(acts)):
            key = f"L{layer}_F{feat_id}"
            info = labels.get(key, {})
            if not isinstance(info, dict):
                continue
            rel = info.get("relation", "")
            if not rel or rel == "-":
                continue

            val = float(acts[feat_id])
            label_activations[rel].append((layer, feat_id, val))

    # For each relation label, compute aggregate signal
    relation_scores = {}
    for rel, activations in label_activations.items():
        # Use mean absolute activation as score
        abs_acts = [abs(a[2]) for a in activations]
        relation_scores[rel] = {
            "mean_abs": float(np.mean(abs_acts)),
            "max_abs": float(np.max(abs_acts)),
            "sum_abs": float(np.sum(abs_acts)),
            "n_features": len(activations),
            "top_features": sorted(activations, key=lambda x: abs(x[2]), reverse=True)[:5],
        }

    # Rank by mean absolute activation
    ranked = sorted(relation_scores.items(), key=lambda x: x[1]["mean_abs"], reverse=True)

    return ranked, relation_scores


# ---- Main ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Detect query relation from syntax-layer features"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--vindex", required=True)
    parser.add_argument("--output", default="output/syntax_circuit_routing/")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading vindex...")
    config, gates, labels = load_vindex(args.vindex)

    bands = config.get("layer_bands", {})
    syntax_end = bands.get("syntax", [0, 13])[1]
    syntax_range = range(0, syntax_end + 1)

    # Count labeled features in syntax layers
    syntax_labels = defaultdict(int)
    for key, info in labels.items():
        if not isinstance(info, dict):
            continue
        parts = key.split("_")
        if len(parts) != 2:
            continue
        layer = int(parts[0][1:])
        if layer <= syntax_end:
            rel = info.get("relation", "")
            if rel and rel != "-":
                syntax_labels[rel] += 1

    print(f"  Labeled syntax features by relation:")
    for rel, count in sorted(syntax_labels.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"    {rel:25s}: {count:3d} features")
    print(f"  Total: {sum(syntax_labels.values())} labeled features in L0-L{syntax_end}")

    print("\nLoading model...")
    from mlx_lm import load as mlx_load
    model, tokenizer = mlx_load(args.model)

    # ---- Run detection ----
    print(f"\n{'='*70}")
    print(f"SYNTAX RELATION DETECTION")
    print(f"{'='*70}")

    n_top1 = 0
    n_top3 = 0
    n_top5 = 0
    total = len(QUERIES)

    for prompt, entity, expected_rel, expected_answer in QUERIES:
        residuals = capture_residuals(model, tokenizer, prompt, max_layer=syntax_end)

        ranked, scores = detect_relation_from_syntax(
            residuals, gates, labels, config, syntax_range
        )

        # Check if expected relation is in top-K
        top_rels = [r[0] for r in ranked[:20]]
        top1 = top_rels[0] if top_rels else "?"
        top3 = top_rels[:3]
        top5 = top_rels[:5]

        hit1 = expected_rel in [top1]
        hit3 = expected_rel in top3
        hit5 = expected_rel in top5

        if hit1: n_top1 += 1
        if hit3: n_top3 += 1
        if hit5: n_top5 += 1

        # Find rank of expected relation
        rank = top_rels.index(expected_rel) + 1 if expected_rel in top_rels else -1

        status = "OK" if hit1 else (f"#{rank}" if rank > 0 else "XX")
        prompt_short = prompt[:45] + ("..." if len(prompt) > 45 else "")

        print(f"\n  [{status:>3s}] \"{prompt_short}\"")
        print(f"         expected: {expected_rel}")
        print(f"         top-5: {', '.join(top5)}")

        if expected_rel in scores:
            s = scores[expected_rel]
            print(f"         {expected_rel}: mean={s['mean_abs']:.1f}  "
                  f"max={s['max_abs']:.1f}  n={s['n_features']}  rank={rank}")

        # Show top-1 score for comparison
        if ranked:
            top_rel, top_score = ranked[0]
            print(f"         top-1 ({top_rel}): mean={top_score['mean_abs']:.1f}  "
                  f"max={top_score['max_abs']:.1f}  n={top_score['n_features']}")

    # ---- Summary ----
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    print(f"\n  Total queries: {total}")
    print(f"  Top-1 accuracy: {n_top1}/{total} ({n_top1/total:.0%})")
    print(f"  Top-3 accuracy: {n_top3}/{total} ({n_top3/total:.0%})")
    print(f"  Top-5 accuracy: {n_top5}/{total} ({n_top5/total:.0%})")

    # Per-relation breakdown
    print(f"\n  Per-relation:")
    rel_results = defaultdict(lambda: {"n": 0, "top1": 0, "top3": 0})
    for prompt, entity, expected_rel, expected_answer in QUERIES:
        residuals = capture_residuals(model, tokenizer, prompt, max_layer=syntax_end)
        ranked, _ = detect_relation_from_syntax(residuals, gates, labels, config, syntax_range)
        top_rels = [r[0] for r in ranked[:5]]

        rel_results[expected_rel]["n"] += 1
        if expected_rel in top_rels[:1]:
            rel_results[expected_rel]["top1"] += 1
        if expected_rel in top_rels[:3]:
            rel_results[expected_rel]["top3"] += 1

    for rel in sorted(rel_results.keys()):
        r = rel_results[rel]
        print(f"    {rel:20s}: {r['top1']}/{r['n']} top-1  {r['top3']}/{r['n']} top-3")

    # ---- Verdict ----
    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")

    if n_top1 / total >= 0.8:
        print(f"\n  SYNTAX FEATURES DETECT THE RELATION")
        print(f"    {n_top1}/{total} queries: correct relation at top-1")
        print(f"    No q_proj needed. No centroids. No L14+ processing.")
        print(f"    Pipeline: embed → syntax gates → read relation label → filtered walk → answer")
    elif n_top3 / total >= 0.8:
        print(f"\n  SYNTAX FEATURES PARTIALLY DETECT (top-3)")
        print(f"    {n_top3}/{total} in top-3, {n_top1}/{total} at top-1")
        print(f"    Relation is visible but noisy — may need scoring refinement")
    elif n_top5 / total >= 0.5:
        print(f"\n  WEAK RELATION SIGNAL IN SYNTAX FEATURES")
        print(f"    Signal exists but buried under generic features")
    else:
        print(f"\n  SYNTAX FEATURES DON'T RELIABLY DETECT RELATION")
        print(f"    The relation labels in syntax layers fire generically")
        print(f"    Still need q_proj routing or explicit query parsing")

    print()


if __name__ == "__main__":
    main()
