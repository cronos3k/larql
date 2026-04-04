#!/usr/bin/env python3
"""
darkspace_visible.py — Make the dark space routing signal visible

The routing signal lives in dark space (99.4% of residual, orthogonal to
unembedding). It's invisible to single-feature probes but readable by W_q.

This script uses W_q as a "lens" to look into dark space, layer by layer:

1. Capture residuals at every layer for prompts from 4 templates
2. Project through W_q SVD components → the "routing lens" → shows separation
3. Project through unembedding → the "content lens" → stays mixed
4. Track how the routing signal BUILDS across layers L0-L13
5. Show the 5 entity heads vs 107 template heads in Q-space

The output: the routing signal emerging from noise, layer by layer,
visible only through the right projection.

USAGE:
  python3 experiments/05_syntax_circuit_routing/darkspace_visible.py \
      --model google/gemma-3-4b-it \
      --vindex output/gemma3-4b-f16.vindex
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

import mlx.core as mx
import mlx.nn as nn


# ---- Prompts: 4 templates, 6 entities each -----------------------------

TEMPLATES = {
    "capital": {
        "color": "blue",
        "prompts": [
            "The capital of France is",
            "The capital of Japan is",
            "The capital of Brazil is",
            "The capital of Egypt is",
            "The capital of Germany is",
            "The capital of India is",
        ],
    },
    "synonym": {
        "color": "red",
        "prompts": [
            "Happy means",
            "Sad means",
            "Big means",
            "Fast means",
            "Brave means",
            "Cold means",
        ],
    },
    "arithmetic": {
        "color": "green",
        "prompts": [
            "2 + 3 =",
            "7 - 4 =",
            "5 * 6 =",
            "8 * 9 =",
            "15 + 27 =",
            "100 - 37 =",
        ],
    },
    "code": {
        "color": "purple",
        "prompts": [
            "def hello():\n    return",
            "def add(a, b):\n    return",
            "class Dog:\n    def __init__",
            "for i in range(10):\n    print",
            "if x > 0:\n    return",
            "fn main() {\n    let x =",
        ],
    },
}


# ---- Model helpers ------------------------------------------------------

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


def forward_all_residuals(model, tokenizer, prompt):
    """Capture residual at every layer + after embedding."""
    embed_fn, layers, norm, lm_head, needs_scale = find_model_parts(model)

    tokens = tokenizer.encode(prompt)
    h = embed_fn(mx.array([tokens]))
    if needs_scale:
        h = h * math.sqrt(h.shape[-1])
    seq_len = len(tokens)
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(h.dtype)

    residuals = {}
    # Layer -1 = after embedding, before any transformer layer
    mx.eval(h)
    residuals[-1] = np.array(h[0, -1, :].astype(mx.float32))

    for i, layer in enumerate(layers):
        h = layer(h, mask=mask)
        mx.eval(h)
        residuals[i] = np.array(h[0, -1, :].astype(mx.float32))

    # Prediction
    h_normed = norm(h[:, -1:, :])
    logits = lm_head(h_normed)
    mx.eval(logits)
    logits_np = np.array(logits[0, 0, :].astype(mx.float32))
    pred_id = int(np.argmax(logits_np))
    pred_tok = tokenizer.decode([pred_id]).strip()

    return residuals, pred_tok


# ---- Analysis -----------------------------------------------------------

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def compute_separation(vectors_by_cat, label=""):
    """Compute within-category vs between-category cosine similarity."""
    cats = list(vectors_by_cat.keys())
    within = []
    between = []

    for cat in cats:
        vecs = vectors_by_cat[cat]
        for i in range(len(vecs)):
            for j in range(i+1, len(vecs)):
                within.append(cosine(vecs[i], vecs[j]))

    for i, ca in enumerate(cats):
        for cb in cats[i+1:]:
            for va in vectors_by_cat[ca]:
                for vb in vectors_by_cat[cb]:
                    between.append(cosine(va, vb))

    w = np.mean(within) if within else 0
    b = np.mean(between) if between else 0
    return w, b, w - b


def main():
    parser = argparse.ArgumentParser(
        description="Make dark space routing signal visible"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--vindex", required=True)
    parser.add_argument("--layer", type=int, default=21,
                        help="Layer for W_q extraction")
    parser.add_argument("--output", default="output/syntax_circuit_routing/")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    from mlx_lm import load as mlx_load
    model, tokenizer = mlx_load(args.model)

    embed_fn, layers_list, norm_fn, lm_head_fn, needs_scale = find_model_parts(model)
    n_layers = len(layers_list)

    # Extract W_q at target layer
    target_layer = args.layer
    sa = layers_list[target_layer].self_attn
    wq = np.array(sa.q_proj.weight.astype(mx.float32))  # [2048, 2560]
    print(f"  W_q at L{target_layer}: {wq.shape}")

    # Extract unembedding matrix
    try:
        unembed = np.array(embed_fn.weight.astype(mx.float32))  # [vocab, 2560]
    except:
        unembed = None
    if unembed is not None:
        print(f"  Unembedding: {unembed.shape}")

    # SVD of W_q
    print("  Computing SVD of W_q...")
    U, S, Vt = np.linalg.svd(wq, full_matrices=False)
    # Vt: [2048, 2560] — right singular vectors in residual space
    # Project residual through top-K right singular vectors = routing lens

    # ---- Capture residuals ----
    total = sum(len(t["prompts"]) for t in TEMPLATES.values())
    print(f"\nCapturing residuals ({total} prompts, {n_layers+1} layers each)...")
    t0 = time.time()
    n = 0

    all_data = {}  # template -> [{residuals, prediction}, ...]
    for tname, tinfo in TEMPLATES.items():
        caps = []
        for prompt in tinfo["prompts"]:
            residuals, pred = forward_all_residuals(model, tokenizer, prompt)
            caps.append({"residuals": residuals, "prediction": pred, "prompt": prompt})
            n += 1
            print(f"\r  {n}/{total} ({time.time()-t0:.0f}s)", end="", flush=True)
        all_data[tname] = caps

    print(f"\n  Done in {time.time()-t0:.0f}s")

    # ---- Layer-by-layer analysis through different lenses ----
    all_layers = list(range(-1, n_layers))
    template_names = list(TEMPLATES.keys())

    print(f"\n{'='*70}")
    print(f"ROUTING SIGNAL EMERGENCE: LAYER BY LAYER")
    print(f"{'='*70}")

    # Three lenses:
    # 1. Raw residual space (no projection)
    # 2. W_q projection (routing lens)
    # 3. Unembedding projection (content lens)

    print(f"\n  {'Layer':>5s}  {'Raw gap':>8s}  {'W_q gap':>8s}  {'Unembed gap':>11s}  {'W_q within':>10s}")
    print(f"  {'-----':>5s}  {'--------':>8s}  {'--------':>8s}  {'-----------':>11s}  {'----------':>10s}")

    layer_data = []

    for layer in all_layers:
        # Collect residuals per template
        raw_vecs = {}
        wq_vecs = {}
        unembed_vecs = {}

        for tname in template_names:
            raw_list = []
            wq_list = []
            unembed_list = []

            for cap in all_data[tname]:
                res = cap["residuals"].get(layer)
                if res is None:
                    continue

                raw_list.append(res)

                # W_q projection: residual -> Q-space
                q_proj = wq @ res  # [2048]
                wq_list.append(q_proj)

                # Unembedding projection: residual -> vocab logits (top-K PCA)
                if unembed is not None:
                    # Project into top-50 vocab directions
                    u_proj = unembed @ res  # [vocab]
                    # Take top-50 by magnitude as feature vector
                    top_idx = np.argsort(-np.abs(u_proj))[:50]
                    unembed_list.append(u_proj[top_idx])

            raw_vecs[tname] = raw_list
            wq_vecs[tname] = wq_list
            if unembed_list:
                unembed_vecs[tname] = unembed_list

        # Compute separation in each space
        raw_w, raw_b, raw_gap = compute_separation(raw_vecs)
        wq_w, wq_b, wq_gap = compute_separation(wq_vecs)

        if unembed_vecs:
            ue_w, ue_b, ue_gap = compute_separation(unembed_vecs)
        else:
            ue_gap = 0

        layer_label = f"emb" if layer == -1 else f"L{layer:2d}"

        # Visual bar for W_q gap
        bar_len = int(max(0, wq_gap) * 100)
        bar = "#" * min(bar_len, 30)

        print(f"  {layer_label:>5s}  {raw_gap:+.4f}   {wq_gap:+.4f}   {ue_gap:+.4f}       {wq_w:.4f}     {bar}")

        layer_data.append({
            "layer": layer,
            "raw_gap": raw_gap,
            "wq_gap": wq_gap,
            "ue_gap": ue_gap,
            "wq_within": wq_w,
            "wq_between": wq_b,
        })

    # ---- PCA visualization of Q-space at key layers ----
    print(f"\n{'='*70}")
    print(f"Q-SPACE PCA AT KEY LAYERS")
    print(f"{'='*70}")

    key_layers = [-1, 0, 6, 12, 13, target_layer, n_layers - 1]
    key_layers = [l for l in key_layers if l < n_layers]

    for layer in key_layers:
        # Collect Q-projected vectors
        all_vecs = []
        all_labels = []
        for tname in template_names:
            for cap in all_data[tname]:
                res = cap["residuals"].get(layer)
                if res is not None:
                    q = wq @ res
                    all_vecs.append(q)
                    all_labels.append(tname)

        if len(all_vecs) < 4:
            continue

        X = np.stack(all_vecs)
        X_n = normalize(X)

        pca = PCA(n_components=2)
        coords = pca.fit_transform(X_n)

        layer_label = "embedding" if layer == -1 else f"L{layer}"
        print(f"\n  {layer_label} (explained var: {pca.explained_variance_ratio_[0]:.1%}, {pca.explained_variance_ratio_[1]:.1%}):")

        for tname in template_names:
            mask = [l == tname for l in all_labels]
            pts = coords[mask]
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            spread = np.sqrt(np.mean((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2))
            print(f"    {tname:12s}: center=({cx:+.3f}, {cy:+.3f})  spread={spread:.3f}")

    # ---- Dark space vs content space decomposition ----
    print(f"\n{'='*70}")
    print(f"DARK SPACE vs CONTENT SPACE DECOMPOSITION")
    print(f"{'='*70}")

    if unembed is not None:
        # Content subspace: span of top-K unembedding vectors
        # Dark subspace: orthogonal complement
        print(f"\n  Computing content/dark subspace split...")

        # Use SVD of unembedding to get content subspace basis
        U_emb, S_emb, Vt_emb = np.linalg.svd(unembed, full_matrices=False)
        # Vt_emb: [min(vocab, 2560), 2560] — right singular vectors
        # Top-K form the content subspace

        for n_content in [16, 64, 256]:
            content_basis = Vt_emb[:n_content]  # [n_content, 2560]

            print(f"\n  Content subspace rank = {n_content}:")

            for layer in [0, 6, 12, 13, target_layer]:
                if layer >= n_layers:
                    continue

                content_vecs = {}
                dark_vecs = {}

                for tname in template_names:
                    c_list = []
                    d_list = []
                    for cap in all_data[tname]:
                        res = cap["residuals"].get(layer)
                        if res is None:
                            continue

                        # Project into content subspace
                        content_proj = content_basis @ res  # [n_content]
                        content_reconstructed = content_basis.T @ content_proj  # [2560]

                        # Dark space = residual minus content projection
                        dark = res - content_reconstructed  # [2560]

                        c_list.append(content_proj)
                        d_list.append(dark)

                    content_vecs[tname] = c_list
                    dark_vecs[tname] = d_list

                # Separation in each subspace
                c_w, c_b, c_gap = compute_separation(content_vecs)
                d_w, d_b, d_gap = compute_separation(dark_vecs)

                # Also: project dark space through W_q
                dark_wq_vecs = {}
                for tname in template_names:
                    dark_wq_vecs[tname] = [wq @ d for d in dark_vecs[tname]]
                dw_w, dw_b, dw_gap = compute_separation(dark_wq_vecs)

                print(f"    L{layer:2d}: content_gap={c_gap:+.4f}  "
                      f"dark_gap={d_gap:+.4f}  "
                      f"dark_via_Wq={dw_gap:+.4f}")

    # ---- The entity heads vs template heads ----
    print(f"\n{'='*70}")
    print(f"ENTITY HEADS vs TEMPLATE HEADS (L21 Q-space)")
    print(f"{'='*70}")

    # Use the L21 Q vectors, split by head
    entity_heads = [(21, 1), (23, 2), (23, 3), (24, 1), (24, 4)]
    template_heads = [(27, 2), (23, 5), (20, 4), (26, 6), (17, 7)]

    # Get Q vectors per head at L21
    sa21 = layers_list[target_layer].self_attn
    n_heads = sa21.n_heads
    head_dim = sa21.head_dim

    print(f"\n  Separation by head type at L{target_layer}:")
    print(f"  Head decomposition of W_q @ residual into {n_heads} heads x {head_dim}d")

    # For each prompt, get the full Q vector and split by head
    for head_set_name, head_set in [("entity_heads", entity_heads),
                                      ("template_heads", template_heads)]:
        # Use heads at target_layer only
        relevant = [(l, h) for l, h in head_set if l == target_layer]
        if not relevant:
            continue

        head_indices = [h for _, h in relevant]

        head_vecs = {}
        for tname in template_names:
            vecs = []
            for cap in all_data[tname]:
                res = cap["residuals"].get(target_layer - 1)  # pre-attention residual
                if res is None:
                    continue
                # Full Q projection then split by head
                q_full = wq @ res  # [2048]
                q_heads = q_full.reshape(n_heads, head_dim)  # [8, 256]
                # Concatenate selected heads
                selected = np.concatenate([q_heads[h] for h in head_indices])
                vecs.append(selected)
            head_vecs[tname] = vecs

        w, b, gap = compute_separation(head_vecs)
        print(f"\n  {head_set_name} (heads {head_indices}):")
        print(f"    Within-template: {w:.4f}")
        print(f"    Between-template: {b:.4f}")
        print(f"    Gap: {gap:+.4f}")

    # ---- Summary table ----
    print(f"\n{'='*70}")
    print(f"SUMMARY: WHERE THE ROUTING SIGNAL LIVES")
    print(f"{'='*70}")

    # Find key transition points
    wq_gaps = [(d["layer"], d["wq_gap"]) for d in layer_data]
    max_gap_layer = max(wq_gaps, key=lambda x: x[1])
    first_significant = next((l for l, g in wq_gaps if g > 0.05), None)

    print(f"\n  Routing signal (W_q gap) emergence:")
    print(f"    First significant (gap > 0.05): {'L'+str(first_significant) if first_significant is not None else 'none'}")
    print(f"    Maximum gap: L{max_gap_layer[0]} (gap={max_gap_layer[1]:.4f})")

    # Compare lenses
    l13_raw = next((d["raw_gap"] for d in layer_data if d["layer"] == 13), 0)
    l13_wq = next((d["wq_gap"] for d in layer_data if d["layer"] == 13), 0)
    l13_ue = next((d["ue_gap"] for d in layer_data if d["layer"] == 13), 0)

    print(f"\n  At L13 (syntax/knowledge boundary):")
    print(f"    Raw residual gap:    {l13_raw:+.4f}  (invisible)")
    print(f"    W_q lens gap:        {l13_wq:+.4f}  (visible!)")
    print(f"    Unembedding lens:    {l13_ue:+.4f}")
    print(f"    Amplification ratio: {l13_wq / l13_raw:.0f}x" if l13_raw > 0.0001 else "")

    print(f"\n  Interpretation:")
    print(f"    The routing signal is invisible in raw residual space")
    print(f"    but becomes visible through the W_q projection.")
    print(f"    W_q acts as a lens that reads distributed dark-space structure.")
    print(f"    The signal builds gradually through syntax layers L0-L13,")
    print(f"    then gets amplified by W_q into the Q-space where")
    print(f"    template-specific attention patterns are selected.")

    # ---- Save ----
    save_data = {
        "layer_data": layer_data,
        "max_gap_layer": max_gap_layer[0],
        "max_gap_value": max_gap_layer[1],
    }
    with open(output_dir / "darkspace_visible_results.json", 'w') as f:
        json.dump(save_data, f, indent=2)

    # ---- Try matplotlib plots ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Plot 1: Gap emergence across layers
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        layers_plot = [d["layer"] for d in layer_data if d["layer"] >= 0]
        raw_gaps = [d["raw_gap"] for d in layer_data if d["layer"] >= 0]
        wq_gaps_plot = [d["wq_gap"] for d in layer_data if d["layer"] >= 0]
        ue_gaps_plot = [d["ue_gap"] for d in layer_data if d["layer"] >= 0]

        axes[0].plot(layers_plot, raw_gaps, 'k-o', markersize=3, label='Raw residual')
        axes[0].set_title('Raw Residual Space')
        axes[0].set_xlabel('Layer')
        axes[0].set_ylabel('Within - Between cosine gap')
        axes[0].axvline(x=13, color='gray', linestyle='--', alpha=0.5, label='L13 boundary')
        axes[0].legend()

        axes[1].plot(layers_plot, wq_gaps_plot, 'b-o', markersize=3, label='W_q projection')
        axes[1].set_title('W_q Lens (Routing Space)')
        axes[1].set_xlabel('Layer')
        axes[1].axvline(x=13, color='gray', linestyle='--', alpha=0.5)
        axes[1].legend()

        axes[2].plot(layers_plot, ue_gaps_plot, 'r-o', markersize=3, label='Unembedding')
        axes[2].set_title('Unembedding Lens (Content Space)')
        axes[2].set_xlabel('Layer')
        axes[2].axvline(x=13, color='gray', linestyle='--', alpha=0.5)
        axes[2].legend()

        plt.tight_layout()
        plt.savefig(str(output_dir / "darkspace_emergence.png"), dpi=150)
        plt.close()
        print(f"\n  Plot saved: {output_dir / 'darkspace_emergence.png'}")

        # Plot 2: Q-space PCA at 4 key layers
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        colors = {"capital": "blue", "synonym": "red", "arithmetic": "green", "code": "purple"}

        for ax_idx, layer in enumerate([0, 6, 13, target_layer]):
            if layer >= n_layers:
                continue
            ax = axes[ax_idx]

            all_vecs = []
            all_labels = []
            for tname in template_names:
                for cap in all_data[tname]:
                    res = cap["residuals"].get(layer)
                    if res is not None:
                        all_vecs.append(wq @ res)
                        all_labels.append(tname)

            if len(all_vecs) < 4:
                continue

            X = normalize(np.stack(all_vecs))
            pca = PCA(n_components=2)
            coords = pca.fit_transform(X)

            for tname in template_names:
                mask = np.array([l == tname for l in all_labels])
                ax.scatter(coords[mask, 0], coords[mask, 1],
                          c=colors[tname], label=tname, s=40, alpha=0.8)

            layer_label = f"L{layer}"
            ax.set_title(f'{layer_label}\n(var: {pca.explained_variance_ratio_[0]:.0%}+{pca.explained_variance_ratio_[1]:.0%})')
            ax.legend(fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])

        plt.suptitle('Routing Signal in Q-Space: From Noise to Separation', fontsize=14)
        plt.tight_layout()
        plt.savefig(str(output_dir / "darkspace_pca.png"), dpi=150)
        plt.close()
        print(f"  Plot saved: {output_dir / 'darkspace_pca.png'}")

    except ImportError:
        print("\n  (matplotlib not available, skipping plots)")

    print()


if __name__ == "__main__":
    main()
