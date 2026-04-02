"""
Demo: MLX Residual Capture → Vindex Analysis

Captures residual stream activations from MLX inference and feeds them
to the vindex for feature analysis. Shows what knowledge features fire
at each layer during a forward pass.

Usage:
    python examples/demos/mlx_residual_capture.py

Requires: mlx, mlx-lm, larql (built with maturin develop --release)
"""

import sys
import os
import numpy as np

# ── Check dependencies ──
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx_lm
except ImportError:
    print("This demo requires mlx and mlx-lm:")
    print("  pip install mlx mlx-lm")
    sys.exit(1)

import larql

VINDEX_PATH = os.environ.get(
    "VINDEX_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "output", "gemma3-4b-v2.vindex")
)
MODEL_ID = "google/gemma-3-4b-it"


def find_model_internals(model):
    """Navigate MLX-LM model wrappers to find embed_tokens and layers.

    Handles different model architectures:
    - Gemma 3: model.language_model.model.{embed_tokens, layers}
    - Llama/Mistral: model.model.{embed_tokens, layers}
    - Simple: model.{embed_tokens, layers}
    """
    # Try common paths
    candidates = [
        model,
        getattr(model, 'model', None),
        getattr(model, 'language_model', None),
    ]
    # Also try model.language_model.model (Gemma 3)
    lm = getattr(model, 'language_model', None)
    if lm is not None:
        candidates.append(getattr(lm, 'model', None))

    for m in candidates:
        if m is None:
            continue
        if hasattr(m, 'embed_tokens') and hasattr(m, 'layers'):
            return m.embed_tokens, m.layers

    raise AttributeError(
        f"Cannot find embed_tokens/layers in model of type {type(model).__name__}. "
        f"Top-level attrs: {[a for a in dir(model) if not a.startswith('_')]}"
    )


def capture_residuals(model, tokenizer, prompt: str) -> dict:
    """Run MLX forward pass and capture residual at each layer.

    Returns dict: layer_index → numpy array of shape (hidden_size,)
    for the last token position.
    """
    embed_fn, layers = find_model_internals(model)

    tokens = tokenizer.encode(prompt)
    x = mx.array([tokens])

    # Get embeddings
    h = embed_fn(x)

    # Gemma models scale embeddings
    if hasattr(embed_fn, 'weight'):
        dim = h.shape[-1]
        # Check if this model uses embed scaling (Gemma does: scale = sqrt(hidden_size))
        if hasattr(model, 'args') and hasattr(model.args, 'hidden_size'):
            h = h * (model.args.hidden_size ** 0.5)

    residuals = {}

    for i, layer in enumerate(layers):
        h = layer(h)
        # Capture last token residual as numpy (convert through float32 for bfloat16 models)
        last_tok = h[0, -1, :]
        if last_tok.dtype != mx.float32:
            last_tok = last_tok.astype(mx.float32)
        mx.eval(last_tok)
        residuals[i] = np.array(last_tok, dtype=np.float32)

    return residuals


def analyse_residuals(vindex, residuals: dict, prompt: str):
    """Feed captured residuals to vindex and show what features fire."""

    bands = vindex.layer_bands()
    knowledge_start = bands["knowledge"][0]
    knowledge_end = bands["knowledge"][1]

    print(f"\n{'='*70}")
    print(f"  Prompt: \"{prompt}\"")
    print(f"  Knowledge layers: L{knowledge_start}-L{knowledge_end}")
    print(f"{'='*70}")

    # Analyse knowledge layers
    for layer in range(knowledge_start, knowledge_end + 1):
        if layer not in residuals:
            continue

        residual = residuals[layer]
        hits = vindex.gate_knn(layer=layer, query_vector=residual.tolist(), top_k=5)

        if not hits:
            continue

        # Only show layers with strong activations
        top_score = abs(hits[0][1])
        if top_score < 10.0:
            continue

        print(f"\n  Layer {layer}:")
        for feat, score in hits[:3]:
            meta = vindex.feature_meta(layer, feat)
            if meta is None:
                continue
            label = vindex.feature_label(layer, feat) or ""
            label_str = f" [{label}]" if label else ""
            print(f"    F{feat:>5}  score={score:>8.1f}  → '{meta.top_token}'{label_str}")


def compare_embed_vs_residual(vindex, residuals: dict, entity: str, prompt: str):
    """Compare vindex entity embedding with captured MLX residuals.

    The key experiment: how close is the embedding-based lookup (what vindex
    uses for DESCRIBE) to the actual residual from a forward pass?
    """
    embed = vindex.embed(entity)

    bands = vindex.layer_bands()
    knowledge_start = bands["knowledge"][0]
    knowledge_end = bands["knowledge"][1]

    print(f"\n{'='*70}")
    print(f"  Embedding vs Residual comparison for '{entity}'")
    print(f"  Prompt: \"{prompt}\"")
    print(f"{'='*70}")

    for layer in range(knowledge_start, knowledge_end + 1):
        if layer not in residuals:
            continue

        residual = residuals[layer]

        # Cosine similarity
        cos = float(np.dot(embed, residual) / (np.linalg.norm(embed) * np.linalg.norm(residual) + 1e-8))

        # Compare KNN results
        hits_embed = vindex.gate_knn(layer=layer, query_vector=embed.tolist(), top_k=3)
        hits_resid = vindex.gate_knn(layer=layer, query_vector=residual.tolist(), top_k=3)

        embed_tokens = set()
        resid_tokens = set()
        for f, _ in hits_embed:
            m = vindex.feature_meta(layer, f)
            if m: embed_tokens.add(m.top_token.strip().lower())
        for f, _ in hits_resid:
            m = vindex.feature_meta(layer, f)
            if m: resid_tokens.add(m.top_token.strip().lower())

        overlap = embed_tokens & resid_tokens
        jaccard = len(overlap) / max(len(embed_tokens | resid_tokens), 1)

        if abs(hits_resid[0][1]) > 10.0 if hits_resid else False:
            embed_top = vindex.feature_meta(layer, hits_embed[0][0])
            resid_top = vindex.feature_meta(layer, hits_resid[0][0])
            et = embed_top.top_token if embed_top else "?"
            rt = resid_top.top_token if resid_top else "?"
            print(f"  L{layer:2d}  cos={cos:+.4f}  overlap={jaccard:.0%}  "
                  f"embed→'{et}'  resid→'{rt}'")


def demo_knowledge_query(vindex):
    """Show pure vindex knowledge queries — no MLX needed."""
    print(f"\n{'='*70}")
    print(f"  Knowledge queries (vindex only, no inference)")
    print(f"{'='*70}")

    entities = ["France", "Einstein", "Python", "Mozart"]
    for entity in entities:
        edges = vindex.describe(entity)
        print(f"\n  {entity}:")
        for edge in edges[:5]:
            rel = edge.relation or "?"
            also = f"  also: {', '.join(edge.also)}" if edge.also else ""
            print(f"    {rel:>20} → {edge.target:<15} score={edge.gate_score:>7.1f} [{edge.source}]{also}")


def main():
    print("Loading vindex...")
    vindex = larql.load(VINDEX_PATH)
    print(f"  {vindex}")

    # Part 1: Pure vindex queries (no MLX)
    demo_knowledge_query(vindex)

    # Part 2: MLX residual capture
    print(f"\nLoading MLX model ({MODEL_ID})...")
    try:
        model, tokenizer = mlx_lm.load(MODEL_ID)
        print(f"  Loaded {MODEL_ID}")

        prompts = [
            ("The capital of France is", "France"),
            ("Albert Einstein was a famous", "Einstein"),
            ("The programming language Python was created by", "Python"),
        ]

        for prompt, entity in prompts:
            residuals = capture_residuals(model, tokenizer, prompt)
            analyse_residuals(vindex, residuals, prompt)
            compare_embed_vs_residual(vindex, residuals, entity, prompt)

    except Exception as e:
        print(f"\n  MLX model loading failed: {e}")
        print("  This is expected if the model isn't downloaded yet.")
        print(f"  Run: mlx_lm.load('{MODEL_ID}') to download first.")
        print("\n  Vindex knowledge queries (above) work without MLX.")


if __name__ == "__main__":
    main()
