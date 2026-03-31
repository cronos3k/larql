#!/usr/bin/env python3
"""Probe entities with MLX — full forward pass with real attention.

Two matching strategies:
  1. Gate matching: project residuals through gate vectors, check if the
     down_meta output token matches a Wikidata triple object.
  2. Prediction matching: project the final residual through the LM head,
     check if the model's top predicted tokens match a Wikidata triple object.

Strategy 2 is the primary signal — it captures what the model actually
predicts, not just what individual features encode.

Model-agnostic: works with any MLX-compatible model (Gemma, Llama, Mistral,
Qwen, etc.). Auto-detects model structure.

Usage:
    pip install mlx mlx-lm
    python3 scripts/probe_mlx.py --model google/gemma-3-4b-it --vindex output/gemma3-4b.vindex
    python3 scripts/probe_mlx.py --model mlx-community/Meta-Llama-3-8B-4bit --vindex output/llama3-8b.vindex
"""

import argparse
import json
import sys
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load as mlx_load
except ImportError as e:
    print(f"Install MLX: pip install mlx mlx-lm ({e})", file=sys.stderr)
    sys.exit(1)


# Resolve paths relative to the monorepo root (larql/)
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent  # knowledge/scripts/ → knowledge/ → larql/

_DEFAULT_TRIPLES = str(_REPO_ROOT / "knowledge" / "data" / "wikidata_triples.json")
_DEFAULT_PROBES = str(_REPO_ROOT / "knowledge" / "probes")

# Stop words to skip when building partial match keys
_STOP_WORDS = frozenset({
    "the", "of", "and", "in", "to", "for", "is", "on", "at", "by",
    "an", "or", "de", "la", "le", "el", "du", "des", "von", "van",
    "al", "bin", "ibn", "di", "da", "do", "das", "den", "der", "het",
})


def load_templates(templates_path: str | None = None) -> dict[str, list[str]]:
    """Load templates from probe_templates.json.

    Returns {relation: [template_str, ...]} with all variants per relation.
    """
    if templates_path is None:
        templates_path = str(
            _REPO_ROOT / "knowledge" / "data" / "probe_templates.json"
        )
    path = Path(templates_path)
    if not path.exists():
        print(f"ERROR: Templates not found at {path}")
        sys.exit(1)
    with open(path) as f:
        raw = json.load(f)
    return {rel: (variants if isinstance(variants, list) else [variants])
            for rel, variants in raw.items()}


def normalize_object(obj: str) -> list[str]:
    """Expand a triple object into all matchable forms.

    "Emmanuel Macron" → ["emmanuel macron", "macron", "emmanuel"]
    "Schindler's List" → ["schindler's list", "schindler"]
    "Paris" → ["paris"]
    "New York" → ["new york"]
    "Leonardo da Vinci" → ["leonardo da vinci", "leonardo", "vinci"]
    """
    obj_lower = obj.lower().strip()
    forms = {obj_lower}

    words = obj_lower.split()
    if len(words) > 1:
        for word in words:
            # Strip possessive suffix carefully
            if word.endswith("'s"):
                clean = word[:-2]
            elif word.endswith("\u2019s"):
                clean = word[:-2]
            else:
                clean = word
            # Must be 4+ chars to avoid false positives (new, old, big, etc.)
            # Must not be a stop word
            if clean not in _STOP_WORDS and len(clean) >= 4:
                forms.add(clean)

    return list(forms)


def build_match_index(triples: dict) -> dict[tuple[str, str], str]:
    """Build a lookup: (subject_lower, object_form) → relation.

    For each triple (subject, object), generates all normalized forms
    of the object. This handles multi-word entities:
      ("france", "emmanuel macron") → head_of_state
      ("france", "macron") → head_of_state
      ("france", "emmanuel") → head_of_state
    """
    index = {}
    for rel_name, rel_data in triples.items():
        for pair in rel_data.get("pairs", []):
            if len(pair) < 2:
                continue
            subj = pair[0].lower().strip()
            for form in normalize_object(pair[1]):
                key = (subj, form)
                # Don't overwrite — first relation wins (higher in file = higher priority)
                if key not in index:
                    index[key] = rel_name
    return index


def load_vindex_gates_and_meta(vindex_dir):
    """Load gate vectors and down_meta from vindex."""
    vindex_dir = Path(vindex_dir)

    with open(vindex_dir / "index.json") as f:
        config = json.load(f)

    hidden_size = config["hidden_size"]
    gate_raw = np.fromfile(vindex_dir / "gate_vectors.bin", dtype=np.float32)
    gates = {}
    for layer_info in config["layers"]:
        layer = layer_info["layer"]
        nf = layer_info["num_features"]
        offset = layer_info["offset"] // 4
        gates[layer] = gate_raw[offset:offset + nf * hidden_size].reshape(nf, hidden_size)

    down_meta = {}
    with open(vindex_dir / "down_meta.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            down_meta[(obj.get("l", 0), obj.get("f", 0))] = obj.get("t", "")

    return config, gates, down_meta


def _find_model_parts(model):
    """Auto-detect model internals: embed, layers, norm, lm_head.

    Supports multiple architectures:
      - Gemma3: model['language_model']['model'].{embed_tokens, layers, norm}
      - Llama/Mistral/Qwen: model.model.{embed_tokens, layers, norm}
      - Generic: tries common attribute paths

    Returns (embed_fn, layers, norm, lm_head_fn, needs_embed_scale) where
    lm_head_fn maps hidden -> logits.
    """
    import math

    # Try Gemma3 VLM structure: model['language_model']['model']
    try:
        lm = model['language_model']
        inner = lm['model']
        embed_fn = inner.embed_tokens
        layers = inner.layers
        norm = inner.norm
        # Gemma uses weight-tied LM head + sqrt(hidden) scaling
        def lm_head(h):
            return h @ embed_fn.weight.T
        return embed_fn, layers, norm, lm_head, True
    except (KeyError, TypeError, AttributeError):
        pass

    # Try Llama/Mistral/Qwen structure: model.model.{embed_tokens, layers, norm}
    inner = getattr(model, 'model', None)
    if inner and hasattr(inner, 'embed_tokens') and hasattr(inner, 'layers'):
        embed_fn = inner.embed_tokens
        layers = inner.layers
        norm = inner.norm

        # Check for separate lm_head (not weight-tied)
        if hasattr(model, 'lm_head'):
            lm_head_layer = model.lm_head
            def lm_head(h):
                return lm_head_layer(h)
        else:
            # Weight-tied fallback
            def lm_head(h):
                return h @ embed_fn.weight.T

        # Gemma text-only models also use this path with embed scaling
        model_type = getattr(getattr(model, 'config', None), 'model_type', '')
        needs_scale = 'gemma' in str(model_type).lower()
        return embed_fn, layers, norm, lm_head, needs_scale

    raise RuntimeError(
        "Could not detect model structure. Expected Gemma3 VLM, "
        "Llama, Mistral, or Qwen architecture."
    )


def get_residuals_and_logits(model, tokenizer, prompt, _model_parts={}):
    """Run forward pass, capture per-layer residuals AND final logits.

    Auto-detects model architecture on first call, caches the result.

    Returns (residuals, top_predictions) where:
      residuals: {layer_idx: np.array of shape (hidden_size,)}
      top_predictions: list of (token_str, probability) for top-20 predictions
    """
    # Cache model parts detection
    if 'parts' not in _model_parts:
        _model_parts['parts'] = _find_model_parts(model)

    embed_fn, layers, norm, lm_head, needs_scale = _model_parts['parts']

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    try:
        h = embed_fn(input_ids)

        if needs_scale:
            import math
            h = h * math.sqrt(h.shape[-1])

        seq_len = h.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        residuals = {}
        for i, layer in enumerate(layers):
            h = layer(h, mask=mask)
            mx.eval(h)
            residuals[i] = np.array(h[0, -1, :].astype(mx.float32))

        # Project through final norm + LM head to get predictions
        h_last = h[:, -1:, :]
        h_normed = norm(h_last)
        logits = lm_head(h_normed)
        mx.eval(logits)

        logits_np = np.array(logits[0, 0, :].astype(mx.float32))

        # Softmax for top-20
        top_indices = np.argsort(-logits_np)[:20]
        # Stable softmax on top tokens only
        top_logits = logits_np[top_indices]
        top_logits = top_logits - top_logits.max()
        probs = np.exp(top_logits)
        probs = probs / probs.sum()

        top_predictions = []
        for idx, prob in zip(top_indices, probs):
            token_str = tokenizer.decode([int(idx)]).strip()
            if len(token_str) >= 2:
                top_predictions.append((token_str.lower(), float(prob)))

        return residuals, top_predictions

    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def _model_slug(model_id: str) -> str:
    """Convert model ID to a directory-safe slug.

    "google/gemma-3-4b-it" -> "gemma-3-4b-it"
    "mlx-community/Meta-Llama-3-8B" -> "Meta-Llama-3-8B"
    """
    return model_id.split("/")[-1]


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Probe model features with MLX inference",
    )
    parser.add_argument(
        "--model", type=str, default="google/gemma-3-4b-it",
        help="HuggingFace model ID (default: google/gemma-3-4b-it)",
    )
    parser.add_argument(
        "--vindex", type=str, default=None,
        help="Path to vindex directory (default: output/<model-slug>.vindex)",
    )
    parser.add_argument(
        "--triples", type=str, default=_DEFAULT_TRIPLES,
        help="Path to combined triples JSON",
    )
    parser.add_argument(
        "--templates", type=str, default=None,
        help="Path to probe templates JSON",
    )
    parser.add_argument(
        "--output", type=str, default=_DEFAULT_PROBES,
        help="Probes output directory (default: knowledge/probes/)",
    )
    parser.add_argument(
        "--top-k", type=int, default=50,
        help="Top-K gate features to check per layer (default: 50)",
    )
    parser.add_argument(
        "--min-gate-score", type=float, default=5.0,
        help="Minimum |gate score| to consider (default: 5.0)",
    )
    parser.add_argument(
        "--offline", action="store_true", default=True,
        help="Use cached model, don't hit HuggingFace (default: true)",
    )
    return parser.parse_args()


def main() -> None:
    """Run full MLX probe: forward pass, gate activation, Wikidata matching."""
    args = parse_args()
    model_id = args.model
    model_slug = _model_slug(model_id)

    # Resolve vindex path
    vindex_path = args.vindex
    if vindex_path is None:
        vindex_path = str(_REPO_ROOT / "output" / f"{model_slug}.vindex")

    print("Loading vindex gates and metadata...")
    config, gates, down_meta = load_vindex_gates_and_meta(vindex_path)
    hidden_size = config["hidden_size"]
    num_layers = config["num_layers"]
    # Knowledge layers: middle ~40% of the model (where factual knowledge lives)
    # Gemma 34L: L14-27, Llama 32L: L13-25, Mistral 32L: L13-25
    knowledge_start = num_layers * 2 // 5
    knowledge_end = num_layers * 4 // 5
    knowledge_layers = range(knowledge_start, knowledge_end)
    print(
        f"  {num_layers} layers, {hidden_size} hidden,"
        f" {len(down_meta)} features,"
        f" knowledge L{knowledge_start}-L{knowledge_end - 1}"
    )

    print("Loading triples...")
    with open(args.triples) as f:
        triples = json.load(f)

    print("Building match index (with normalized short forms)...")
    match_index = build_match_index(triples)
    print(f"  {len(match_index)} matchable (subject, form) pairs")

    print(f"Loading MLX model: {model_id}...")
    import os
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model, tokenizer = mlx_load(model_id)
    print("  Model loaded")

    # Quick test
    print("\nTest: 'The capital of France is'")
    residuals, predictions = get_residuals_and_logits(
        model, tokenizer, "The capital of France is"
    )
    if residuals is None:
        print("ERROR: Could not capture residuals from MLX model")
        sys.exit(1)

    print(f"  Captured residuals at {len(residuals)} layers")
    print(f"  Top predictions: {predictions[:5]}")

    # Check gate activations at peak knowledge layer
    peak_layer = knowledge_end - 1
    if peak_layer in residuals and peak_layer in gates:
        r = residuals[peak_layer]
        scores = gates[peak_layer] @ r
        top5 = np.argsort(-np.abs(scores))[:5]
        print(f"  L{peak_layer} top features:")
        for idx in top5:
            target = down_meta.get((peak_layer, int(idx)), "?")
            print(f"    F{idx} gate={scores[idx]:+.1f} -> {target}")

    # Full probe
    TEMPLATES = load_templates(args.templates)
    print(f"\nProbing with {len(TEMPLATES)} relations...")
    feature_labels = {}
    prediction_labels = {}  # features confirmed via model prediction
    relation_counts = defaultdict(int)
    total_probes = 0
    start_time = time.time()

    for rel_name, templates in TEMPLATES.items():
        if rel_name not in triples:
            continue

        # Use ALL subjects, prioritizing short/well-known names
        all_subjects = list(set(
            pair[0] for pair in triples[rel_name].get("pairs", [])
            if len(pair) >= 2 and 2 <= len(pair[0]) <= 30
        ))
        all_subjects.sort(key=lambda s: (len(s.split()), len(s)))
        subjects = all_subjects

        if not subjects:
            continue

        gate_matched = 0
        pred_matched = 0
        rel_start = time.time()
        probe_items = [(s, t) for s in subjects for t in templates]

        for pi, (subject, template) in enumerate(probe_items):
            prompt = template.replace("{X}", subject)
            residuals, predictions = get_residuals_and_logits(
                model, tokenizer, prompt
            )
            if residuals is None:
                continue

            total_probes += 1
            subj_key = subject.lower().strip()

            if (pi + 1) % 50 == 0:
                elapsed_rel = time.time() - rel_start
                rate = (pi + 1) / elapsed_rel if elapsed_rel > 0 else 0
                eta = (len(probe_items) - pi - 1) / rate if rate > 0 else 0
                total_matched = gate_matched + pred_matched
                sys.stdout.write(
                    f"\r  {rel_name:<20s} {pi+1}/{len(probe_items)}"
                    f" ({total_matched} labels, {rate:.0f}/s,"
                    f" ETA {eta:.0f}s)"
                )
                sys.stdout.flush()

            # --- Strategy 1: Gate matching ---
            # Project residuals through gate vectors, match down_meta tokens
            for layer in knowledge_layers:
                if layer not in residuals or layer not in gates:
                    continue
                r = residuals[layer]
                scores = gates[layer] @ r
                top_indices = np.argsort(-np.abs(scores))[:args.top_k]

                for feat_idx in top_indices:
                    score = float(scores[feat_idx])
                    if abs(score) < args.min_gate_score:
                        continue
                    target = down_meta.get((layer, int(feat_idx)), "")
                    if len(target) < 2:
                        continue

                    tgt_lower = target.lower().strip()
                    key = (subj_key, tgt_lower)
                    if match_index.get(key) == rel_name:
                        feat_key = f"L{layer}_F{feat_idx}"
                        if feat_key not in feature_labels:
                            feature_labels[feat_key] = rel_name
                            relation_counts[rel_name] += 1
                            gate_matched += 1

            # --- Strategy 2: Prediction matching ---
            # Check if the model's top predicted tokens match any triple object
            if predictions:
                for pred_token, pred_prob in predictions[:10]:
                    if pred_prob < 0.01:
                        break
                    key = (subj_key, pred_token)
                    if match_index.get(key) == rel_name:
                        # Find which features are most active for this
                        # prediction — attribute to the top gate feature
                        # at the peak knowledge layer (L26-27)
                        best_feat = None
                        best_score = 0
                        for layer in [27, 26, 25, 24]:
                            if layer not in residuals or layer not in gates:
                                continue
                            r = residuals[layer]
                            scores = gates[layer] @ r
                            top_idx = np.argmax(np.abs(scores))
                            if abs(scores[top_idx]) > best_score:
                                best_score = abs(scores[top_idx])
                                best_feat = f"L{layer}_F{top_idx}"
                        if best_feat and best_feat not in feature_labels:
                            if best_feat not in prediction_labels:
                                prediction_labels[best_feat] = rel_name
                                pred_matched += 1
                        break  # Only count first matching prediction

        elapsed = time.time() - start_time
        rate = total_probes / elapsed if elapsed > 0 else 0
        nvariants = len(templates)
        total_matched = gate_matched + pred_matched
        print(
            f"  {rel_name:<20s} {len(subjects):3d} entities"
            f" x {nvariants} templates -> {total_matched:3d} features"
            f"  (gate={gate_matched}, pred={pred_matched})"
            f"  ({rate:.1f} probes/s)"
        )

    elapsed = time.time() - start_time
    print(
        f"\nTotal: {total_probes} probes in {elapsed:.0f}s"
        f" ({total_probes/elapsed:.1f}/s)"
    )
    print(
        f"Gate-matched: {len(feature_labels)} features, "
        f"Prediction-matched: {len(prediction_labels)} features"
    )

    # Merge: gate labels take priority, prediction labels fill gaps
    all_labels = dict(feature_labels)
    for key, rel in prediction_labels.items():
        if key not in all_labels:
            all_labels[key] = rel

    print(f"Combined: {len(all_labels)} unique features")

    if relation_counts:
        print("\nRelation distribution (gate-matched):")
        for rel, count in sorted(
            relation_counts.items(), key=lambda x: -x[1]
        ):
            print(f"  {rel:<25s} {count:4d}")

    # Count prediction-only relations
    pred_only_counts = defaultdict(int)
    for key, rel in prediction_labels.items():
        if key not in feature_labels:
            pred_only_counts[rel] += 1
    if pred_only_counts:
        print("\nPrediction-only features:")
        for rel, count in sorted(
            pred_only_counts.items(), key=lambda x: -x[1]
        ):
            print(f"  {rel:<25s} {count:4d}")

    # Merge with existing labels on disk
    vindex_labels = Path(vindex_path) / "feature_labels.json"
    existing = {}
    if vindex_labels.exists():
        with open(vindex_labels) as f:
            existing = json.load(f)

    new_count = 0
    for key, rel in all_labels.items():
        if key not in existing:
            existing[key] = rel
            new_count += 1

    with open(vindex_labels, "w") as f:
        json.dump(existing, f, indent=2)

    probe_path = Path(args.output) / model_slug / "feature_labels.json"
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    with open(probe_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(
        f"\nMerged: {new_count} new + {len(existing) - new_count}"
        f" existing = {len(existing)} total"
    )
    print(f"Saved to {vindex_labels}")
    print(f"Saved to {probe_path}")


if __name__ == "__main__":
    main()
