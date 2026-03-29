#!/usr/bin/env python3
"""Trace which FFN features actually fire during inference.

Compares forward-pass activations against static extraction.

Usage:
    python scripts/attention_trace.py \
        --model google/gemma-3-4b-it \
        --entities "Germany,Spain,Paris,Mozart,Amsterdam" \
        --template "Toulouse is French. Rome is Roman. {entity} is"
"""

import argparse
import json
import sys

import mlx.core as mx
import mlx_lm
import numpy as np


def trace_activations(model, tokenizer, prompt, layers):
    """Run forward pass capturing FFN gate*up activations at specified layers."""
    tokens = tokenizer.encode(prompt)
    x = mx.array([tokens])

    # Embed + scale
    hidden_size = model.language_model.model.layers[0].input_layernorm.weight.shape[0]
    h = model.language_model.model.embed_tokens(x) * (hidden_size ** 0.5)

    results = {}

    for i, layer in enumerate(model.language_model.model.layers):
        # Attention
        h_norm = layer.input_layernorm(h)
        attn_out = layer.self_attn(h_norm)
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        h = h + layer.post_attention_layernorm(attn_out)

        # FFN
        h_ffn = layer.pre_feedforward_layernorm(h)
        gate_out = layer.mlp.gate_proj(h_ffn)
        up_out = layer.mlp.up_proj(h_ffn)
        activation = (gate_out * mx.sigmoid(gate_out)) * up_out

        if i in layers:
            act = activation[0, -1, :].astype(mx.float32)
            mx.eval(act)
            act_np = np.array(act)
            top_idx = np.argpartition(-np.abs(act_np), 50)[:50]
            top_features = [(int(idx), float(act_np[idx])) for idx in top_idx]
            top_features.sort(key=lambda x: -abs(x[1]))
            results[i] = top_features

        ffn_out = layer.mlp.down_proj(activation)
        h = h + layer.post_feedforward_layernorm(ffn_out)

    # Also get the model's prediction
    h_final = model.language_model.model.norm(h)
    logits = (h_final @ model.language_model.model.embed_tokens.weight.T).astype(mx.float32)
    last_logits = logits[0, -1, :]
    mx.eval(last_logits)
    top_pred_idx = mx.argmax(last_logits).item()
    prediction = tokenizer.decode([top_pred_idx]).strip()

    return results, prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--entities", required=True)
    parser.add_argument("--template", default="{entity}")
    parser.add_argument("--edges-dir", default="output/edges")
    parser.add_argument("--layers", default="25,26")
    args = parser.parse_args()

    entities = [e.strip() for e in args.entities.split(",")]
    trace_layers = set(int(l) for l in args.layers.split(","))

    print(f"Loading model: {args.model}", file=sys.stderr)
    model, tokenizer = mlx_lm.load(args.model)

    # Load edge data
    edge_features = {}
    for layer in trace_layers:
        edge_features[layer] = {}
        try:
            with open(f"{args.edges_dir}/L{layer}_edges.jsonl") as f:
                for line in f:
                    e = json.loads(line)
                    edge_features[layer][e['feature']] = e
        except FileNotFoundError:
            pass

    for entity in entities:
        prompt = args.template.replace("{entity}", entity)
        print(f"\n{'='*80}")
        print(f"Entity: {entity}")
        print(f"Prompt: {prompt}")

        activations, prediction = trace_activations(model, tokenizer, prompt, trace_layers)
        print(f"Model predicts: {prediction}")
        print(f"{'='*80}")

        for layer in sorted(trace_layers):
            if layer not in activations:
                continue
            top = activations[layer]

            print(f"\n  L{layer} — Top 15 features that ACTUALLY fired:")
            for feat_idx, mag in top[:15]:
                edge = edge_features.get(layer, {}).get(feat_idx, {})
                src = edge.get('source', '?')
                tgt = edge.get('target', '?')
                dd = edge.get('down_dist', 0)
                print(f"    F{feat_idx:5d}  mag={mag:8.2f}  gate_knn={src:15s} → down_knn={tgt:15s}  dd={dd:.3f}")

            # Compare
            static = {eid for eid, e in edge_features.get(layer, {}).items()
                      if e.get('source', '').lower() == entity.lower()}
            active = {f for f, _ in top[:50]}
            overlap = active & static

            print(f"\n    Static extraction found {len(static)} features for '{entity}'")
            print(f"    Forward pass activated {len(active)} features (top-50)")
            print(f"    Overlap: {len(overlap)}")
            if static and not overlap:
                print(f"    Static features: {sorted(static)[:5]}")
                print(f"    → None of the statically-extracted features actually fired!")


if __name__ == "__main__":
    main()
