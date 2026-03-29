#!/usr/bin/env python3
"""Validate the knowledge graph against model inference.

For each entity, runs a forward pass with a prompt template,
then queries the graph for edges from that entity. Reports match rate.

Usage:
    python scripts/validate_graph.py \
        --model google/gemma-3-4b-it \
        --graph output/merged.larql.json \
        --entities "Toulouse,Rome,Dutch,France,Germany,Japan,Mozart,Einstein" \
        --template "The language associated with {entity} is"

Requires: mlx-lm (pip install mlx-lm)
"""

import argparse
import json
import sys
import time


def load_graph_edges(graph_path, entity):
    """Load all outgoing edges for an entity from the graph."""
    edges = []
    # Stream the graph file — it might be large
    with open(graph_path) as f:
        data = json.load(f)

    for e in data.get("edges", []):
        subj = e.get("s", "")
        if subj.lower() == entity.lower() or subj == entity:
            edges.append({
                "relation": e.get("r", ""),
                "object": e.get("o", ""),
                "confidence": e.get("c", 0),
            })

    # Sort by confidence descending
    edges.sort(key=lambda x: -x["confidence"])
    return edges


def model_predict(model, tokenizer, prompt, top_k=5):
    """Run a forward pass and get top-k next token predictions."""
    import mlx.core as mx
    import mlx_lm

    result = mlx_lm.generate(
        model, tokenizer, prompt=prompt, max_tokens=1, verbose=False,
    )
    # Get just the generated token
    return result.strip()


def model_predict_logprobs(model, tokenizer, prompt, top_k=10):
    """Run forward pass and return top-k tokens by probability."""
    import mlx.core as mx

    tokens = tokenizer.encode(prompt)
    x = mx.array([tokens])

    logits = model(x)
    last_logits = logits[0, -1, :]

    # Top-k
    probs = mx.softmax(last_logits)
    mx.eval(probs)

    top_indices = mx.argpartition(-probs, kth=top_k)[:top_k]
    mx.eval(top_indices)

    results = []
    for idx in top_indices.tolist():
        token = tokenizer.decode([idx]).strip()
        prob = float(probs[idx])
        results.append((token, prob))

    results.sort(key=lambda x: -x[1])
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate graph against model inference")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--graph", required=True, help="Path to .larql.json graph")
    parser.add_argument("--entities", required=True, help="Comma-separated entities or file")
    parser.add_argument("--template", default="{entity}", help="Prompt template with {entity}")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k model predictions to check")
    args = parser.parse_args()

    # Parse entities
    if args.entities.endswith(".txt"):
        entities = [l.strip() for l in open(args.entities) if l.strip()]
    else:
        entities = [e.strip() for e in args.entities.split(",") if e.strip()]

    # Load model
    print(f"Loading model: {args.model}", file=sys.stderr)
    import mlx_lm
    model, tokenizer = mlx_lm.load(args.model)
    print(f"  Model loaded.", file=sys.stderr)

    # Load graph (once — it's big)
    print(f"Loading graph: {args.graph}", file=sys.stderr)
    start = time.time()
    with open(args.graph) as f:
        graph_data = json.load(f)
    graph_edges = {}
    for e in graph_data.get("edges", []):
        subj = e.get("s", "")
        if subj not in graph_edges:
            graph_edges[subj] = []
        graph_edges[subj].append({
            "relation": e.get("r", ""),
            "object": e.get("o", ""),
            "confidence": e.get("c", 0),
        })
    # Sort each entity's edges by confidence
    for subj in graph_edges:
        graph_edges[subj].sort(key=lambda x: -x["confidence"])
    print(f"  {len(graph_edges)} subjects loaded ({time.time()-start:.1f}s)", file=sys.stderr)

    # Validate
    print(f"\n{'Entity':<20} {'Model top-1':<20} {'Graph top-1':<20} {'Graph top-3':<40} {'Match'}")
    print("─" * 120)

    matches = 0
    top3_matches = 0
    tested = 0

    for entity in entities:
        prompt = args.template.replace("{entity}", entity)

        # Model prediction
        model_preds = model_predict_logprobs(model, tokenizer, prompt, args.top_k)
        model_top1 = model_preds[0][0] if model_preds else "?"
        model_top_tokens = {t.lower().strip() for t, _ in model_preds[:5]}

        # Graph prediction
        entity_edges = graph_edges.get(entity, [])
        if not entity_edges:
            # Try case variants
            for key in graph_edges:
                if key.lower() == entity.lower():
                    entity_edges = graph_edges[key]
                    break

        graph_top1 = entity_edges[0]["object"] if entity_edges else "?"
        graph_top3 = [e["object"] for e in entity_edges[:3]]
        graph_top3_set = {t.lower().strip() for t in graph_top3}

        # Check match
        exact = model_top1.lower().strip() == graph_top1.lower().strip()
        fuzzy = bool(model_top_tokens & graph_top3_set)

        if exact:
            match_str = "✓ exact"
            matches += 1
            top3_matches += 1
        elif fuzzy:
            match_str = "~ top-3"
            top3_matches += 1
        else:
            match_str = "✗"

        tested += 1
        graph_top3_str = ", ".join(graph_top3[:3]) if graph_top3 else "?"
        print(f"{entity:<20} {model_top1:<20} {graph_top1:<20} {graph_top3_str:<40} {match_str}")

    print("─" * 120)
    print(f"\nResults: {tested} tested, {matches} exact matches ({100*matches/tested:.0f}%), "
          f"{top3_matches} top-3 matches ({100*top3_matches/tested:.0f}%)")


if __name__ == "__main__":
    main()
