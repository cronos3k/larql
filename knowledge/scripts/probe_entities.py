#!/usr/bin/env python3
"""Probe entities through the model to confirm which features fire.

Runs a small set of entities through the model (via the vindex walk),
captures gate activations at L14-27, and pairs them with down_meta outputs.

This is a quick test to see if probe-based pair matching works before
scaling to the full 16K entity set.

Usage:
    python3 scripts/probe_entities.py

Requires: a vindex directory (pass via --vindex)
"""

import json
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer


def load_vindex(vindex_dir):
    """Load gate vectors, embeddings, down_meta, and tokenizer from a vindex."""
    vindex_dir = Path(vindex_dir)

    # Load config
    with open(vindex_dir / "index.json") as f:
        config = json.load(f)

    hidden_size = config["hidden_size"]
    vocab_size = config["vocab_size"]
    embed_scale = config["embed_scale"]

    # Load embeddings
    embed_raw = np.fromfile(vindex_dir / "embeddings.bin", dtype=np.float32)
    embed = embed_raw.reshape(vocab_size, hidden_size)

    # Load gate vectors
    gate_raw = np.fromfile(vindex_dir / "gate_vectors.bin", dtype=np.float32)
    gates = {}
    for layer_info in config["layers"]:
        layer = layer_info["layer"]
        nf = layer_info["num_features"]
        offset = layer_info["offset"] // 4  # byte offset to float offset
        gates[layer] = gate_raw[offset:offset + nf * hidden_size].reshape(nf, hidden_size)

    # Load down_meta
    down_meta = {}  # (layer, feature) → top_token
    with open(vindex_dir / "down_meta.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            layer = obj.get("l", 0)
            feat = obj.get("f", 0)
            tok = obj.get("t", "")
            down_meta[(layer, feat)] = tok

    # Load tokenizer
    tokenizer = Tokenizer.from_file(str(vindex_dir / "tokenizer.json"))

    return config, embed, embed_scale, gates, down_meta, tokenizer


def embed_entity(entity, embed, embed_scale, tokenizer):
    """Get the scaled embedding for an entity (averaged if multi-token)."""
    encoding = tokenizer.encode(entity, add_special_tokens=False)
    ids = [i for i in encoding.ids if i > 2]
    if not ids:
        return None
    vecs = [embed[i] * embed_scale for i in ids]
    avg = np.mean(vecs, axis=0)
    return avg


def gate_knn(query, gate_matrix, top_k=10):
    """Find top-K features by gate dot product."""
    scores = gate_matrix @ query
    top_indices = np.argsort(-np.abs(scores))[:top_k]
    return [(int(i), float(scores[i])) for i in top_indices]


def probe_entity(entity, embed, embed_scale, gates, down_meta, tokenizer, layers, top_k=10):
    """Probe a single entity: find which features fire and what they output."""
    query = embed_entity(entity, embed, embed_scale, tokenizer)
    if query is None:
        return []

    results = []
    for layer in layers:
        if layer not in gates:
            continue
        hits = gate_knn(query, gates[layer], top_k)
        for feat, score in hits:
            if score > 5.0:  # minimum gate score
                target = down_meta.get((layer, feat), "")
                if target and len(target) >= 2:
                    results.append({
                        "entity": entity,
                        "target": target,
                        "layer": layer,
                        "feature": feat,
                        "gate_score": round(score, 2),
                    })
    return results


def check_wikidata_match(entity, target, triples):
    """Check if (entity, target) matches any Wikidata triple."""
    matches = []
    entity_lower = entity.lower()
    target_lower = target.lower()
    for rel_name, rel_data in triples.items():
        for pair in rel_data.get("pairs", []):
            if len(pair) >= 2:
                if pair[0].lower() == entity_lower and pair[1].lower() == target_lower:
                    matches.append(rel_name)
    return matches


def main() -> None:
    """Probe test entities through the vindex and match against Wikidata."""
    import argparse
    parser = argparse.ArgumentParser(description="Probe entities through vindex")
    parser.add_argument("--vindex", type=str, required=True, help="Path to vindex directory")
    parser.add_argument("--triples", type=str, default="data/wikidata_triples.json", help="Path to triples JSON")
    args = parser.parse_args()

    vindex_dir = args.vindex
    triples_path = args.triples

    print("Loading vindex...")
    config, embed, embed_scale, gates, down_meta, tokenizer = load_vindex(vindex_dir)
    print(f"  {config['num_layers']} layers, {config['hidden_size']} hidden, {len(down_meta)} features")

    print("Loading triples...")
    with open(triples_path) as f:
        triples = json.load(f)
    total_pairs = sum(len(v["pairs"]) for v in triples.values())
    print(f"  {len(triples)} relations, {total_pairs} pairs")

    # Test entities
    test_entities = ["France", "Mozart", "Google", "Cheese", "Germany", "Shakespeare", "Japan"]
    layers = list(range(14, 28))

    print(f"\nProbing {len(test_entities)} entities at L14-27...")
    print()

    for entity in test_entities:
        results = probe_entity(entity, embed, embed_scale, gates, down_meta, tokenizer, layers, top_k=5)

        print(f"── {entity} ──")
        if not results:
            print("  (no activations)")
            continue

        # Sort by gate score
        results.sort(key=lambda r: -abs(r["gate_score"]))

        for r in results[:15]:
            target = r["target"]
            wikidata = check_wikidata_match(entity, target, triples)
            match_str = f" → {', '.join(wikidata)}" if wikidata else ""
            print(f"  L{r['layer']:2d} F{r['feature']:<5d}  gate={r['gate_score']:+.1f}  → {target:<20s}{match_str}")

        # Count Wikidata matches
        matched = sum(1 for r in results if check_wikidata_match(entity, r["target"], triples))
        print(f"  ({matched}/{len(results)} pairs match Wikidata)")
        print()


if __name__ == "__main__":
    main()
