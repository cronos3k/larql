#!/usr/bin/env python3
"""Build per-feature relation labels from probe data + Wikidata matching.

For each entity in the reference data:
1. Embed it, run gate KNN at L14-27
2. For each feature that fires, pair entity with the feature's output token
3. Check if (entity, output) matches a Wikidata triple
4. If yes, that feature gets that relation label

Output: data/feature_labels.json — maps "L{layer}_F{feature}" → relation_name

Usage:
    python3 scripts/build_feature_labels.py
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_vindex(vindex_dir):
    """Load gate vectors, embeddings, down_meta from a vindex."""
    vindex_dir = Path(vindex_dir)

    with open(vindex_dir / "index.json") as f:
        config = json.load(f)

    hidden_size = config["hidden_size"]
    vocab_size = config["vocab_size"]
    embed_scale = config["embed_scale"]

    embed_raw = np.fromfile(vindex_dir / "embeddings.bin", dtype=np.float32)
    embed = embed_raw.reshape(vocab_size, hidden_size)

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
            layer = obj.get("l", 0)
            feat = obj.get("f", 0)
            tok = obj.get("t", "")
            down_meta[(layer, feat)] = tok

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(vindex_dir / "tokenizer.json"))

    return config, embed, embed_scale, gates, down_meta, tokenizer


def embed_entity(entity, embed, embed_scale, tokenizer):
    """Get averaged scaled embedding for an entity."""
    encoding = tokenizer.encode(entity, add_special_tokens=False)
    ids = [i for i in encoding.ids if i > 2]
    if not ids:
        return None
    vecs = [embed[i] * embed_scale for i in ids]
    return np.mean(vecs, axis=0)


def main() -> None:
    """Build per-feature relation labels from probe data and Wikidata matching."""
    import argparse
    parser = argparse.ArgumentParser(description="Build feature labels from vindex + triples")
    parser.add_argument("--vindex", type=str, required=True, help="Path to vindex directory")
    parser.add_argument("--triples", type=str, default="data/wikidata_triples.json", help="Path to triples JSON")
    args = parser.parse_args()

    vindex_dir = args.vindex
    triples_path = args.triples

    print("Loading vindex...")
    config, embed, embed_scale, gates, down_meta, tokenizer = load_vindex(vindex_dir)

    print("Loading triples...")
    with open(triples_path) as f:
        triples = json.load(f)

    # Build lookup: (entity_lower, target_lower) → relation_name
    pair_to_relation = {}
    for rel_name, rel_data in triples.items():
        for pair in rel_data.get("pairs", []):
            if len(pair) >= 2:
                key = (pair[0].lower(), pair[1].lower())
                pair_to_relation[key] = rel_name
    print(f"  {len(pair_to_relation)} unique (entity, target) pairs")

    # Extract unique entities from triples (both subjects and objects)
    entities = set()
    for rel_data in triples.values():
        for pair in rel_data.get("pairs", []):
            for item in pair:
                item = item.strip()
                if 2 <= len(item) <= 30 and "(" not in item and "http" not in item:
                    entities.add(item)
    entities = sorted(entities)
    print(f"  {len(entities)} unique entities to probe")

    # Probe and match
    layers = list(range(14, 28))
    feature_labels = {}  # "L{layer}_F{feat}" → relation_name
    feature_counts = defaultdict(int)  # relation → count
    total_probed = 0
    total_matched = 0

    for ei, entity in enumerate(entities):
        if ei % 2000 == 0 and ei > 0:
            print(f"  Probed {ei}/{len(entities)} entities, {total_matched} labels so far...")

        query = embed_entity(entity, embed, embed_scale, tokenizer)
        if query is None:
            continue

        total_probed += 1

        for layer in layers:
            if layer not in gates:
                continue
            scores = gates[layer] @ query
            top_indices = np.argsort(-np.abs(scores))[:5]

            for feat_idx in top_indices:
                score = float(scores[feat_idx])
                if abs(score) < 5.0:
                    continue

                target = down_meta.get((layer, int(feat_idx)), "")
                if len(target) < 2:
                    continue

                # Check Wikidata match
                key = (entity.lower(), target.lower())
                if key in pair_to_relation:
                    rel = pair_to_relation[key]
                    feat_key = f"L{layer}_F{feat_idx}"
                    if feat_key not in feature_labels:
                        feature_labels[feat_key] = rel
                        feature_counts[rel] += 1
                        total_matched += 1

    print(f"\nProbed {total_probed} entities")
    print(f"Labeled {len(feature_labels)} features")
    print(f"\nRelation distribution:")
    for rel, count in sorted(feature_counts.items(), key=lambda x: -x[1]):
        print(f"  {rel:<25s} {count:4d} features")

    # Save
    output_path = Path("data/feature_labels.json")
    with open(output_path, "w") as f:
        json.dump(feature_labels, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Also save in the vindex directory for the REPL to find
    vindex_labels_path = Path(vindex_dir) / "feature_labels.json"
    with open(vindex_labels_path, "w") as f:
        json.dump(feature_labels, f, indent=2)
    print(f"Saved to {vindex_labels_path}")


if __name__ == "__main__":
    main()
