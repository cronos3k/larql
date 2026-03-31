#!/usr/bin/env python3
"""Probe entities with REAL model inference via the larql CLI.

Runs `larql walk --index ... --predict --verbose` for prompted templates,
parses the walk trace to find which features fire, and matches against
Wikidata triples.

This uses actual attention — the model processes the full prompt.

Usage:
    python3 scripts/probe_with_inference.py

Requires: compiled larql binary, vindex with model weights
Time: ~20-30 min for 18 templates × 50 entities
"""

import json
import subprocess
import sys
import re
from pathlib import Path
from collections import defaultdict


TEMPLATES = {
    "capital": "The capital of {X} is",
    "language": "The official language of {X} is",
    "continent": "{X} is located in the continent of",
    "borders": "{X} shares a border with",
    "occupation": "{X} was a famous",
    "birthplace": "{X} was born in",
    "currency": "The currency of {X} is the",
    "located in": "{X} is located in",
    "author": "The author of {X} was",
    "director": "{X} was directed by",
    "genre": "The genre of {X} is",
    "founder": "{X} was founded by",
    "nationality": "The nationality of {X} is",
}

VINDEX = None  # Set via --vindex argument
BINARY = "target/release/larql"


def run_walk(prompt, vindex, top_k=10):
    """Run larql walk --predict --verbose and parse the trace."""
    try:
        result = subprocess.run(
            [BINARY, "walk", "--index", vindex, "-p", prompt,
             "--predict", "--verbose", "-k", str(top_k)],
            capture_output=True, text=True, timeout=60,
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return ""
    except FileNotFoundError:
        return ""


def parse_walk_trace(output):
    """Parse the walk trace to extract (layer, feature, gate_score, top_token) tuples."""
    features = []
    # Parse lines like: "  1. F9515  gate=+9.200  hears="Paris"  c=0.89  down=[...]"
    # Or: "Layer 27:" followed by feature lines
    current_layer = None

    for line in output.split("\n"):
        line = line.strip()

        # Match "Layer N:" header
        m = re.match(r"Layer (\d+):", line)
        if m:
            current_layer = int(m.group(1))
            continue

        # Match feature line
        m = re.match(r"\d+\.\s+F(\d+)\s+gate=([+-]?\d+\.?\d*)\s+hears=\"?([^\"]+)\"?\s+c=(\d+\.?\d*)", line)
        if m and current_layer is not None:
            feat = int(m.group(1))
            gate = float(m.group(2))
            top_token = m.group(3).strip()
            features.append((current_layer, feat, gate, top_token))

    return features


def main() -> None:
    """Probe entities via larql CLI walk and match against Wikidata triples."""
    global VINDEX
    import argparse
    parser = argparse.ArgumentParser(description="Probe entities via larql CLI walk")
    parser.add_argument("--vindex", type=str, required=True, help="Path to vindex directory")
    parser.add_argument("--triples", type=str, default="data/wikidata_triples.json", help="Path to triples JSON")
    parser.add_argument("--binary", type=str, default=BINARY, help="Path to larql binary")
    args = parser.parse_args()

    VINDEX = args.vindex
    triples_path = args.triples

    # Check binary exists
    if not Path(args.binary).exists():
        print(f"Build first: cargo build --release")
        sys.exit(1)

    print("Loading triples...")
    with open(triples_path) as f:
        triples = json.load(f)

    pair_to_relation = {}
    for rel_name, rel_data in triples.items():
        for pair in rel_data.get("pairs", []):
            if len(pair) >= 2:
                pair_to_relation[(pair[0].lower(), pair[1].lower())] = rel_name

    feature_labels = {}
    relation_counts = defaultdict(int)
    total_probes = 0
    total_matched = 0

    print(f"Probing with {len(TEMPLATES)} templates via model inference...")
    print(f"Binary: {BINARY}")
    print(f"Vindex: {VINDEX}")
    print()

    for rel_name, template in TEMPLATES.items():
        if rel_name not in triples:
            continue

        subjects = list(set(
            pair[0] for pair in triples[rel_name].get("pairs", [])
            if len(pair) >= 2 and len(pair[0]) <= 25 and " " not in pair[0]
        ))[:20]  # Start small — 20 per relation

        if not subjects:
            print(f"  {rel_name}: no single-word subjects")
            continue

        matched = 0
        for si, subject in enumerate(subjects):
            prompt = template.replace("{X}", subject)
            output = run_walk(prompt, VINDEX)
            total_probes += 1

            features = parse_walk_trace(output)

            # Check each feature's output against Wikidata
            for layer, feat, gate, top_token in features:
                if 14 <= layer <= 27 and abs(gate) > 5.0:
                    key = (subject.lower(), top_token.lower())
                    if key in pair_to_relation and pair_to_relation[key] == rel_name:
                        feat_key = f"L{layer}_F{feat}"
                        if feat_key not in feature_labels:
                            feature_labels[feat_key] = rel_name
                            relation_counts[rel_name] += 1
                            total_matched += 1
                            matched += 1

            if (si + 1) % 5 == 0:
                sys.stdout.write(f"\r  {rel_name:<20s} {si+1}/{len(subjects)} entities, {matched} labels")
                sys.stdout.flush()

        print(f"\r  {rel_name:<20s} {len(subjects):3d} entities  → {matched} features labeled")

    print(f"\nTotal probes: {total_probes}")
    print(f"Total labeled: {len(feature_labels)} features")

    if relation_counts:
        print(f"\nRelation distribution:")
        for rel, count in sorted(relation_counts.items(), key=lambda x: -x[1]):
            print(f"  {rel:<25s} {count:4d}")

    # Merge with existing
    existing_path = Path(VINDEX) / "feature_labels.json"
    existing = {}
    if existing_path.exists():
        with open(existing_path) as f:
            existing = json.load(f)

    new_count = 0
    for key, rel in feature_labels.items():
        if key not in existing:
            existing[key] = rel
            new_count += 1

    with open(existing_path, "w") as f:
        json.dump(existing, f, indent=2)

    data_path = Path("data/feature_labels.json")
    with open(data_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\nMerged: {new_count} new + {len(existing) - new_count} existing = {len(existing)} total")
    print(f"Saved to {existing_path}")


if __name__ == "__main__":
    main()
