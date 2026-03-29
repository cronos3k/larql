#!/usr/bin/env python3
"""Convert discovered edge JSONL files to .larql.json graph format.

Usage:
    python scripts/edges_to_larql.py output/edges/all_edges.jsonl -o output/discovered.larql.json
    python scripts/edges_to_larql.py output/edges/ -o output/discovered.larql.json
"""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Convert edge JSONL to .larql.json")
    parser.add_argument("input", help="JSONL file or directory of JSONL files")
    parser.add_argument("-o", "--output", required=True, help="Output .larql.json file")
    parser.add_argument("--min-down-dist", type=float, default=None, help="Max down_dist to include (e.g. 0.8)")
    args = parser.parse_args()

    # Collect input files
    if os.path.isdir(args.input):
        files = sorted(
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.endswith("_edges.jsonl")
        )
    else:
        files = [args.input]

    edges = []
    seen = set()
    total = 0
    filtered = 0

    for path in files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                total += 1

                # Optional distance filter
                if args.min_down_dist is not None and e.get("down_dist", 1.0) > args.min_down_dist:
                    filtered += 1
                    continue

                # Skip empty tokens
                source = e.get("source", "").strip()
                target = e.get("target", "").strip()
                if not source or not target:
                    filtered += 1
                    continue

                # Deduplicate by (source, relation, target)
                triple = (source, e["relation"], target)
                if triple in seen:
                    continue
                seen.add(triple)

                # Convert to compact edge format
                confidence = round(1.0 - e["down_dist"], 4)
                compact = {
                    "s": source,
                    "r": e["relation"],
                    "o": target,
                    "c": confidence,
                    "src": "parametric",
                    "meta": {
                        "layer": e["layer"],
                        "feature": e["feature"],
                        "gate_dist": e["gate_dist"],
                        "down_dist": e["down_dist"],
                        "circuit_type": e.get("circuit_type", "unknown"),
                    },
                }
                edges.append(compact)

    # Build graph
    graph = {
        "larql_version": "0.1.0",
        "metadata": {
            "model": "google/gemma-3-4b-it",
            "method": "edge-discovery",
            "extraction": "gate-down-knn",
            "total_features": total,
            "filtered": filtered,
            "edges": len(edges),
        },
        "schema": {},
        "edges": edges,
    }

    with open(args.output, "w") as f:
        json.dump(graph, f)

    size = os.path.getsize(args.output)
    print(f"Converted {total} features → {len(edges)} edges ({filtered} filtered)", file=sys.stderr)
    print(f"Output: {args.output} ({size / 1024 / 1024:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
