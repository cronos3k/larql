#!/usr/bin/env python3
"""Discover factual edges from FFN features via gate/down KNN against embeddings.

For each feature at a layer:
  1. Gate KNN → what entity triggers this feature
  2. Down KNN → what entity/concept it produces
  3. Emit edge with distances as confidence metadata

No forward pass. No model loaded. Pure SurrealDB queries.

Usage:
    python scripts/edge_discover.py --layer 26 --output output/edges/L26_edges.jsonl
    python scripts/edge_discover.py --layer 26 --layers 23,24,25,26,27,28,29 -o output/edges/
"""

import argparse
import json
import os
import sys
import time

import requests


def query_surreal(endpoint, ns, db, user, password, sql):
    resp = requests.post(
        f"{endpoint}/sql",
        headers={
            "surreal-ns": ns,
            "surreal-db": db,
            "Accept": "application/json",
        },
        auth=(user, password),
        data=sql,
    )
    resp.raise_for_status()
    return resp.json()


def discover_layer(endpoint, ns, db, user, password, layer, top_k_embed, circuit_data):
    """Discover edges for all features at a layer."""

    # Count features
    result = query_surreal(
        endpoint, ns, db, user, password,
        f"SELECT count() FROM ffn_gate WHERE layer = {layer} GROUP ALL;"
    )
    total = result[0]["result"][0]["count"]

    edges = []
    start = time.time()

    for feat in range(total):
        # Gate KNN: what triggers this feature
        sql = (
            f"LET $v = (SELECT vector FROM ONLY ffn_gate:L{layer}_F{feat}).vector;\n"
            f"SELECT top_token, vector::distance::knn() AS dist "
            f"FROM embeddings WHERE vector <|1, COSINE|> $v ORDER BY dist LIMIT 1;"
        )
        result = query_surreal(endpoint, ns, db, user, password, sql)
        gate_token = ""
        gate_dist = 1.0
        for item in result:
            if isinstance(item.get("result"), list) and item["result"]:
                gate_token = item["result"][0].get("top_token", "")
                gate_dist = item["result"][0].get("dist", 1.0)

        # Down KNN: what this feature produces
        sql = (
            f"LET $v = (SELECT vector FROM ONLY ffn_down:L{layer}_F{feat}).vector;\n"
            f"SELECT top_token, vector::distance::knn() AS dist "
            f"FROM embeddings WHERE vector <|1, COSINE|> $v ORDER BY dist LIMIT 1;"
        )
        result = query_surreal(endpoint, ns, db, user, password, sql)
        down_token = ""
        down_dist = 1.0
        for item in result:
            if isinstance(item.get("result"), list) and item["result"]:
                down_token = item["result"][0].get("top_token", "")
                down_dist = item["result"][0].get("dist", 1.0)

        # Circuit type from pre-computed classification
        circuit_type = "unknown"
        if circuit_data and str(feat) in circuit_data:
            circuit_type = circuit_data[str(feat)]

        edge = {
            "source": gate_token,
            "target": down_token,
            "relation": f"L{layer}-F{feat}",
            "layer": layer,
            "feature": feat,
            "gate_dist": round(gate_dist, 6),
            "down_dist": round(down_dist, 6),
            "circuit_type": circuit_type,
        }
        edges.append(edge)

        if (feat + 1) % 100 == 0:
            elapsed = time.time() - start
            rate = (feat + 1) / elapsed
            eta = (total - feat - 1) / rate
            print(
                f"\r  L{layer}: {feat+1}/{total} ({elapsed:.0f}s, {rate:.0f}/s, ETA {eta:.0f}s)",
                end="", file=sys.stderr,
            )

    print(file=sys.stderr)
    return edges


def main():
    parser = argparse.ArgumentParser(description="Discover factual edges from FFN gate/down KNN")
    parser.add_argument("--layer", type=int, default=None, help="Single layer")
    parser.add_argument("--layers", type=str, default=None, help="Comma-separated layers")
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--ns", default="larql")
    parser.add_argument("--db", default="gemma3_4b")
    parser.add_argument("--user", default="root")
    parser.add_argument("--password", default="root")
    parser.add_argument("-o", "--output", required=True, help="Output JSONL file or directory")
    parser.add_argument("--circuits-dir", default="output/circuits", help="Directory with circuit classification JSONs")
    args = parser.parse_args()

    # Determine layers
    if args.layers:
        layers = [int(l) for l in args.layers.split(",")]
    elif args.layer is not None:
        layers = [args.layer]
    else:
        layers = [26]

    overall_start = time.time()
    total_edges = 0

    for layer in layers:
        # Load circuit classifications if available
        circuit_data = {}
        circuit_path = os.path.join(args.circuits_dir, f"L{layer}_circuits.json")
        if os.path.exists(circuit_path):
            data = json.load(open(circuit_path))
            circuit_data = {str(f["feature"]): f["circuit_type"] for f in data["features"]}

        print(f"Layer {layer}: discovering edges...", file=sys.stderr)
        edges = discover_layer(
            args.endpoint, args.ns, args.db, args.user, args.password,
            layer, 1, circuit_data,
        )

        # Determine output path
        if os.path.isdir(args.output) or len(layers) > 1:
            os.makedirs(args.output, exist_ok=True)
            out_path = os.path.join(args.output, f"L{layer}_edges.jsonl")
        else:
            out_path = args.output

        with open(out_path, "w") as f:
            for edge in edges:
                f.write(json.dumps(edge) + "\n")

        # Summary stats
        clean = sum(1 for e in edges if e["down_dist"] < 0.8)
        total_edges += len(edges)

        print(
            f"  L{layer}: {len(edges)} edges, {clean} clean (down_dist < 0.8), "
            f"saved to {out_path}",
            file=sys.stderr,
        )

    elapsed = time.time() - overall_start
    print(f"\nTotal: {total_edges} edges in {elapsed:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
