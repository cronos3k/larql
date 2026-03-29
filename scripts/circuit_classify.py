#!/usr/bin/env python3
"""Classify all FFN features at a layer by circuit type.

Computes cosine(gate, down) for every feature and outputs a classification table.

Usage:
    python scripts/circuit_classify.py --layer 26 --ns larql --db gemma3_4b

Requires: requests (pip install requests)
"""

import argparse
import json
import sys
import time
from collections import Counter

import requests


def query_surreal(endpoint, ns, db, user, password, sql):
    """Execute a SurQL query and return parsed results."""
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


def classify_circuit(cos):
    """Classify a feature by its gate-down cosine similarity."""
    if cos > 0.5:
        return "identity"
    elif cos > 0.2:
        return "transform"
    elif cos > -0.2:
        return "projector"
    elif cos > -0.5:
        return "suppressor"
    else:
        return "inverter"


def main():
    parser = argparse.ArgumentParser(description="Classify FFN circuit types at a layer")
    parser.add_argument("--layer", type=int, default=26)
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--ns", default="larql")
    parser.add_argument("--db", default="gemma3_4b")
    parser.add_argument("--user", default="root")
    parser.add_argument("--password", default="root")
    parser.add_argument("--output", default=None, help="Output JSON file")
    parser.add_argument("--batch-size", type=int, default=50, help="Features per query batch")
    args = parser.parse_args()

    layer = args.layer

    # Count features
    result = query_surreal(
        args.endpoint, args.ns, args.db, args.user, args.password,
        f"SELECT count() FROM ffn_gate WHERE layer = {layer} GROUP ALL;"
    )
    total = result[0]["result"][0]["count"]
    print(f"Layer {layer}: {total} features", file=sys.stderr)

    features = []
    start = time.time()

    # Batch query — fetch features in chunks
    for offset in range(0, total, args.batch_size):
        batch_size = min(args.batch_size, total - offset)
        sql_parts = []
        for i in range(offset, offset + batch_size):
            sql_parts.append(
                f"LET $g{i} = (SELECT vector FROM ONLY ffn_gate:L{layer}_F{i}).vector;"
                f"LET $d{i} = (SELECT vector FROM ONLY ffn_down:L{layer}_F{i}).vector;"
                f"RETURN {{"
                f"  f: {i},"
                f"  gate: (SELECT top_token FROM ONLY ffn_gate:L{layer}_F{i}).top_token,"
                f"  down: (SELECT top_token FROM ONLY ffn_down:L{layer}_F{i}).top_token,"
                f"  cos: vector::similarity::cosine($g{i}, $d{i})"
                f"}};"
            )

        sql = "\n".join(sql_parts)
        result = query_surreal(
            args.endpoint, args.ns, args.db, args.user, args.password, sql
        )

        # Parse results — every 3rd result is the RETURN (after two LETs)
        for item in result:
            if item.get("result") and isinstance(item["result"], dict) and "f" in item["result"]:
                r = item["result"]
                cos = r["cos"]
                features.append({
                    "feature": r["f"],
                    "gate_token": r["gate"],
                    "down_token": r["down"],
                    "cosine": round(cos, 6),
                    "circuit_type": classify_circuit(cos),
                })

        elapsed = time.time() - start
        done = min(offset + batch_size, total)
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        print(
            f"\r  {done}/{total} ({elapsed:.0f}s, ETA {eta:.0f}s)",
            end="", file=sys.stderr,
        )

    print(file=sys.stderr)
    elapsed = time.time() - start
    print(f"Classified {len(features)} features in {elapsed:.1f}s", file=sys.stderr)

    # Distribution
    types = Counter(f["circuit_type"] for f in features)
    print(f"\nCircuit type distribution (L{layer}):", file=sys.stderr)
    for ctype in ["identity", "transform", "projector", "suppressor", "inverter"]:
        count = types.get(ctype, 0)
        pct = count / len(features) * 100 if features else 0
        print(f"  {ctype:12s}: {count:5d} ({pct:.1f}%)", file=sys.stderr)

    # Top examples per type
    for ctype in ["identity", "transform", "projector", "inverter"]:
        examples = [f for f in features if f["circuit_type"] == ctype]
        if ctype == "inverter":
            examples.sort(key=lambda x: x["cosine"])
        else:
            examples.sort(key=lambda x: -x["cosine"])
        print(f"\n  Top {ctype}:", file=sys.stderr)
        for f in examples[:5]:
            print(
                f"    F{f['feature']:5d}: {f['gate_token']:15s} → {f['down_token']:15s} cos={f['cosine']:.3f}",
                file=sys.stderr,
            )

    # Output
    if args.output:
        with open(args.output, "w") as f:
            json.dump({"layer": layer, "features": features}, f, indent=2)
        print(f"\nSaved to {args.output}", file=sys.stderr)
    else:
        json.dump({"layer": layer, "features": features}, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
