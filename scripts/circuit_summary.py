#!/usr/bin/env python3
"""Summarise circuit type distributions across all layers.

Usage:
    python scripts/circuit_summary.py output/circuits/
"""

import json
import sys
from pathlib import Path


def main():
    circuit_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/circuits")

    print(f"{'Layer':>5}  {'Proj':>6}  {'Trans':>6}  {'Supp':>6}  {'Ident':>6}  {'Inv':>6}  Role")
    print("─" * 70)

    for layer in range(34):
        path = circuit_dir / f"L{layer}_circuits.json"
        if not path.exists():
            continue

        data = json.load(open(path))
        features = data["features"]
        n = len(features)
        if n == 0:
            continue

        from collections import Counter
        types = Counter(f["circuit_type"] for f in features)

        proj = types.get("projector", 0) / n * 100
        trans = types.get("transform", 0) / n * 100
        supp = types.get("suppressor", 0) / n * 100
        ident = types.get("identity", 0) / n * 100
        inv = types.get("inverter", 0) / n * 100

        active = trans + supp + ident + inv
        if active < 5:
            role = "passive"
        elif ident + inv > 8:
            role = "FORMAT GATE"
        elif active > 35:
            role = "ACTIVE"
        elif proj > 85:
            role = "knowledge"
        else:
            role = "mixed"

        print(f"L{layer:>2}    {proj:5.1f}%  {trans:5.1f}%  {supp:5.1f}%  {ident:5.1f}%  {inv:5.1f}%  {role}")

    # Save as JSON summary
    summary_path = circuit_dir / "summary.json"
    summary = []
    for layer in range(34):
        path = circuit_dir / f"L{layer}_circuits.json"
        if not path.exists():
            continue
        data = json.load(open(path))
        features = data["features"]
        n = len(features)
        from collections import Counter
        types = Counter(f["circuit_type"] for f in features)
        summary.append({
            "layer": layer,
            "total": n,
            "projector": types.get("projector", 0),
            "transform": types.get("transform", 0),
            "suppressor": types.get("suppressor", 0),
            "identity": types.get("identity", 0),
            "inverter": types.get("inverter", 0),
        })

    json.dump(summary, open(summary_path, "w"), indent=2)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
