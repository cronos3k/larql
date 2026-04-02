"""
Knowledge queries — describe, relations, steering.

No inference needed. Everything comes from the vindex index.

Usage:
    python examples/demos/knowledge.py [path/to/model.vindex]
"""

import sys
import numpy as np
import larql

vindex = larql.load(sys.argv[1] if len(sys.argv) > 1 else "output/gemma3-4b-v2.vindex")
print(vindex)
print()

# Describe
for entity in ["France", "Einstein", "Python"]:
    print(f"{entity}:")
    for e in vindex.describe(entity)[:5]:
        rel = e.relation or "?"
        also = f"  also: {', '.join(e.also)}" if e.also else ""
        print(f"  {rel:>15} → {e.target:<15} score={e.gate_score:.0f} L{e.layer}{also}")
    print()

# Relations
print("Top relations:")
for r in vindex.relations()[:10]:
    print(f"  {r.name:<25} count={r.count}")
print()

# Steering
capital = vindex.cluster_centre("capital")
language = vindex.cluster_centre("language")
if capital is not None and language is not None:
    france = vindex.embed("France")
    layer = vindex.layer_bands()["knowledge"][1]

    cap_hits = vindex.gate_knn(layer, (france + 2.0 * capital).tolist(), top_k=5)
    lang_hits = vindex.gate_knn(layer, (france + 2.0 * language).tolist(), top_k=5)

    cap = [vindex.feature_meta(layer, f).top_token for f, _ in cap_hits if vindex.feature_meta(layer, f)]
    lang = [vindex.feature_meta(layer, f).top_token for f, _ in lang_hits if vindex.feature_meta(layer, f)]
    print("Relation steering:")
    print(f"  France + capital  → {cap}")
    print(f"  France + language → {lang}")
