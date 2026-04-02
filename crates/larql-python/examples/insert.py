"""
Insert knowledge into a vindex — no training.

Usage:
    python examples/demos/insert.py [path/to/model.vindex]
"""

import sys
import larql

vindex = larql.load(sys.argv[1] if len(sys.argv) > 1 else "output/gemma3-4b-v2.vindex")
print(vindex)
print()

# Insert facts
facts = [
    ("Colchester", "country", "England"),
    ("Colchester", "located_in", "Essex"),
    ("John Coyle", "occupation", "engineer"),
]

for entity, relation, target in facts:
    layer, feat = vindex.insert(entity, relation, target)
    meta = vindex.feature_meta(layer, feat)
    print(f"  {entity} --{relation}--> {target}  →  L{layer}:F{feat} (token='{meta.top_token}')")

# Verify
print()
print("Verify via describe:")
for entity in ["Colchester", "John Coyle"]:
    edges = vindex.describe(entity, verbose=True)
    for e in edges[:3]:
        print(f"  {entity}: {e.relation or '?'} → {e.target}  score={e.gate_score:.0f}")
