"""
LQL session — query language + numpy access in one object.

Usage:
    python examples/demos/session.py [path/to/model.vindex]
"""

import sys
import numpy as np
import larql

session = larql.session(sys.argv[1] if len(sys.argv) > 1 else "output/gemma3-4b-v2.vindex")
print(session)
print()

# LQL queries
print("STATS:")
for line in session.query("STATS")[:5]:
    print(f"  {line}")

print()
print("DESCRIBE 'France':")
for line in session.query("DESCRIBE 'France'")[:8]:
    print(f"  {line}")

print()
print("WALK 'The capital of France is' TOP 5:")
for line in session.query("WALK 'The capital of France is' TOP 5")[:8]:
    print(f"  {line}")

# Direct numpy access on same session
print()
v = session.vindex
gates = v.gate_vectors(layer=v.layer_bands()["knowledge"][1])
print(f"Gate vectors L27: {gates.shape} ({gates.nbytes / 1e6:.0f} MB)")
