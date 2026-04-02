"""
Demo: Basic vindex operations — load, describe, walk, insert.

Shows the core Python API without MLX dependency.

Usage:
    python examples/demos/basic_vindex.py [path/to/model.vindex]
"""

import sys
import os
import numpy as np

import larql

VINDEX_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "..", "..", "output", "gemma3-4b-v2.vindex"
)


def main():
    # ── Load ──
    print("Loading vindex...")
    vindex = larql.load(VINDEX_PATH)
    print(f"  {vindex}")
    print(f"  Layers: {vindex.num_layers}, Hidden: {vindex.hidden_size}")
    print(f"  Features: {vindex.total_gate_vectors:,}")
    print(f"  Bands: {vindex.layer_bands()}")

    # ── Embeddings ──
    print("\n─── Embeddings ───")
    france = vindex.embed("France")
    germany = vindex.embed("Germany")
    japan = vindex.embed("Japan")

    cos_fg = float(np.dot(france, germany) / (np.linalg.norm(france) * np.linalg.norm(germany)))
    cos_fj = float(np.dot(france, japan) / (np.linalg.norm(france) * np.linalg.norm(japan)))
    print(f"  France·Germany = {cos_fg:.4f}")
    print(f"  France·Japan   = {cos_fj:.4f}")

    # ── Describe ──
    print("\n─── DESCRIBE 'France' ───")
    edges = vindex.describe("France")
    for edge in edges[:10]:
        rel = edge.relation or "?"
        src = f"[{edge.source}]" if edge.source != "none" else ""
        also = f"  also: {', '.join(edge.also)}" if edge.also else ""
        print(f"  {rel:>20} → {edge.target:<15} score={edge.gate_score:>7.1f} L{edge.layer} {src}{also}")

    # ── Relations ──
    print("\n─── Relations (top 20) ───")
    rels = vindex.relations()
    for r in rels[:20]:
        tops = ", ".join(r.top_tokens[:3]) if r.top_tokens else ""
        print(f"  {r.name:<25} count={r.count:>5}  e.g. [{tops}]")

    # ── Cluster centres ──
    print("\n─── Cluster Centres ───")
    for rel_name in ["capital", "language", "country", "occupation", "author"]:
        centre = vindex.cluster_centre(rel_name)
        if centre is not None:
            layer = vindex.typical_layer(rel_name)
            print(f"  {rel_name:<15} dim={centre.shape[0]}, norm={np.linalg.norm(centre):.1f}, typical_layer=L{layer}")
        else:
            print(f"  {rel_name:<15} (not found)")

    # ── Entity Walk ──
    print("\n─── Entity Walk: 'Einstein' ───")
    bands = vindex.layer_bands()
    hits = vindex.entity_walk("Einstein",
        layers=list(range(bands["knowledge"][0], bands["knowledge"][1] + 1)),
        top_k=3)
    seen = set()
    for hit in hits:
        key = (hit.layer, hit.top_token)
        if key in seen: continue
        seen.add(key)
        label = vindex.feature_label(hit.layer, hit.feature) or ""
        label_str = f" [{label}]" if label else ""
        print(f"  L{hit.layer:2d} F{hit.feature:>5} score={hit.gate_score:>7.1f} → '{hit.top_token}'{label_str}")
        if len(seen) >= 15: break

    # ── Gate KNN comparison ──
    print("\n─── Relation Steering ───")
    capital_centre = vindex.cluster_centre("capital")
    language_centre = vindex.cluster_centre("language")
    if capital_centre is not None and language_centre is not None:
        france_embed = vindex.embed("France")
        layer = bands["knowledge"][1]  # last knowledge layer

        # Steer toward capital
        steered_capital = france_embed + 2.0 * capital_centre
        hits_cap = vindex.gate_knn(layer=layer, query_vector=steered_capital.tolist(), top_k=5)
        cap_tokens = []
        for f, s in hits_cap:
            m = vindex.feature_meta(layer, f)
            if m: cap_tokens.append(m.top_token)

        # Steer toward language
        steered_lang = france_embed + 2.0 * language_centre
        hits_lang = vindex.gate_knn(layer=layer, query_vector=steered_lang.tolist(), top_k=5)
        lang_tokens = []
        for f, s in hits_lang:
            m = vindex.feature_meta(layer, f)
            if m: lang_tokens.append(m.top_token)

        print(f"  France + capital direction → {cap_tokens}")
        print(f"  France + language direction → {lang_tokens}")

    # ── Insert ──
    print("\n─── INSERT ───")
    layer, feat = vindex.insert("TestEntity", "capital", "TestCapital")
    meta = vindex.feature_meta(layer, feat)
    print(f"  Inserted: L{layer} F{feat} → '{meta.top_token}' (c={meta.c_score:.2f})")

    # Verify via KNN
    hits = vindex.entity_knn("TestEntity", layer=layer, top_k=5)
    found = False
    for f, s in hits:
        if f == feat:
            print(f"  Verified: entity_knn finds it at score={s:.1f}")
            found = True
            break
    if not found:
        print(f"  Note: entity_knn didn't find it in top 5 (expected for synthetic)")

    # ── has_edge / get_target ──
    print("\n─── Edge Queries ───")
    print(f"  has_edge('France') = {vindex.has_edge('France')}")
    print(f"  has_edge('France', 'capital') = {vindex.has_edge('France', 'capital')}")
    target = vindex.get_target("France", "capital")
    print(f"  get_target('France', 'capital') = {target}")

    # ── LQL Session ──
    print("\n─── LQL Session ───")
    session = larql.session(VINDEX_PATH)
    print(f"  {session}")

    result = session.query("STATS")
    for line in result[:5]:
        print(f"  {line}")

    print("\n  DESCRIBE 'Germany':")
    result = session.query("DESCRIBE 'Germany'")
    for line in result[:8]:
        print(f"  {line}")

    # ── Bulk access for research ──
    print("\n─── Bulk Gate Vectors (for SVD/PCA) ───")
    layer = bands["knowledge"][1]
    gates = vindex.gate_vectors(layer=layer)
    print(f"  Layer {layer}: {gates.shape[0]} vectors × {gates.shape[1]}D")
    print(f"  Memory: {gates.nbytes / 1e6:.1f} MB")
    print(f"  Mean norm: {np.linalg.norm(gates, axis=1).mean():.1f}")

    # Quick SVD to show dimensionality
    centred = gates - gates.mean(axis=0)
    _, S, _ = np.linalg.svd(centred[:1000], full_matrices=False)  # sample for speed
    cumvar = np.cumsum(S**2) / np.sum(S**2)
    for thresh in [0.90, 0.95, 0.99]:
        d = int(np.searchsorted(cumvar, thresh)) + 1
        print(f"  {thresh*100:.0f}% variance in {d}D (compression {vindex.hidden_size/d:.0f}x)")

    print("\nDone.")


if __name__ == "__main__":
    main()
