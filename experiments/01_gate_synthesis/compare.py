"""
Experiment 1: Gate vector synthesis vs forward pass capture.

Hypothesis: heuristic gate synthesis (entity_embed * scale + relation_centre)
produces vectors close to the residuals a forward pass would generate.

If cosine > 0.9, the heuristic works. If < 0.5, forward pass capture is essential.
"""

import numpy as np
import json
import sys
import os

# Add parent for shared utils
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import _larql_core as larql


def load_relation_clusters(vindex_path: str) -> dict:
    """Load relation cluster centres from the vindex directory."""
    path = os.path.join(vindex_path, "relation_clusters.json")
    if not os.path.exists(path):
        print(f"No relation_clusters.json at {path}")
        return {}
    with open(path) as f:
        return json.load(f)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def run_experiment(vindex_path: str):
    print(f"Loading vindex from {vindex_path}...")
    vindex = larql.load_vindex(vindex_path)
    print(f"  {vindex}")

    clusters = load_relation_clusters(vindex_path)
    if not clusters:
        print("Cannot run without relation clusters. Exiting.")
        return

    # Test entities with known relations
    test_cases = [
        ("France", "capital"),
        ("Germany", "capital"),
        ("Japan", "language"),
        ("Einstein", "occupation"),
        ("Python", "creator"),
    ]

    results = []
    bands = vindex.layer_bands()
    knowledge_start = bands["knowledge"][0] if bands else 14
    knowledge_end = bands["knowledge"][1] if bands else 27

    for entity, relation in test_cases:
        print(f"\n--- {entity} / {relation} ---")

        # Route A: heuristic synthesis
        entity_embed = np.array(vindex.embed(entity))

        if relation not in clusters:
            print(f"  Relation '{relation}' not in clusters, skipping")
            continue

        cluster_centre = np.array(clusters[relation]["centre"], dtype=np.float32)

        # Simple heuristic: entity embedding + relation direction
        gate_heuristic = entity_embed + cluster_centre

        # Route B: find what the vindex actually has (closest real gate vector)
        # Walk knowledge layers with entity embedding to find matching features
        for layer in range(knowledge_start, knowledge_end + 1):
            hits = vindex.entity_knn(entity, layer=layer, top_k=5)
            if not hits:
                continue

            best_feat, best_score = hits[0]
            meta = vindex.feature_meta(layer, best_feat)
            if meta is None:
                continue

            # Get the actual gate vector stored in the vindex
            gate_actual = np.array(vindex.gate_vector(layer, best_feat))

            cos = cosine_similarity(gate_heuristic, gate_actual)
            print(f"  L{layer} F{best_feat} token='{meta.top_token}' "
                  f"gate_score={best_score:.1f} cosine={cos:.4f}")

            results.append({
                "entity": entity,
                "relation": relation,
                "layer": layer,
                "feature": best_feat,
                "target_token": meta.top_token,
                "gate_score": best_score,
                "cosine_heuristic_vs_actual": cos,
            })

    # Summary
    if results:
        cosines = [r["cosine_heuristic_vs_actual"] for r in results]
        print(f"\n=== Summary ===")
        print(f"  Comparisons: {len(cosines)}")
        print(f"  Mean cosine:   {np.mean(cosines):.4f}")
        print(f"  Median cosine: {np.median(cosines):.4f}")
        print(f"  Min cosine:    {np.min(cosines):.4f}")
        print(f"  Max cosine:    {np.max(cosines):.4f}")

        # Save results
        out_path = os.path.join(os.path.dirname(__file__), "..", "results", "01_gate_cosine.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    vindex_path = sys.argv[1] if len(sys.argv) > 1 else "output/gemma3-4b-v2.vindex"
    run_experiment(vindex_path)
