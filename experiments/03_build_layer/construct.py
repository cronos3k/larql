"""
Experiment 3: Build a knowledge layer from Wikidata triples.

Hypothesis: you can construct working knowledge layers (L14-27) from a database
of (entity, relation, target) triples, without any training.

Method:
1. For each triple: gate = embed(entity), down = embed(target)
2. Assign to layer by relation type
3. Write as gate+down vectors in the vindex
4. Query with DESCRIBE/entity_knn — does France -> Paris?
"""

import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import _larql_core as larql


# Test triples — things we know should work
TEST_TRIPLES = [
    # (entity, relation, target, expected_layer_range)
    ("France", "capital", "Paris"),
    ("Germany", "capital", "Berlin"),
    ("Japan", "capital", "Tokyo"),
    ("Italy", "capital", "Rome"),
    ("Spain", "capital", "Madrid"),
    ("France", "language", "French"),
    ("Germany", "language", "German"),
    ("Japan", "language", "Japanese"),
    ("Einstein", "occupation", "physicist"),
    ("Mozart", "occupation", "composer"),
    ("Shakespeare", "occupation", "playwright"),
    ("Python", "creator", "Guido"),
]


def run_experiment(vindex_path: str):
    print(f"Loading vindex from {vindex_path}...")
    vindex = larql.load_vindex(vindex_path)
    print(f"  {vindex}")

    bands = vindex.layer_bands()
    knowledge_start = bands["knowledge"][0] if bands else 14
    knowledge_end = bands["knowledge"][1] if bands else 27

    print(f"\nKnowledge layers: L{knowledge_start}-L{knowledge_end}")
    print(f"Hidden size: {vindex.hidden_size}")

    # Step 1: For each test triple, check what the vindex already knows
    print(f"\n=== Phase 1: Verify existing knowledge ===")
    existing_results = []

    for entity, relation, expected_target in TEST_TRIPLES:
        print(f"\n  {entity} --{relation}--> {expected_target} (expected)")

        # Embed entity and search knowledge layers
        found = False
        for layer in range(knowledge_start, knowledge_end + 1):
            hits = vindex.entity_knn(entity, layer=layer, top_k=10)
            for feat, score in hits:
                meta = vindex.feature_meta(layer, feat)
                if meta and meta.top_token.strip().lower() == expected_target.lower():
                    print(f"    FOUND: L{layer}:F{feat} token='{meta.top_token}' "
                          f"score={score:.1f} c={meta.c_score:.3f}")
                    existing_results.append({
                        "entity": entity,
                        "relation": relation,
                        "target": expected_target,
                        "found": True,
                        "layer": layer,
                        "feature": feat,
                        "gate_score": float(score),
                    })
                    found = True
                    break
            if found:
                break

        if not found:
            print(f"    NOT FOUND in knowledge layers")
            existing_results.append({
                "entity": entity,
                "relation": relation,
                "target": expected_target,
                "found": False,
            })

    found_count = sum(1 for r in existing_results if r["found"])
    print(f"\n  Found {found_count}/{len(TEST_TRIPLES)} test triples in vindex")

    # Step 2: Synthesise gate vectors from entity embeddings
    print(f"\n=== Phase 2: Gate vector synthesis ===")
    synthesis_results = []

    for entity, relation, target in TEST_TRIPLES:
        entity_embed = np.array(vindex.embed(entity))
        target_embed = np.array(vindex.embed(target))

        # The gate vector is what the feature "responds to" — entity embedding
        # The down vector is what the feature "outputs" — target embedding
        gate_vec = entity_embed  # normalised entity embedding
        down_vec = target_embed  # what this feature should output

        # Check: does gate_knn with this synthetic gate find the right target?
        # Simulate: search existing features at each knowledge layer
        # Compare entity embedding cosine with existing gate vectors
        best_match = None
        for layer in range(knowledge_start, knowledge_end + 1):
            hits = vindex.gate_knn(layer=layer, query_vector=gate_vec.tolist(), top_k=3)
            for feat, score in hits:
                meta = vindex.feature_meta(layer, feat)
                if meta:
                    cos_down = float(np.dot(down_vec, np.array(vindex.gate_vector(layer, feat))) /
                                    (np.linalg.norm(down_vec) * np.linalg.norm(np.array(vindex.gate_vector(layer, feat))) + 1e-8))
                    if best_match is None or score > best_match["gate_score"]:
                        best_match = {
                            "layer": layer,
                            "feature": feat,
                            "gate_score": float(score),
                            "matched_token": meta.top_token,
                            "cosine_with_target_embed": cos_down,
                        }

        if best_match:
            correct = best_match["matched_token"].strip().lower() == target.lower()
            print(f"  {entity} -> {target}: "
                  f"matched='{best_match['matched_token']}' "
                  f"score={best_match['gate_score']:.1f} "
                  f"{'CORRECT' if correct else 'WRONG'}")
        else:
            print(f"  {entity} -> {target}: no match found")
            correct = False

        synthesis_results.append({
            "entity": entity,
            "relation": relation,
            "expected_target": target,
            "best_match": best_match,
            "correct": correct,
        })

    correct_count = sum(1 for r in synthesis_results if r["correct"])
    print(f"\n  Synthesis accuracy: {correct_count}/{len(TEST_TRIPLES)} "
          f"({100*correct_count/len(TEST_TRIPLES):.0f}%)")

    # Save results
    results = {
        "vindex": str(vindex),
        "knowledge_layers": [knowledge_start, knowledge_end],
        "existing_knowledge": existing_results,
        "synthesis": synthesis_results,
        "existing_found": found_count,
        "synthesis_correct": correct_count,
        "total_triples": len(TEST_TRIPLES),
    }

    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "03_build_layer.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    vindex_path = sys.argv[1] if len(sys.argv) > 1 else "output/gemma3-4b-v2.vindex"
    run_experiment(vindex_path)
