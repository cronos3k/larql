#!/usr/bin/env python3
"""Report coverage of triples, probes, and labels.

Shows which relations have triples, which have probe results,
and overall coverage statistics.

Usage:
    python3 scripts/coverage_report.py
"""

import json
from pathlib import Path
from collections import Counter


def main() -> None:
    """Report coverage of triples, probes, and labels."""
    data_dir = Path(__file__).parent.parent / "data"
    probes_dir = Path(__file__).parent.parent / "probes"

    # Triples coverage
    print("=== Triples Coverage ===\n")
    triples_dir = data_dir / "triples"
    total_pairs = 0
    relations = []

    if triples_dir.exists():
        for f in sorted(triples_dir.glob("*.json")):
            with open(f) as fh:
                d = json.load(fh)
            n = len(d.get("pairs", []))
            total_pairs += n
            rel = d.get("relation", f.stem)
            relations.append((rel, n, f.name))

    print(f"  Relations: {len(relations)}")
    print(f"  Total pairs: {total_pairs:,}")
    print()
    print(f"  {'Relation':<25s} {'Pairs':>6s}  File")
    print(f"  {'-'*25} {'-'*6}  {'-'*20}")
    for rel, n, fname in sorted(relations, key=lambda x: -x[1]):
        print(f"  {rel:<25s} {n:>6d}  {fname}")

    # WordNet coverage
    print("\n=== WordNet Coverage ===\n")
    wordnet_path = data_dir / "wordnet_relations.json"
    wn = {}
    wn_total = 0
    if wordnet_path.exists():
        with open(wordnet_path) as f:
            wn = json.load(f)
        wn_total = sum(len(v["pairs"]) for v in wn.values())
        print(f"  Relations: {len(wn)}")
        print(f"  Total pairs: {wn_total:,}")
        for rel, data in sorted(wn.items(), key=lambda x: -len(x[1]["pairs"])):
            print(f"    {rel:<20s} {len(data['pairs']):>6d} pairs")
    else:
        print("  Not found. Run: python3 scripts/fetch_wordnet_relations.py")

    # AST coverage
    print("\n=== AST Coverage ===\n")
    ast_dir = data_dir / "ast"
    if ast_dir.exists():
        for f in sorted(ast_dir.glob("*.json")):
            with open(f) as fh:
                d = json.load(fh)
            n_rels = len(d.get("relations", {}))
            n_pairs = sum(len(r.get("pairs", [])) for r in d.get("relations", {}).values())
            print(f"  {f.stem:<25s} {n_rels:>3d} relations, {n_pairs:>5d} pairs")
    else:
        print("  No AST data yet. Run: python3 scripts/extract_ast_pairs.py")

    # Grammar coverage
    print("\n=== English Grammar Coverage ===\n")
    grammar_path = data_dir / "english_grammar.json"
    if grammar_path.exists():
        with open(grammar_path) as fh:
            gd = json.load(fh)
        g_rels = gd.get("relations", {})
        g_total = sum(len(r.get("pairs", [])) for r in g_rels.values())
        for cat, rel_data in sorted(g_rels.items()):
            n = len(rel_data.get("pairs", []))
            print(f"  {cat:<25s} {n:>5d} pairs")
        print(f"\n  Total: {g_total:,} pairs across {len(g_rels)} categories")
    else:
        print(
            "  No grammar data yet. Run:\n"
            "    python3 scripts/generate_grammar.py"
        )

    # Templates coverage
    print("\n=== Template Coverage ===\n")
    templates_path = data_dir / "probe_templates.json"
    templates = {}
    if templates_path.exists():
        with open(templates_path) as f:
            templates = json.load(f)
        total_templates = sum(len(v) for v in templates.values())
        print(f"  Relations with templates: {len(templates)}")
        print(f"  Total templates: {total_templates}")

        # Check which triple relations have templates
        triple_rels = set(r for r, _, _ in relations)
        template_rels = set(templates.keys())
        missing = triple_rels - template_rels
        if missing:
            print(f"  Missing templates for: {', '.join(sorted(missing))}")
    else:
        print("  Not found. Create: data/probe_templates.json")

    # Probe coverage
    print("\n=== Probe Coverage ===\n")
    if probes_dir.exists():
        for model_dir in sorted(probes_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            labels_path = model_dir / "feature_labels.json"
            meta_path = model_dir / "probe_meta.json"

            print(f"  Model: {model_dir.name}")
            if labels_path.exists():
                with open(labels_path) as f:
                    labels = json.load(f)
                rel_counts = Counter(labels.values())
                print(f"    Features labeled: {len(labels)}")
                print(f"    Relations found: {len(rel_counts)}")
                for rel, count in rel_counts.most_common():
                    print(f"      {rel:<20s} {count:>4d}")

            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                print(f"    Probes run: {meta.get('num_probes', '?')}")
                print(f"    Entities probed: {meta.get('num_entities', '?')}")
            print()
    else:
        print("  No probe results yet. Run: python3 scripts/probe_mlx.py")

    # Summary
    print("=== Summary ===\n")
    print(f"  Wikidata: {len(relations)} relations, {total_pairs:,} pairs")
    if wn:
        print(f"  WordNet: {len(wn)} relations, {wn_total:,} pairs")
    if templates:
        print(f"  Templates: {len(templates)} relations")
    probe_total = 0
    if probes_dir.exists():
        for model_dir in probes_dir.iterdir():
            lp = model_dir / "feature_labels.json"
            if lp.exists():
                with open(lp) as f:
                    probe_total += len(json.load(f))
    print(f"  Probe labels: {probe_total} features")


if __name__ == "__main__":
    main()
