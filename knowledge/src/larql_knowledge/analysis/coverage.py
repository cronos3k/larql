"""Coverage reporting — show what's available across all data sources."""

import json
from pathlib import Path
from collections import Counter


def coverage_report(data_dir: Path | None = None, probes_dir: Path | None = None):
    """Print a coverage report of all data sources."""
    if data_dir is None:
        data_dir = Path("data")
    if probes_dir is None:
        probes_dir = Path("probes")

    _report_triples(data_dir / "triples")
    _report_wordnet(data_dir / "wordnet_relations.json")
    _report_ast(data_dir / "ast")
    _report_grammar(data_dir / "english_grammar.json")
    _report_templates(data_dir / "probe_templates.json")
    _report_probes(probes_dir)


def _report_triples(triples_dir: Path):
    print("=== Triples ===\n")
    if not triples_dir.exists():
        print("  Not found.")
        return

    total = 0
    for f in sorted(triples_dir.glob("*.json")):
        with open(f) as fh:
            d = json.load(fh)
        n = len(d.get("pairs", []))
        total += n
        print(f"  {d.get('relation', f.stem):<25s} {n:>5d} pairs")

    print(f"\n  Total: {total:,} pairs across {len(list(triples_dir.glob('*.json')))} relations\n")


def _report_wordnet(path: Path):
    print("=== WordNet ===\n")
    if not path.exists():
        print("  Not found. Run: larql-knowledge ingest-wordnet\n")
        return

    with open(path) as f:
        wn = json.load(f)
    total = sum(len(v["pairs"]) for v in wn.values())
    for rel, data in sorted(wn.items(), key=lambda x: -len(x[1]["pairs"])):
        print(f"  {rel:<20s} {len(data['pairs']):>5d} pairs")
    print(f"\n  Total: {total:,} pairs across {len(wn)} relations\n")


def _report_ast(ast_dir: Path):
    print("=== AST ===\n")
    if not ast_dir.exists() or not list(ast_dir.glob("*.json")):
        print("  No AST data yet.\n")
        return

    for f in sorted(ast_dir.glob("*.json")):
        with open(f) as fh:
            d = json.load(fh)
        n_rels = len(d.get("relations", {}))
        n_pairs = sum(len(r.get("pairs", [])) for r in d.get("relations", {}).values())
        print(f"  {f.stem:<25s} {n_rels:>3d} relations, {n_pairs:>5d} pairs")
    print()


def _report_grammar(path: Path):
    print("=== English Grammar ===\n")
    if not path.exists():
        print("  No grammar data yet.\n")
        return

    with open(path) as f:
        data = json.load(f)

    relations = data.get("relations", {})
    total = sum(len(r.get("pairs", [])) for r in relations.values())
    for cat, rel_data in sorted(relations.items()):
        n = len(rel_data.get("pairs", []))
        print(f"  {cat:<25s} {n:>5d} pairs")
    print(f"\n  Total: {total:,} pairs across {len(relations)} categories\n")


def _report_templates(path: Path):
    print("=== Templates ===\n")
    if not path.exists():
        print("  Not found.\n")
        return

    with open(path) as f:
        templates = json.load(f)
    total = sum(len(v) for v in templates.values())
    print(f"  {len(templates)} relations, {total} templates total\n")


def _report_probes(probes_dir: Path):
    print("=== Probes ===\n")
    if not probes_dir.exists():
        print("  No probes yet.\n")
        return

    for model_dir in sorted(probes_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        labels_path = model_dir / "feature_labels.json"
        if not labels_path.exists():
            continue

        with open(labels_path) as f:
            labels = json.load(f)
        # Handle both flat dict {"L27_F9515": "capital"} and
        # rich list [{"layer": 27, "feature": 9515, "relation": "capital"}]
        if isinstance(labels, dict):
            rel_counts = Counter(labels.values())
            num_labels = len(labels)
        elif isinstance(labels, list):
            rel_counts = Counter(entry.get("relation", "?") for entry in labels)
            num_labels = len(labels)
        else:
            continue

        print(f"  {model_dir.name}:")
        print(f"    {num_labels} features, {len(rel_counts)} relations")
        for rel, count in rel_counts.most_common(10):
            print(f"      {rel:<20s} {count:>4d}")

        meta_path = model_dir / "probe_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            print(f"    Probed: {meta.get('num_entities', '?')} entities, {meta.get('num_probes', '?')} passes")
        print()
