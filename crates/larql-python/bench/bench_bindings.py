"""
Benchmark larql Python bindings.

Measures load time, embed, KNN, walk, describe, gate_vectors, and MLX load.

Usage:
    python bench/bench_bindings.py [path/to/model.vindex]
"""

import sys
import os
import time
import numpy as np

VINDEX_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "output", "gemma3-4b-v2.vindex"
)


def bench(name, fn, n=10):
    """Run fn n times, report min/mean."""
    # Warmup
    fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    mean = sum(times) / len(times)
    mn = min(times)
    print(f"  {name:<30} min={mn*1000:>8.2f}ms  mean={mean*1000:>8.2f}ms")
    return mean


def main():
    import larql

    print(f"Benchmarking: {VINDEX_PATH}")
    print()

    # Load
    t0 = time.perf_counter()
    vindex = larql.load(VINDEX_PATH)
    load_time = time.perf_counter() - t0
    print(f"  {'load()':<30} {load_time*1000:>8.1f}ms")
    print(f"  {vindex}")
    print()

    bands = vindex.layer_bands()
    last_knowledge = bands["knowledge"][1]

    # Embed
    bench("embed('France')", lambda: vindex.embed("France"))
    bench("embed('John Coyle')", lambda: vindex.embed("John Coyle"))

    # Tokenize
    bench("tokenize('hello world')", lambda: vindex.tokenize("hello world"))

    # Gate vector (single)
    bench("gate_vector(L27, F0)", lambda: vindex.gate_vector(last_knowledge, 0))

    # Gate vectors (full layer)
    bench("gate_vectors(L27)", lambda: vindex.gate_vectors(last_knowledge), n=3)

    # KNN
    embed = vindex.embed("France")
    bench("gate_knn(L27, top=10)", lambda: vindex.gate_knn(last_knowledge, embed.tolist(), 10))
    bench("entity_knn('France', L27)", lambda: vindex.entity_knn("France", last_knowledge, 10))

    # Walk
    layers = list(range(bands["knowledge"][0], bands["knowledge"][1] + 1))
    bench("entity_walk('France', knowledge)", lambda: vindex.entity_walk("France", layers, 5))

    # Describe
    bench("describe('France')", lambda: vindex.describe("France"), n=5)
    bench("describe('France', verbose)", lambda: vindex.describe("France", verbose=True), n=5)

    # Feature metadata
    bench("feature_meta(L27, F0)", lambda: vindex.feature_meta(last_knowledge, 0))
    bench("feature_label(L27, F0)", lambda: vindex.feature_label(last_knowledge, 0))

    # Relations
    bench("relations()", lambda: vindex.relations(), n=5)

    # Cluster centre
    bench("cluster_centre('capital')", lambda: vindex.cluster_centre("capital"))

    # Stats
    bench("stats()", lambda: vindex.stats())

    # Insert
    bench("insert()", lambda: vindex.insert("BenchEntity", "capital", "BenchCity"), n=5)

    # MLX load
    print()
    try:
        t0 = time.perf_counter()
        import larql.mlx
        model, tokenizer = larql.mlx.load(VINDEX_PATH)
        mlx_time = time.perf_counter() - t0
        print(f"  {'larql.mlx.load()':<30} {mlx_time*1000:>8.1f}ms")

        import mlx_lm
        t0 = time.perf_counter()
        model2, tokenizer2 = mlx_lm.load(vindex.model)
        native_time = time.perf_counter() - t0
        print(f"  {'mlx_lm.load() (native)':<30} {native_time*1000:>8.1f}ms")
        print(f"  {'ratio':<30} {mlx_time/native_time:.1f}x")
    except ImportError:
        print("  (mlx not installed — skipping MLX benchmarks)")


if __name__ == "__main__":
    main()
