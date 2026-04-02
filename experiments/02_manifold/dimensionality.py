"""
Experiment 2: Dimensionality of the knowledge manifold.

Hypothesis: knowledge gate vectors live on a low-dimensional manifold (~9-15D).
If 99% of variance is captured in K dimensions, compress 71 GB to 71 * K/2560 GB.

Method: SVD of all gate vectors from knowledge layers (L14-27).
"""

import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import _larql_core as larql


def run_experiment(vindex_path: str, sample_per_layer: int = 0):
    print(f"Loading vindex from {vindex_path}...")
    vindex = larql.load_vindex(vindex_path)
    print(f"  {vindex}")

    bands = vindex.layer_bands()
    knowledge_start = bands["knowledge"][0] if bands else 14
    knowledge_end = bands["knowledge"][1] if bands else 27

    # Collect gate vectors from all knowledge layers
    print(f"\nCollecting gate vectors from L{knowledge_start}-L{knowledge_end}...")
    all_gates = []

    for layer in range(knowledge_start, knowledge_end + 1):
        n_feat = vindex.num_features(layer)
        if n_feat == 0:
            print(f"  L{layer}: no features, skipping")
            continue

        gates = np.array(vindex.gate_vectors(layer=layer))
        print(f"  L{layer}: {gates.shape[0]} vectors x {gates.shape[1]}D")

        if sample_per_layer > 0 and gates.shape[0] > sample_per_layer:
            idx = np.random.choice(gates.shape[0], sample_per_layer, replace=False)
            gates = gates[idx]

        all_gates.append(gates)

    if not all_gates:
        print("No gate vectors found. Exiting.")
        return

    X = np.vstack(all_gates)
    print(f"\nTotal: {X.shape[0]} vectors x {X.shape[1]}D")
    print(f"  Memory: {X.nbytes / 1e9:.2f} GB")

    # Centre the data
    mean = X.mean(axis=0)
    X_centred = X - mean

    # SVD (truncated for efficiency if very large)
    print("\nRunning SVD...")
    if X.shape[0] > 50000:
        # Use randomized SVD for large datasets
        from numpy.linalg import svd
        # Only compute top-K singular values
        k = min(256, X.shape[1])
        # For very large matrices, sample rows
        if X.shape[0] > 200000:
            idx = np.random.choice(X.shape[0], 200000, replace=False)
            X_sample = X_centred[idx]
            print(f"  Sampled {X_sample.shape[0]} vectors for SVD")
        else:
            X_sample = X_centred
        U, S, Vt = np.linalg.svd(X_sample, full_matrices=False)
    else:
        U, S, Vt = np.linalg.svd(X_centred, full_matrices=False)

    # Compute cumulative explained variance
    variance = S ** 2
    total_variance = variance.sum()
    cumulative = np.cumsum(variance) / total_variance

    # Find dimensionalities for different thresholds
    thresholds = [0.90, 0.95, 0.99, 0.999]
    dims = {}
    for t in thresholds:
        d = int(np.searchsorted(cumulative, t)) + 1
        dims[t] = d
        compression = vindex.hidden_size / d
        print(f"  {t*100:.1f}% variance: {d}D (compression {compression:.1f}x)")

    # Top singular values
    print(f"\nTop 20 singular values:")
    for i in range(min(20, len(S))):
        print(f"  S[{i}] = {S[i]:.2f}  (cumulative {cumulative[i]*100:.2f}%)")

    # Spectral gap analysis
    ratios = S[:-1] / S[1:]
    biggest_gap_idx = int(np.argmax(ratios[:50]))  # Look in first 50
    print(f"\nBiggest spectral gap: between S[{biggest_gap_idx}] and S[{biggest_gap_idx+1}]")
    print(f"  Ratio: {ratios[biggest_gap_idx]:.2f}")
    print(f"  Suggests intrinsic dimensionality ~{biggest_gap_idx + 1}")

    # Save results
    results = {
        "total_vectors": int(X.shape[0]),
        "hidden_size": int(X.shape[1]),
        "knowledge_layers": list(range(knowledge_start, knowledge_end + 1)),
        "thresholds": {str(k): v for k, v in dims.items()},
        "spectral_gap_dim": int(biggest_gap_idx + 1),
        "top_singular_values": S[:50].tolist(),
        "cumulative_variance": cumulative[:50].tolist(),
    }

    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "02_svd_spectrum.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Plot if matplotlib available
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Singular value spectrum
        axes[0].semilogy(S[:100], 'b-', linewidth=2)
        axes[0].set_xlabel("Component")
        axes[0].set_ylabel("Singular Value")
        axes[0].set_title("Singular Value Spectrum")
        axes[0].grid(True, alpha=0.3)

        # Cumulative variance
        axes[1].plot(cumulative[:100] * 100, 'r-', linewidth=2)
        for t in thresholds:
            d = dims[t]
            axes[1].axhline(y=t*100, color='gray', linestyle='--', alpha=0.5)
            axes[1].axvline(x=d, color='gray', linestyle='--', alpha=0.5)
            axes[1].annotate(f'{d}D', (d, t*100), textcoords="offset points",
                           xytext=(10, -10), fontsize=9)
        axes[1].set_xlabel("Number of Components")
        axes[1].set_ylabel("Cumulative Variance (%)")
        axes[1].set_title("Explained Variance")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(os.path.dirname(__file__), "..", "results", "02_svd_spectrum.png")
        plt.savefig(plot_path, dpi=150)
        print(f"Plot saved to {plot_path}")
        plt.close()
    except ImportError:
        print("(matplotlib not available — skipping plot)")


if __name__ == "__main__":
    vindex_path = sys.argv[1] if len(sys.argv) > 1 else "output/gemma3-4b-v2.vindex"
    sample = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    run_experiment(vindex_path, sample_per_layer=sample)
