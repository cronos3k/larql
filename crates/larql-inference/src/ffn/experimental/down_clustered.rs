use std::collections::HashMap;

use ndarray::Array2;

use crate::ffn::FfnBackend;
use crate::model::ModelWeights;

// ── Down-clustered FFN: select features by output direction, not gate scan ──

/// Per-layer down clusters: centroids of down-projection columns.
struct DownClusters {
    /// Centroid vectors: (num_clusters, hidden_size) — average down direction per cluster.
    centroids: ndarray::Array2<f32>,
    /// members[c] = feature indices whose down vectors belong to cluster c.
    members: Vec<Vec<usize>>,
}

/// Down-clustered gate index: features grouped by what they OUTPUT.
/// Runtime: residual → nearest down centroids → candidate features → sparse gate/up/down.
#[deprecated(note = "Research artifact — not scalable. Use WalkFfn.")]
pub struct DownClusteredIndex {
    layers: HashMap<usize, DownClusters>,
    pub num_clusters: usize,
    pub top_c: usize,
}

impl DownClusteredIndex {
    /// Build by clustering the columns of w_down at each layer.
    pub fn build(
        weights: &ModelWeights,
        layers: &[usize],
        num_clusters: usize,
        top_c: usize,
        kmeans_iters: usize,
        mut on_layer: impl FnMut(usize, usize),
    ) -> Self {
        let mut layer_map = HashMap::new();
        let total = layers.len();
        for (idx, &layer) in layers.iter().enumerate() {
            on_layer(idx, total);
            let arch = &*weights.arch;
            let w_down = match weights.tensors.get(&arch.ffn_down_key(layer)) {
                Some(w) => w,
                None => continue,
            };
            // w_down is (hidden, intermediate). We need to cluster by columns (features).
            // Transpose to (intermediate, hidden) so each row is a feature's down vector.
            let down_t = w_down.t().to_owned();
            layer_map.insert(layer, Self::kmeans(&down_t, num_clusters, kmeans_iters));
        }
        DownClusteredIndex { layers: layer_map, num_clusters, top_c }
    }

    fn kmeans(features: &ndarray::Array2<f32>, k: usize, iters: usize) -> DownClusters {
        let n = features.shape()[0];
        let d = features.shape()[1];
        let k = k.min(n);

        let mut centroids = ndarray::Array2::<f32>::zeros((k, d));
        for c in 0..k { centroids.row_mut(c).assign(&features.row(c * n / k)); }

        let mut assignments = vec![0usize; n];
        for _iter in 0..iters {
            let scores = features.dot(&centroids.t());
            for i in 0..n {
                let row = scores.row(i);
                let (best, _) = row.iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
                assignments[i] = best;
            }
            let mut sums = ndarray::Array2::<f32>::zeros((k, d));
            let mut counts = vec![0usize; k];
            for i in 0..n {
                let c = assignments[i];
                counts[c] += 1;
                for j in 0..d { sums[[c, j]] += features[[i, j]]; }
            }
            for c in 0..k {
                if counts[c] > 0 {
                    let cnt = counts[c] as f32;
                    for j in 0..d { centroids[[c, j]] = sums[[c, j]] / cnt; }
                }
                let norm: f32 = centroids.row(c).iter().map(|v| v * v).sum::<f32>().sqrt();
                if norm > 1e-12 { for j in 0..d { centroids[[c, j]] /= norm; } }
            }
        }

        let mut members = vec![Vec::new(); k];
        for i in 0..n { members[assignments[i]].push(i); }
        DownClusters { centroids, members }
    }

    /// Look up features whose down vectors point in the residual's direction.
    pub fn lookup(&self, layer: usize, residual: &ndarray::ArrayView1<f32>) -> Vec<usize> {
        let dc = match self.layers.get(&layer) { Some(dc) => dc, None => return vec![] };
        let scores = dc.centroids.dot(residual);
        let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        let c = self.top_c.min(indexed.len());
        if c < indexed.len() {
            indexed.select_nth_unstable_by(c, |a, b| b.1.partial_cmp(&a.1).unwrap());
            indexed.truncate(c);
        }
        let mut features = Vec::new();
        for &(cid, _) in &indexed { features.extend_from_slice(&dc.members[cid]); }
        features.sort_unstable();
        features.dedup();
        features
    }

    pub fn num_layers(&self) -> usize { self.layers.len() }
    pub fn avg_cluster_size(&self) -> f64 {
        let (mut t, mut c) = (0usize, 0usize);
        for dc in self.layers.values() { for m in &dc.members { t += m.len(); c += 1; } }
        if c > 0 { t as f64 / c as f64 } else { 0.0 }
    }
}

/// Down-clustered FFN backend: selects features by output direction, then computes
/// actual gate/up/down for those features only. No gate scan.
#[deprecated(note = "Research artifact — not scalable. Use WalkFfn.")]
pub struct DownClusteredFfn<'a> {
    pub weights: &'a ModelWeights,
    pub down_index: &'a DownClusteredIndex,
}

impl<'a> FfnBackend for DownClusteredFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> { self.forward_inner(layer, x).0 }
    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) { self.forward_inner(layer, x) }
    fn name(&self) -> &str { "down-clustered" }
}

impl<'a> DownClusteredFfn<'a> {
    fn forward_inner(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let seq_len = x.shape()[0];
        let hidden = x.shape()[1];
        let intermediate = self.weights.tensors.get(&self.weights.arch.ffn_gate_key(layer))
            .unwrap().shape()[0];

        let mut out = ndarray::Array2::<f32>::zeros((seq_len, hidden));
        let mut full_act = ndarray::Array2::<f32>::zeros((seq_len, intermediate));

        for s in 0..seq_len {
            let x_row = x.row(s);
            let features = self.down_index.lookup(layer, &x_row);
            if features.is_empty() { continue; }

            let x_slice = x.slice(ndarray::s![s..s+1, ..]).to_owned();
            let (pos_out, pos_act) = crate::ffn::sparse_compute::sparse_ffn_forward(
                self.weights, layer, &x_slice, &features);
            out.row_mut(s).assign(&pos_out.row(0));
            full_act.row_mut(s).assign(&pos_act.row(0));
        }
        (out, full_act)
    }
}
