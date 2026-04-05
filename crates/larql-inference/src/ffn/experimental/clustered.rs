use std::collections::HashMap;

use ndarray::Array2;

use crate::ffn::FfnBackend;
use crate::model::ModelWeights;

// ── Clustered gate index: hierarchical two-level feature selection ──

struct LayerClusters {
    centroids: ndarray::Array2<f32>,
    members: Vec<Vec<usize>>,
}

/// Clustered gate index: K-means on gate vectors per layer.
#[deprecated(note = "Research artifact — 0% accuracy. Use WalkFfn.")]
pub struct ClusteredGateIndex {
    layers: HashMap<usize, LayerClusters>,
    pub num_clusters: usize,
    pub top_c: usize,
}

impl ClusteredGateIndex {
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
            let gate_key = weights.arch.ffn_gate_key(layer);
            let w_gate = match weights.tensors.get(&gate_key) {
                Some(w) => w,
                None => continue,
            };
            layer_map.insert(layer, Self::kmeans(w_gate, num_clusters, kmeans_iters));
        }
        ClusteredGateIndex { layers: layer_map, num_clusters, top_c }
    }

    fn kmeans(w_gate: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>, k: usize, iters: usize) -> LayerClusters {
        let n = w_gate.shape()[0];
        let d = w_gate.shape()[1];
        let k = k.min(n);
        let mut centroids = ndarray::Array2::<f32>::zeros((k, d));
        for c in 0..k { centroids.row_mut(c).assign(&w_gate.row(c * n / k)); }
        let mut assignments = vec![0usize; n];
        for _iter in 0..iters {
            let scores = w_gate.dot(&centroids.t());
            for i in 0..n {
                let row = scores.row(i);
                let (best_c, _) = row.iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
                assignments[i] = best_c;
            }
            let mut sums = ndarray::Array2::<f32>::zeros((k, d));
            let mut counts = vec![0usize; k];
            for i in 0..n {
                let c = assignments[i];
                counts[c] += 1;
                for j in 0..d { sums[[c, j]] += w_gate[[i, j]]; }
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
        LayerClusters { centroids, members }
    }

    pub fn lookup(&self, layer: usize, residual: &ndarray::ArrayView1<f32>, top_k: usize) -> Vec<usize> {
        let lc = match self.layers.get(&layer) { Some(lc) => lc, None => return vec![] };
        let scores = lc.centroids.dot(residual);
        let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        let c = self.top_c.min(indexed.len());
        if c < indexed.len() {
            indexed.select_nth_unstable_by(c, |a, b| b.1.partial_cmp(&a.1).unwrap());
            indexed.truncate(c);
        }
        let mut features: Vec<usize> = Vec::new();
        for &(cid, _) in &indexed { features.extend_from_slice(&lc.members[cid]); }
        features.sort_unstable();
        features.dedup();
        features.truncate(top_k);
        features
    }

    pub fn num_layers(&self) -> usize { self.layers.len() }
    pub fn avg_cluster_size(&self) -> f64 {
        let (mut t, mut c) = (0usize, 0usize);
        for lc in self.layers.values() { for m in &lc.members { t += m.len(); c += 1; } }
        if c > 0 { t as f64 / c as f64 } else { 0.0 }
    }
}

/// Clustered FFN backend.
#[deprecated(note = "Research artifact — 0% accuracy. Use WalkFfn.")]
pub struct ClusteredFfn<'a> {
    pub weights: &'a ModelWeights,
    pub cluster_index: &'a ClusteredGateIndex,
    pub top_k: usize,
}

impl<'a> FfnBackend for ClusteredFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> { self.forward_inner(layer, x).0 }
    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) { self.forward_inner(layer, x) }
    fn name(&self) -> &str { "clustered" }
}

impl<'a> ClusteredFfn<'a> {
    fn forward_inner(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let seq_len = x.shape()[0];
        let hidden = x.shape()[1];
        let intermediate = self.weights.tensors.get(&self.weights.arch.ffn_gate_key(layer))
            .unwrap().shape()[0];

        // Per-position feature selection via cluster lookup, then sparse FFN
        let mut out = ndarray::Array2::<f32>::zeros((seq_len, hidden));
        let mut full_act = ndarray::Array2::<f32>::zeros((seq_len, intermediate));

        for s in 0..seq_len {
            let x_row = x.row(s);
            let features = self.cluster_index.lookup(layer, &x_row, self.top_k);
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
