//! WalkFfn — FFN backend that uses VectorIndex gate KNN for feature selection.
//!
//! Gate KNN from the (potentially patched) vindex selects which features fire.
//! The actual FFN computation uses the model's up/down weights for those features.
//! This means INSERT/DELETE/UPDATE to the vindex affect inference output.

use ndarray::Array2;

use crate::ffn::FfnBackend;
use crate::ffn::sparse_compute::sparse_ffn_forward;
use crate::model::ModelWeights;

use larql_vindex::{GateIndex, WalkHit, WalkTrace};

/// FFN backend that uses a GateIndex for gate selection.
///
/// The gate matmul IS the KNN. `residual × gate_vectors^T` is both the gate
/// computation and the similarity search. Same operation, different framing.
///
/// When the vindex has been patched (INSERT/DELETE/UPDATE), the KNN uses the
/// patched gate vectors. The selected features then go through the model's
/// actual up/down weights for the FFN computation.
pub struct WalkFfn<'a> {
    pub weights: &'a ModelWeights,
    pub index: &'a dyn GateIndex,
    pub top_k: usize,
    trace: std::cell::RefCell<Vec<(usize, Vec<WalkHit>)>>,
}

impl<'a> WalkFfn<'a> {
    pub fn new(weights: &'a ModelWeights, index: &'a dyn GateIndex, top_k: usize) -> Self {
        Self {
            weights,
            index,
            top_k,
            trace: std::cell::RefCell::new(Vec::new()),
        }
    }

    /// Take the accumulated walk trace (clears internal state).
    pub fn take_trace(&self) -> WalkTrace {
        let layers = self.trace.borrow_mut().drain(..).collect();
        WalkTrace { layers }
    }

    /// Gate KNN for a single position, capturing trace and returning feature indices.
    fn knn_select_and_trace(&self, layer: usize, x_row: &ndarray::ArrayView1<f32>) -> Vec<usize> {
        if self.index.num_features(layer) == 0 {
            return vec![];
        }

        let hits = self.index.gate_knn(layer, &x_row.to_owned(), self.top_k);

        // Capture trace (which features fired and what they mean)
        let walk_hits: Vec<WalkHit> = hits
            .iter()
            .filter_map(|&(feature, gate_score)| {
                let meta = self.index.feature_meta(layer, feature)?.clone();
                Some(WalkHit { layer, feature, gate_score, meta })
            })
            .collect();
        self.trace.borrow_mut().push((layer, walk_hits));

        // Return feature indices for sparse FFN computation
        hits.into_iter().map(|(f, _)| f).collect()
    }
}

impl<'a> FfnBackend for WalkFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        self.forward_with_activation(layer, x).0
    }

    fn forward_with_activation(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>) {
        let seq_len = x.shape()[0];

        // If vindex has features for this layer, use KNN-based sparse FFN
        if self.index.num_features(layer) > 0 {
            // Select features per position via vindex gate KNN, union them
            let mut all_features = std::collections::BTreeSet::new();
            for s in 0..seq_len {
                let x_row = x.row(s);
                let feats = self.knn_select_and_trace(layer, &x_row);
                all_features.extend(feats);
            }
            let features: Vec<usize> = all_features.into_iter().collect();

            // Sparse FFN: compute gate/up/down for selected features only
            sparse_ffn_forward(self.weights, layer, x, &features)
        } else {
            // No vindex data for this layer — fall back to dense
            let dense_ffn = crate::ffn::WeightFfn { weights: self.weights };
            dense_ffn.forward_with_activation(layer, x)
        }
    }

    fn name(&self) -> &str {
        "walk"
    }
}
