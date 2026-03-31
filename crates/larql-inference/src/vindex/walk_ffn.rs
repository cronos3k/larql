//! WalkFfn — FFN backend that uses VectorIndex for gate selection + interpretability.

use ndarray::Array2;

use crate::ffn::FfnBackend;
use crate::model::ModelWeights;

use larql_vindex::{VectorIndex, WalkHit, WalkTrace};

/// FFN backend that uses the VectorIndex for gate selection.
///
/// Gate KNN finds which features fire. Then uses the model's actual up/down
/// weights for the sparse computation — same as SparseFfn but with KNN-based
/// feature selection instead of full gate matmul.
///
/// The gate matmul IS the KNN. residual × gate_vectors^T is both the gate
/// computation and the similarity search. Same operation, different framing.
pub struct WalkFfn<'a> {
    pub weights: &'a ModelWeights,
    pub index: &'a VectorIndex,
    pub top_k: usize,
    /// If set, captures walk traces per layer during forward pass.
    trace: std::cell::RefCell<Vec<(usize, Vec<WalkHit>)>>,
}

impl<'a> WalkFfn<'a> {
    pub fn new(weights: &'a ModelWeights, index: &'a VectorIndex, top_k: usize) -> Self {
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

    /// Capture walk trace for the last position using gate KNN from the vindex.
    /// This is the interpretability layer — it records which features activate
    /// and what they mean (via down_meta labels). Does not affect computation.
    fn capture_trace(&self, layer: usize, x: &Array2<f32>) {
        let has_index = self.index.num_features(layer) > 0;
        if !has_index {
            return;
        }

        let seq_len = x.shape()[0];
        let last_row = x.row(seq_len - 1).to_owned();

        // Use vindex gate vectors for KNN (interpretability — which features match)
        let hits = self.index.gate_knn(layer, &last_row, self.top_k);

        let walk_hits: Vec<WalkHit> = hits
            .iter()
            .filter_map(|&(feature, gate_score)| {
                let meta = self.index.feature_meta(layer, feature)?.clone();
                Some(WalkHit {
                    layer,
                    feature,
                    gate_score,
                    meta,
                })
            })
            .collect();

        self.trace.borrow_mut().push((layer, walk_hits));
    }
}

impl<'a> FfnBackend for WalkFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        // Delegate to WeightFfn for exact architecture-correct computation
        let dense_ffn = crate::ffn::WeightFfn { weights: self.weights };
        let out = dense_ffn.forward(layer, x);

        // Capture walk trace for the last position (interpretability layer only)
        self.capture_trace(layer, x);

        out
    }

    fn forward_with_activation(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>) {
        let dense_ffn = crate::ffn::WeightFfn { weights: self.weights };
        let result = dense_ffn.forward_with_activation(layer, x);

        self.capture_trace(layer, x);

        result
    }

    fn name(&self) -> &str {
        "walk"
    }
}
