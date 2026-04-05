use ndarray::Array2;

use crate::ffn::FfnBackend;
use crate::ffn::sparse_compute::sparse_ffn_forward;
use crate::model::ModelWeights;
use crate::graph_ffn::GateIndex;

/// Graph FFN backend: uses a precomputed gate index instead of the gate matmul.
///
/// Runtime: residual → embedding projection → token lookup → feature list → sparse up/down.
/// Eliminates the gate matmul (500ms → ~0.01ms for the lookup).
#[deprecated(note = "Research artifact — not scalable. Use WalkFfn.")]
pub struct GraphFfn<'a> {
    pub weights: &'a ModelWeights,
    pub gate_index: &'a GateIndex,
    /// Max features to use per position.
    pub top_k: usize,
}

impl<'a> FfnBackend for GraphFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        let (out, _) = self.forward_inner(layer, x);
        out
    }

    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        self.forward_inner(layer, x)
    }

    fn name(&self) -> &str {
        "graph"
    }
}

impl<'a> GraphFfn<'a> {
    fn forward_inner(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let arch = &*self.weights.arch;
        let w_up = self.weights.tensors.get(&arch.ffn_up_key(layer)).unwrap();
        let hidden = x.shape()[1];
        let intermediate = w_up.shape()[0];
        let seq_len = x.shape()[0];

        let mut full_activation = Array2::<f32>::zeros((seq_len, intermediate));
        let mut out = Array2::<f32>::zeros((seq_len, hidden));

        // Embedding projection for feature selection (BLAS matmul, not scalar loop)
        let embed_scale = self.weights.arch.embed_scale();
        let embed_proj = x.dot(&self.weights.embed.t()) * embed_scale;

        for s in 0..seq_len {
            // Step 1: find nearest tokens via embedding projection (already computed)
            let logits = embed_proj.row(s);
            let mut token_scores: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
            let n = self.gate_index.top_tokens.min(token_scores.len());
            if n < token_scores.len() {
                token_scores.select_nth_unstable_by(n, |a, b| b.1.partial_cmp(&a.1).unwrap());
                token_scores.truncate(n);
            }

            // Step 2: look up candidate features from index, dedup
            let features = self.gate_index.lookup_from_tokens(&token_scores, layer, self.top_k);
            if features.is_empty() {
                continue;
            }

            // Step 3: sparse FFN forward for this position
            let x_row = x.slice(ndarray::s![s..s + 1, ..]).to_owned();
            let (pos_out, pos_act) = sparse_ffn_forward(self.weights, layer, &x_row, &features);

            out.row_mut(s).assign(&pos_out.row(0));
            full_activation.row_mut(s).assign(&pos_act.row(0));
        }

        (out, full_activation)
    }
}
