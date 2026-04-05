use ndarray::Array2;

use crate::ffn::FfnBackend;
use crate::model::ModelWeights;
use crate::graph_ffn::GateIndex;

// ── Entity-routed FFN: preselect features once, reuse across all layers ──

/// Entity-routed FFN backend: resolves entity tokens once at construction,
/// then uses the gate index for O(1) feature lookup per layer.
/// Eliminates both the gate matmul AND per-layer embedding projection.
///
/// Flow:
/// 1. Construction: input embedding → top-N tokens (one-time embedding projection)
/// 2. Per-layer forward: token_ids → GateIndex hash lookup → feature_ids
/// 3. Gather gate+up rows for selected features, compute SiLU(gate)*up, sparse down
#[deprecated(note = "Research artifact — not scalable. Use WalkFfn.")]
pub struct EntityRoutedFfn<'a> {
    pub weights: &'a ModelWeights,
    pub gate_index: &'a GateIndex,
    /// Pre-resolved token IDs from input embedding.
    pub entity_tokens: Vec<(usize, f32)>,
    /// Max features per layer.
    pub top_k: usize,
}

impl<'a> EntityRoutedFfn<'a> {
    /// Create from a pre-FFN hidden state. Projects against embeddings once
    /// to identify entity tokens, which are reused for all layers.
    pub fn from_hidden(
        weights: &'a ModelWeights,
        gate_index: &'a GateIndex,
        hidden_state: &ndarray::Array1<f32>,
        top_k: usize,
    ) -> Self {
        let embed = &weights.embed;
        let embed_scale = weights.arch.embed_scale();
        let vocab_size = embed.shape()[0];

        // Single BLAS gemv: hidden_state @ embed.T → (vocab_size,)
        let logits = embed.dot(hidden_state) * embed_scale;

        let mut token_scores: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        let n = gate_index.top_tokens.min(vocab_size);
        if n < vocab_size {
            token_scores.select_nth_unstable_by(n, |a, b| b.1.partial_cmp(&a.1).unwrap());
            token_scores.truncate(n);
        }

        EntityRoutedFfn {
            weights,
            gate_index,
            entity_tokens: token_scores,
            top_k,
        }
    }

    /// Create directly from known token IDs (e.g., from input tokens).
    pub fn from_token_ids(
        weights: &'a ModelWeights,
        gate_index: &'a GateIndex,
        token_ids: &[u32],
        top_k: usize,
    ) -> Self {
        let entity_tokens: Vec<(usize, f32)> =
            token_ids.iter().map(|&t| (t as usize, 1.0)).collect();
        EntityRoutedFfn {
            weights,
            gate_index,
            entity_tokens,
            top_k,
        }
    }
}

impl<'a> FfnBackend for EntityRoutedFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        let (out, _) = self.forward_inner(layer, x);
        out
    }

    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        self.forward_inner(layer, x)
    }

    fn name(&self) -> &str {
        "entity-routed"
    }
}

impl<'a> EntityRoutedFfn<'a> {
    fn forward_inner(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        // Feature selection: hash lookup from pre-resolved entity tokens (no matmul)
        let features = self
            .gate_index
            .lookup_from_tokens(&self.entity_tokens, layer, self.top_k);

        // Architecture-correct sparse FFN on selected features
        crate::ffn::sparse_compute::sparse_ffn_forward(self.weights, layer, x, &features)
    }
}
