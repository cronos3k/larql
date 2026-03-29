//! Feed-forward network computation — SiLU gate, up projection, down projection.

use ndarray::Array2;

/// SiLU(gate) * up — the gated FFN activation used in Gemma/Llama.
pub fn silu_gate_up(gate: &Array2<f32>, up: &Array2<f32>) -> Array2<f32> {
    let activated = gate.mapv(|v| v * sigmoid(v));
    &activated * up
}

/// Full FFN forward: SiLU(x @ gate.T) * (x @ up.T) @ down.T
pub fn ffn_forward(
    x: &Array2<f32>,
    w_gate: &Array2<f32>,
    w_up: &Array2<f32>,
    w_down: &Array2<f32>,
) -> Array2<f32> {
    let gate = x.dot(&w_gate.t());
    let up = x.dot(&w_up.t());
    let activation = silu_gate_up(&gate, &up);
    activation.dot(&w_down.t())
}

/// Full FFN forward, also returning the pre-down activation for capture.
pub fn ffn_forward_with_activation(
    x: &Array2<f32>,
    w_gate: &Array2<f32>,
    w_up: &Array2<f32>,
    w_down: &Array2<f32>,
) -> (Array2<f32>, Array2<f32>) {
    let gate = x.dot(&w_gate.t());
    let up = x.dot(&w_up.t());
    let activation = silu_gate_up(&gate, &up);
    let out = activation.dot(&w_down.t());
    (out, activation)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
