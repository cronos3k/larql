use super::provider::{ModelProvider, ProviderError};

/// Result of chaining multiple forward passes.
#[derive(Debug, Clone)]
pub struct ChainResult {
    pub answer: String,
    pub tokens: Vec<String>,
    pub probabilities: Vec<f64>,
    pub num_passes: usize,
}

impl ChainResult {
    pub fn avg_probability(&self) -> f64 {
        if self.probabilities.is_empty() {
            return 0.0;
        }
        self.probabilities.iter().sum::<f64>() / self.probabilities.len() as f64
    }

    pub fn min_probability(&self) -> f64 {
        self.probabilities
            .iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0)
    }
}

const DEFAULT_STOP: &[char] = &['.', '\n', ',', ';', '!', '?', '(', ')', '[', ']', '<', '>'];

/// Chain forward passes to assemble a multi-token answer.
pub fn chain_tokens(
    provider: &dyn ModelProvider,
    prompt: &str,
    max_tokens: usize,
    min_probability: f64,
    stop_tokens: Option<&[char]>,
) -> Result<ChainResult, ProviderError> {
    let stops = stop_tokens.unwrap_or(DEFAULT_STOP);
    let mut tokens: Vec<String> = Vec::new();
    let mut probs: Vec<f64> = Vec::new();
    let mut current_prompt = prompt.to_string();

    for _ in 0..max_tokens {
        let result = provider.predict_next_token(&current_prompt, 1)?;
        let Some(top) = result.top() else { break };

        let stripped = top.token.trim();
        if stripped.is_empty() {
            break;
        }
        if top.probability < min_probability {
            break;
        }
        if stripped.chars().any(|c| stops.contains(&c)) {
            break;
        }

        tokens.push(top.token.clone());
        probs.push(top.probability);
        current_prompt.push_str(&top.token);
    }

    let answer = tokens.join("").trim().to_string();
    let num_passes = tokens.len();

    Ok(ChainResult {
        answer,
        tokens,
        probabilities: probs,
        num_passes,
    })
}
