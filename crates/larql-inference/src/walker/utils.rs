//! Shared utilities for walker modules.

use super::weight_walker::ThresholdCounts;

/// Decode a single token ID to a trimmed string.
pub fn decode_token(tokenizer: &tokenizers::Tokenizer, id: u32) -> Option<String> {
    crate::tokenizer::decode_token(tokenizer, id)
}

/// Round to 4 decimal places.
pub fn round4(v: f64) -> f64 {
    (v * 10000.0).round() / 10000.0
}

/// Extract top-N entities by count, with average confidence.
pub fn top_entities(
    counts: &std::collections::HashMap<String, (usize, f64)>,
    n: usize,
) -> Vec<(String, usize, f64)> {
    let mut sorted: Vec<_> = counts
        .iter()
        .map(|(name, (count, sum_conf))| (name.clone(), *count, sum_conf / *count as f64))
        .collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    sorted.truncate(n);
    sorted
}

/// Increment threshold counters for a normalized score.
pub fn count_threshold(t: &mut ThresholdCounts, v: f64) {
    if v >= 0.01 {
        t.t_01 += 1;
    }
    if v >= 0.05 {
        t.t_05 += 1;
    }
    if v >= 0.10 {
        t.t_10 += 1;
    }
    if v >= 0.25 {
        t.t_25 += 1;
    }
    if v >= 0.50 {
        t.t_50 += 1;
    }
    if v >= 0.75 {
        t.t_75 += 1;
    }
    if v >= 0.90 {
        t.t_90 += 1;
    }
}

/// Approximate current date without a chrono dependency.
pub fn current_date() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let days = now / 86400;
    let year = 1970 + (days / 365);
    let remaining = days % 365;
    let month = remaining / 30 + 1;
    let day = remaining % 30 + 1;
    format!("{year}-{month:02}-{day:02}")
}

// ── Top-K utilities ──

/// Top-k (index, value) from a flat slice using partial sort.
pub fn partial_top_k(data: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = data.iter().copied().enumerate().collect();
    let k = k.min(indexed.len());
    if k == 0 {
        return vec![];
    }
    if k >= indexed.len() {
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        return indexed;
    }
    indexed.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.truncate(k);
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed
}

/// Top-k from a matrix column.
pub fn partial_top_k_column(
    matrix: &ndarray::Array2<f32>,
    col: usize,
    k: usize,
) -> Vec<(usize, f32)> {
    let nrows = matrix.shape()[0];
    let mut indexed: Vec<(usize, f32)> = Vec::with_capacity(nrows);
    for i in 0..nrows {
        indexed.push((i, matrix[[i, col]]));
    }

    let k = k.min(indexed.len());
    if k == 0 {
        return vec![];
    }
    if k >= indexed.len() {
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        return indexed;
    }
    indexed.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.truncate(k);
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed
}
