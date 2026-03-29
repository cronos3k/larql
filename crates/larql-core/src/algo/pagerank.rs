//! PageRank — iterative importance ranking for graph entities.

use std::collections::HashMap;

use crate::core::graph::Graph;

/// PageRank result.
#[derive(Debug)]
pub struct PageRankResult {
    pub ranks: HashMap<String, f64>,
    pub iterations: usize,
    pub converged: bool,
}

impl PageRankResult {
    /// Top-k entities by rank.
    pub fn top_k(&self, k: usize) -> Vec<(&str, f64)> {
        let mut sorted: Vec<_> = self.ranks.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        sorted.truncate(k);
        sorted
    }
}

/// Compute PageRank over the graph.
///
/// - `damping`: probability of following an edge (typically 0.85)
/// - `max_iterations`: convergence limit
/// - `tolerance`: convergence threshold (sum of rank changes)
pub fn pagerank(
    graph: &Graph,
    damping: f64,
    max_iterations: usize,
    tolerance: f64,
) -> PageRankResult {
    let entities = graph.list_entities();
    let n = entities.len();
    if n == 0 {
        return PageRankResult {
            ranks: HashMap::new(),
            iterations: 0,
            converged: true,
        };
    }

    let init = 1.0 / n as f64;
    let mut ranks: HashMap<String, f64> = entities.iter().map(|e| (e.clone(), init)).collect();

    // Precompute out-degree
    let out_degree: HashMap<&str, usize> = entities
        .iter()
        .map(|e| (e.as_str(), graph.select(e, None).len()))
        .collect();

    let mut converged = false;
    let mut iterations = 0;

    for _ in 0..max_iterations {
        iterations += 1;
        let mut new_ranks: HashMap<String, f64> = entities
            .iter()
            .map(|e| (e.clone(), (1.0 - damping) / n as f64))
            .collect();

        for entity in &entities {
            let rank = ranks[entity];
            let degree = out_degree[entity.as_str()];
            if degree == 0 {
                // Dangling node — distribute evenly
                let share = rank * damping / n as f64;
                for nr in new_ranks.values_mut() {
                    *nr += share;
                }
            } else {
                let share = rank * damping / degree as f64;
                for edge in graph.select(entity, None) {
                    *new_ranks.entry(edge.object.clone()).or_insert(0.0) += share;
                }
            }
        }

        // Check convergence
        let delta: f64 = entities
            .iter()
            .map(|e| (ranks[e] - new_ranks[e]).abs())
            .sum();

        ranks = new_ranks;

        if delta < tolerance {
            converged = true;
            break;
        }
    }

    PageRankResult {
        ranks,
        iterations,
        converged,
    }
}
