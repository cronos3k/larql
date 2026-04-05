//! Multi-hop walk strategies.
//!
//! `Graph::walk()` follows a sequence of relations, picking the highest-confidence
//! edge at each hop. This module provides alternative strategies.

use crate::core::edge::Edge;
use crate::core::graph::Graph;

/// Walk result with full path info.
pub struct WalkResult {
    pub destination: String,
    pub path: Vec<Edge>,
    pub min_confidence: f64,
}

/// Walk following relations, collecting ALL paths (not just highest confidence).
/// Returns paths sorted by minimum confidence along the path (best first).
pub fn walk_all_paths(
    graph: &Graph,
    subject: &str,
    relations: &[&str],
    max_paths: usize,
) -> Vec<WalkResult> {
    let mut results = Vec::new();
    walk_recursive(graph, subject, relations, 0, &mut Vec::new(), &mut results, max_paths * 10);

    results.sort_by(|a, b| b.min_confidence.partial_cmp(&a.min_confidence).unwrap());
    results.truncate(max_paths);
    results
}

fn walk_recursive(
    graph: &Graph,
    current: &str,
    relations: &[&str],
    depth: usize,
    path: &mut Vec<Edge>,
    results: &mut Vec<WalkResult>,
    limit: usize,
) {
    if depth >= relations.len() {
        let min_conf = path.iter().map(|e| e.confidence).fold(f64::INFINITY, f64::min);
        results.push(WalkResult {
            destination: current.to_string(),
            path: path.clone(),
            min_confidence: if path.is_empty() { 0.0 } else { min_conf },
        });
        return;
    }
    if results.len() >= limit { return; }

    let edges = graph.select(current, Some(relations[depth]));
    for edge in edges {
        path.push(edge.clone());
        walk_recursive(graph, &edge.object, relations, depth + 1, path, results, limit);
        path.pop();
    }
}
