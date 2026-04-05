//! Connected components analysis.
//!
//! Basic component counting is on `Graph::stats()`. This module provides
//! richer analysis: enumerate components, find largest, check connectivity.

use crate::core::graph::Graph;

/// Enumerate all connected components as sets of node names.
/// Treats the graph as undirected (follows both adjacency and reverse edges).
pub fn connected_components(graph: &Graph) -> Vec<Vec<String>> {
    let mut visited = std::collections::HashSet::new();
    let mut components = Vec::new();

    // Collect all node names
    let mut all_nodes = std::collections::HashSet::new();
    for edge in graph.edges() {
        all_nodes.insert(edge.subject.clone());
        all_nodes.insert(edge.object.clone());
    }

    for node in &all_nodes {
        if visited.contains(node) { continue; }

        // BFS from this node
        let mut component = Vec::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(node.clone());
        visited.insert(node.clone());

        while let Some(current) = queue.pop_front() {
            component.push(current.clone());

            // Forward edges
            for edge in graph.select(&current, None) {
                if !visited.contains(&edge.object) {
                    visited.insert(edge.object.clone());
                    queue.push_back(edge.object.clone());
                }
            }
            // Reverse edges
            for edge in graph.select_reverse(&current, None) {
                if !visited.contains(&edge.subject) {
                    visited.insert(edge.subject.clone());
                    queue.push_back(edge.subject.clone());
                }
            }
        }

        component.sort();
        components.push(component);
    }

    components.sort_by_key(|c| std::cmp::Reverse(c.len()));
    components
}

/// Check if two nodes are in the same connected component.
pub fn are_connected(graph: &Graph, a: &str, b: &str) -> bool {
    let mut visited = std::collections::HashSet::new();
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(a.to_string());
    visited.insert(a.to_string());

    while let Some(current) = queue.pop_front() {
        if current == b { return true; }
        for edge in graph.select(&current, None) {
            if !visited.contains(&edge.object) {
                visited.insert(edge.object.clone());
                queue.push_back(edge.object.clone());
            }
        }
        for edge in graph.select_reverse(&current, None) {
            if !visited.contains(&edge.subject) {
                visited.insert(edge.subject.clone());
                queue.push_back(edge.subject.clone());
            }
        }
    }
    false
}
