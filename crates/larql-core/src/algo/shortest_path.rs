//! Shortest path algorithms — Dijkstra and A*.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use crate::core::edge::Edge;
use crate::core::graph::Graph;

#[derive(Debug, Clone)]
struct State {
    cost: f64,
    node: String,
}

impl PartialEq for State {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost && self.node == other.node
    }
}
impl Eq for State {}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Result of a shortest path search.
#[derive(Debug)]
pub struct PathResult {
    pub found: bool,
    pub path: Vec<Edge>,
    pub cost: f64,
    pub nodes_explored: usize,
}

/// Default weight function: weight = 1.0 - confidence.
pub fn default_weight(edge: &Edge) -> f64 {
    1.0 - edge.confidence
}

/// Find the shortest path using Dijkstra's algorithm.
/// Weight function defaults to `1.0 - confidence`.
pub fn shortest_path(graph: &Graph, from: &str, to: &str) -> Option<(f64, Vec<Edge>)> {
    shortest_path_with_weight(graph, from, to, default_weight)
}

/// Dijkstra with a custom weight function.
pub fn shortest_path_with_weight(
    graph: &Graph,
    from: &str,
    to: &str,
    weight_fn: fn(&Edge) -> f64,
) -> Option<(f64, Vec<Edge>)> {
    let result = search_internal(graph, from, to, weight_fn, |_, _| 0.0);
    if result.found {
        Some((result.cost, result.path))
    } else {
        None
    }
}

/// A* search with a heuristic function.
///
/// The heuristic should estimate the remaining cost from a node to the target.
/// For admissible A*, the heuristic must never overestimate.
pub fn astar(
    graph: &Graph,
    from: &str,
    to: &str,
    weight_fn: fn(&Edge) -> f64,
    heuristic: fn(&str, &str) -> f64,
) -> PathResult {
    search_internal(graph, from, to, weight_fn, heuristic)
}

/// Unified Dijkstra/A* implementation. When heuristic always returns 0, this is Dijkstra.
fn search_internal(
    graph: &Graph,
    from: &str,
    to: &str,
    weight_fn: fn(&Edge) -> f64,
    heuristic: fn(&str, &str) -> f64,
) -> PathResult {
    let mut dist: HashMap<String, f64> = HashMap::new();
    let mut prev: HashMap<String, String> = HashMap::new();
    let mut heap = BinaryHeap::new();
    let mut nodes_explored = 0;

    dist.insert(from.to_string(), 0.0);
    heap.push(State {
        cost: heuristic(from, to),
        node: from.to_string(),
    });

    while let Some(State { cost: _, node }) = heap.pop() {
        nodes_explored += 1;

        if node == to {
            // Reconstruct path
            let mut path = Vec::new();
            let mut current = to.to_string();
            while let Some(prev_node) = prev.get(&current) {
                let edges = graph.select(prev_node, None);
                if let Some(edge) = edges.iter().find(|e| e.object == current) {
                    path.push((*edge).clone());
                }
                current = prev_node.clone();
            }
            path.reverse();
            return PathResult {
                found: true,
                path,
                cost: dist[to],
                nodes_explored,
            };
        }

        let node_dist = *dist.get(&node).unwrap_or(&f64::INFINITY);

        for edge in graph.select(&node, None) {
            let weight = weight_fn(edge);
            let next_cost = node_dist + weight;

            if next_cost < *dist.get(&edge.object).unwrap_or(&f64::INFINITY) {
                dist.insert(edge.object.clone(), next_cost);
                prev.insert(edge.object.clone(), node.clone());
                heap.push(State {
                    cost: next_cost + heuristic(&edge.object, to),
                    node: edge.object.clone(),
                });
            }
        }
    }

    PathResult {
        found: false,
        path: Vec::new(),
        cost: f64::INFINITY,
        nodes_explored,
    }
}
