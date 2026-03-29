use std::collections::{HashSet, VecDeque};

use crate::core::edge::Edge;
use crate::core::enums::SourceType;
use crate::core::graph::Graph;

use super::chain::chain_tokens;
use super::provider::ModelProvider;
use super::templates::TemplateRegistry;

/// BFS extraction configuration.
pub struct BfsConfig {
    pub max_depth: u32,
    pub max_entities: usize,
    pub min_confidence: f64,
    pub max_chain_tokens: usize,
}

impl Default for BfsConfig {
    fn default() -> Self {
        Self {
            max_depth: 3,
            max_entities: 1000,
            min_confidence: 0.3,
            max_chain_tokens: 5,
        }
    }
}

/// Callbacks for progress reporting.
pub trait BfsCallbacks {
    fn on_entity(&mut self, _entity: &str, _depth: u32, _visited: usize, _queue: usize) {}
    fn on_edge(&mut self, _edge: &Edge, _depth: u32) {}
    fn on_checkpoint(&mut self, _graph: &Graph) {}
}

/// No-op callbacks for headless extraction.
pub struct SilentCallbacks;
impl BfsCallbacks for SilentCallbacks {}

/// Run BFS extraction from seed entities.
///
/// For each entity: probe all templates, cache results as edges,
/// queue valid entities discovered in answers. Checkpoint after
/// each entity.
pub fn extract_bfs(
    provider: &dyn ModelProvider,
    templates: &TemplateRegistry,
    seeds: &[String],
    config: &BfsConfig,
    graph: &mut Graph,
    callbacks: &mut dyn BfsCallbacks,
) -> BfsResult {
    let mut visited: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<(String, u32)> = seeds.iter().map(|s| (s.clone(), 0u32)).collect();

    let mut total_passes: usize = 0;
    let mut edges_added: usize = 0;

    while let Some((entity, depth)) = queue.pop_front() {
        if visited.contains(&entity) || depth > config.max_depth {
            continue;
        }
        if visited.len() >= config.max_entities {
            break;
        }
        visited.insert(entity.clone());

        callbacks.on_entity(&entity, depth, visited.len(), queue.len());

        for template in templates.all() {
            let prompt = template.format(&entity);
            let max_tok = if template.multi_token {
                config.max_chain_tokens
            } else {
                1
            };

            let result = match chain_tokens(provider, &prompt, max_tok, config.min_confidence, None)
            {
                Ok(r) => r,
                Err(_) => continue,
            };

            total_passes += result.num_passes;

            if !result.answer.is_empty() && result.avg_probability() >= config.min_confidence {
                let edge = Edge::new(&entity, &template.relation, &result.answer)
                    .with_confidence(result.avg_probability())
                    .with_source(SourceType::Parametric)
                    .with_metadata(
                        "forward_passes",
                        serde_json::Value::from(result.num_passes as u64),
                    )
                    .with_metadata("model", serde_json::Value::from(provider.model_name()));

                if !graph.exists(&entity, &template.relation, &result.answer) {
                    graph.add_edge(edge.clone());
                    edges_added += 1;
                    callbacks.on_edge(&edge, depth);
                }

                // Queue valid entities for further exploration
                let obj = result.answer.trim().to_string();
                if !visited.contains(&obj) && depth < config.max_depth && is_valid_entity(&obj) {
                    queue.push_back((obj, depth + 1));
                }
            }
        }

        callbacks.on_checkpoint(graph);
    }

    BfsResult {
        entities_visited: visited.len(),
        edges_added,
        total_forward_passes: total_passes,
        queue_remaining: queue.len(),
    }
}

#[derive(Debug, Clone)]
pub struct BfsResult {
    pub entities_visited: usize,
    pub edges_added: usize,
    pub total_forward_passes: usize,
    pub queue_remaining: usize,
}

/// Should this string be followed in BFS? Proper nouns and numbers only.
fn is_valid_entity(text: &str) -> bool {
    let clean = text.trim();
    if clean.is_empty() {
        return false;
    }

    let first = clean.chars().next().unwrap();
    if !first.is_uppercase() && !first.is_ascii_digit() {
        return false;
    }

    // Skip articles
    let lower = clean.to_lowercase();
    if lower.starts_with("the ") || lower.starts_with("a ") {
        return false;
    }

    // Skip long phrases
    if clean.split_whitespace().count() > 4 {
        return false;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_entity() {
        assert!(is_valid_entity("France"));
        assert!(is_valid_entity("Paris"));
        assert!(is_valid_entity("1756"));
        assert!(is_valid_entity("New York"));
        assert!(!is_valid_entity("the city"));
        assert!(!is_valid_entity("a very long phrase that is not an entity"));
        assert!(!is_valid_entity("lowercase"));
        assert!(!is_valid_entity(""));
    }
}
