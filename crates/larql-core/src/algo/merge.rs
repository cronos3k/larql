//! Graph merging with conflict resolution strategies.

use crate::core::enums::MergeStrategy;
use crate::core::graph::Graph;

/// Merge edges from `other` into `target` using the given strategy.
/// Returns count of edges added or updated.
pub fn merge_graphs(target: &mut Graph, other: &Graph) -> usize {
    merge_graphs_with_strategy(target, other, MergeStrategy::Union)
}

/// Merge with explicit strategy.
pub fn merge_graphs_with_strategy(
    target: &mut Graph,
    other: &Graph,
    strategy: MergeStrategy,
) -> usize {
    let mut count = 0;

    for edge in other.edges() {
        if !target.exists(&edge.subject, &edge.relation, &edge.object) {
            // New edge — always add
            target.add_edge(edge.clone());
            count += 1;
        } else {
            // Duplicate triple — apply strategy
            match strategy {
                MergeStrategy::Union => {
                    // Keep existing, skip duplicate
                }
                MergeStrategy::MaxConfidence => {
                    // Replace if new has higher confidence
                    let existing = target.select(&edge.subject, Some(&edge.relation));
                    if let Some(old) = existing.iter().find(|e| e.object == edge.object) {
                        if edge.confidence > old.confidence {
                            target.remove_edge(&edge.subject, &edge.relation, &edge.object);
                            target.add_edge(edge.clone());
                            count += 1;
                        }
                    }
                }
                MergeStrategy::SourcePriority => {
                    // Replace if new source has higher priority
                    let existing = target.select(&edge.subject, Some(&edge.relation));
                    if let Some(old) = existing.iter().find(|e| e.object == edge.object) {
                        if source_priority(&edge.source) > source_priority(&old.source) {
                            target.remove_edge(&edge.subject, &edge.relation, &edge.object);
                            target.add_edge(edge.clone());
                            count += 1;
                        }
                    }
                }
            }
        }
    }

    count
}

/// Source priority ordering (higher = preferred).
fn source_priority(source: &crate::core::enums::SourceType) -> u8 {
    use crate::core::enums::SourceType;
    match source {
        SourceType::Manual => 5,
        SourceType::Wikidata => 4,
        SourceType::Parametric => 3,
        SourceType::Document => 2,
        SourceType::Installed => 1,
        SourceType::Unknown => 0,
    }
}
