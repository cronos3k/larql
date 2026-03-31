//! Entity probing — run tokens through the model to confirm which features fire.
//!
//! For each entity in the reference data, embed it, run gate KNN at each
//! knowledge layer, and record which features activate. This gives confirmed
//! (entity, feature) mappings — ground truth for pair-based relation labeling.

use std::collections::HashMap;

use ndarray::Array1;

use larql_vindex::VectorIndex;

/// Result of probing: maps (layer, feature) → list of entity names that activate it.
pub struct ProbeResult {
    /// (layer, feature) → entity names
    pub feature_entities: HashMap<(usize, usize), Vec<String>>,
    /// Total entities probed
    pub num_entities: usize,
    /// Total (entity, feature) activations found
    pub num_activations: usize,
}

/// Probe a set of entities against a VectorIndex to find which features they activate.
///
/// For each entity:
/// 1. Look up its embedding (may be multi-token → averaged)
/// 2. Run gate KNN at each knowledge layer
/// 3. Record the top-K features that fire
///
/// Returns a map from (layer, feature) → entity names.
pub fn probe_entities(
    entities: &[String],
    index: &VectorIndex,
    embed: &ndarray::Array2<f32>,
    embed_scale: f32,
    tokenizer: &tokenizers::Tokenizer,
    layers: &[usize],
    top_k: usize,
) -> ProbeResult {
    let hidden = embed.shape()[1];
    let vocab_size = embed.shape()[0];
    let mut feature_entities: HashMap<(usize, usize), Vec<String>> = HashMap::new();
    let mut num_activations = 0;

    let total = entities.len();
    for (ei, entity) in entities.iter().enumerate() {
        if ei % 1000 == 0 && ei > 0 {
            eprint!("\r    Probed {}/{} entities ({} activations)...", ei, total, num_activations);
        }
        // Encode entity → token IDs → averaged embedding
        let encoding = match tokenizer.encode(entity.as_str(), false) {
            Ok(e) => e,
            Err(_) => continue,
        };
        let ids = encoding.get_ids();
        if ids.is_empty() {
            continue;
        }

        let mut avg = Array1::<f32>::zeros(hidden);
        let mut n = 0;
        for &id in ids {
            if id > 2 && (id as usize) < vocab_size {
                avg += &embed.row(id as usize).mapv(|v| v * embed_scale);
                n += 1;
            }
        }
        if n == 0 {
            continue;
        }
        avg /= n as f32;

        // Run gate KNN at each layer — only keep strong activations
        for &layer in layers {
            let hits = index.gate_knn(layer, &avg, top_k);
            // Only keep the top 3 strongest activations per layer (most specific)
            for (feature, score) in hits.into_iter().take(3) {
                if score > 5.0 {
                    feature_entities
                        .entry((layer, feature))
                        .or_default()
                        .push(entity.clone());
                    num_activations += 1;
                }
            }
        }
    }

    if total > 1000 {
        eprintln!("\r    Probed {}/{} entities ({} activations)    ", total, total, num_activations);
    }

    ProbeResult {
        feature_entities,
        num_entities: entities.len(),
        num_activations,
    }
}

/// Extract unique entity names from the reference triple files.
pub fn extract_probe_entities(triples_path: &std::path::Path) -> Vec<String> {
    let mut entities = std::collections::HashSet::new();

    if let Ok(text) = std::fs::read_to_string(triples_path) {
        if let Ok(data) = serde_json::from_str::<serde_json::Value>(&text) {
            if let Some(obj) = data.as_object() {
                for (_rel, val) in obj {
                    if let Some(pairs) = val.get("pairs").and_then(|v| v.as_array()) {
                        for pair in pairs {
                            if let Some(arr) = pair.as_array() {
                                // Add both subject and object as probe entities
                                for item in arr {
                                    if let Some(s) = item.as_str() {
                                        let s = s.trim();
                                        // Only probe short, clean entity names
                                        if s.len() >= 2
                                            && s.len() <= 30
                                            && !s.contains('(')
                                            && !s.contains("http")
                                        {
                                            entities.insert(s.to_string());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let mut sorted: Vec<String> = entities.into_iter().collect();
    sorted.sort();
    sorted
}

/// Build confirmed (input_entity, output_token) pairs for each feature
/// by combining probe results with down_meta output tokens.
///
/// Only includes features that fire SELECTIVELY — if a feature fires for
/// too many entities, it's a category feature, not a specific fact.
/// Max 20 entities per feature keeps only the specific ones.
pub fn build_confirmed_pairs(
    probe: &ProbeResult,
    index: &VectorIndex,
) -> Vec<(String, String, usize, usize)> {
    // (entity, target, layer, feature)
    let mut pairs = Vec::new();
    let mut skipped_broad = 0;

    for (&(layer, feature), entities) in &probe.feature_entities {
        // Skip broad features that fire for many entities — they're category features
        if entities.len() > 20 {
            skipped_broad += 1;
            continue;
        }

        if let Some(meta) = index.feature_meta(layer, feature) {
            let target = &meta.top_token;
            if target.len() >= 2 {
                for entity in entities {
                    pairs.push((
                        entity.clone(),
                        target.clone(),
                        layer,
                        feature,
                    ));
                }
            }
        }
    }

    if skipped_broad > 0 {
        eprintln!("  Skipped {} broad features (>20 entities each)", skipped_broad);
    }

    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_entities_from_json() {
        let dir = std::env::temp_dir().join("probe_test");
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join("test_triples.json");
        std::fs::write(&path, r#"{
            "capital": {"pairs": [["France", "Paris"], ["Germany", "Berlin"]]},
            "language": {"pairs": [["France", "French"]]}
        }"#).unwrap();

        let entities = extract_probe_entities(&path);
        assert!(entities.contains(&"France".to_string()));
        assert!(entities.contains(&"Paris".to_string()));
        assert!(entities.contains(&"Berlin".to_string()));
        assert!(entities.contains(&"French".to_string()));
        assert!(entities.contains(&"Germany".to_string()));

        std::fs::remove_file(&path).ok();
    }
}
