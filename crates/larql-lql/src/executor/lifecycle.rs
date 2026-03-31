/// Lifecycle executor: USE, STATS, EXTRACT, COMPILE, DIFF.

use std::path::PathBuf;

use crate::ast::*;
use crate::error::LqlError;
use crate::relations::RelationClassifier;
use super::{Backend, Session};
use super::helpers::{format_number, format_bytes, dir_size};

impl Session {
    pub(crate) fn exec_use(&mut self, target: &UseTarget) -> Result<Vec<String>, LqlError> {
        match target {
            UseTarget::Vindex(path) => {
                let path = PathBuf::from(path);
                if !path.exists() {
                    return Err(LqlError::Execution(format!(
                        "vindex not found: {}",
                        path.display()
                    )));
                }

                let config = larql_vindex::load_vindex_config(&path)
                    .map_err(|e| LqlError::Execution(format!("failed to load vindex config: {e}")))?;

                let mut cb = larql_vindex::SilentLoadCallbacks;
                let index = larql_vindex::VectorIndex::load_vindex(&path, &mut cb)
                    .map_err(|e| LqlError::Execution(format!("failed to load vindex: {e}")))?;

                let total_features: usize = config.layers.iter().map(|l| l.num_features).sum();

                let relation_classifier = RelationClassifier::from_vindex(&path);

                let rc_status = match &relation_classifier {
                    Some(rc) if rc.has_clusters() => {
                        let probe_info = if rc.num_probe_labels() > 0 {
                            format!(", {} probe-confirmed", rc.num_probe_labels())
                        } else {
                            String::new()
                        };
                        format!(", relations: {} types{}", rc.num_clusters(), probe_info)
                    }
                    _ => String::new(),
                };

                let out = vec![format!(
                    "Using: {} ({} layers, {} features, model: {}{})",
                    path.display(),
                    config.num_layers,
                    format_number(total_features),
                    config.model,
                    rc_status,
                )];

                self.backend = Backend::Vindex { path, config, index, relation_classifier };
                Ok(out)
            }
            UseTarget::Model { id, auto_extract } => {
                let mut out = vec![format!(
                    "Direct model access not yet implemented. Extract first:"
                )];
                out.push(format!(
                    "  EXTRACT MODEL \"{}\" INTO \"{}.vindex\";",
                    id,
                    id.split('/').last().unwrap_or(id)
                ));
                if *auto_extract {
                    out.push("  (AUTO_EXTRACT noted — will be supported in a future version)".into());
                }
                Ok(out)
            }
        }
    }

    pub(crate) fn exec_stats(&self, _vindex_path: Option<&str>) -> Result<Vec<String>, LqlError> {
        match &self.backend {
            Backend::Vindex { path, config, index, relation_classifier } => {
                let total_features: usize = config.layers.iter().map(|l| l.num_features).sum();
                let file_size = dir_size(path);

                let mut out = Vec::new();
                out.push(format!("Model:           {}", config.model));
                out.push(String::new());
                out.push(format!(
                    "Features:        {} ({} x {} layers)",
                    format_number(total_features),
                    format_number(config.intermediate_size),
                    config.num_layers,
                ));

                // Knowledge graph coverage
                out.push(String::new());
                out.push("Knowledge Graph:".into());

                if let Some(rc) = relation_classifier {
                    let num_clusters = rc.num_clusters();
                    let num_probes = rc.num_probe_labels();

                    // Count mapped vs unmapped clusters
                    let mut mapped_clusters = 0;
                    for cluster_id in 0..num_clusters {
                        if let Some((label, _, _)) = rc.cluster_info(cluster_id) {
                            if !label.is_empty() {
                                mapped_clusters += 1;
                            }
                        }
                    }
                    let unmapped_clusters = num_clusters.saturating_sub(mapped_clusters);

                    // Count probe-confirmed relation types
                    // (unique labels among probe labels)
                    let probe_type_count = if num_probes > 0 {
                        let mut types = std::collections::HashSet::new();
                        // We can approximate by scanning loaded layers
                        let layers = index.loaded_layers();
                        for layer in &layers {
                            let n = index.num_features(*layer);
                            for feat in 0..n {
                                if rc.is_probe_label(*layer, feat) {
                                    if let Some(label) = rc.label_for_feature(*layer, feat) {
                                        types.insert(label.to_string());
                                    }
                                }
                            }
                        }
                        types.len()
                    } else {
                        0
                    };

                    out.push(format!("  Clusters:          {}", num_clusters));
                    if num_probes > 0 {
                        out.push(format!(
                            "  Mapped relations:  {} features ({} types, probe-confirmed)",
                            num_probes, probe_type_count,
                        ));
                    }
                    if mapped_clusters > 0 {
                        out.push(format!(
                            "  Partially mapped:  {} clusters (Wikidata/WordNet matched)",
                            mapped_clusters,
                        ));
                    }
                    out.push(format!(
                        "  Unmapped:          {} clusters (model knows, we haven't identified yet)",
                        unmapped_clusters,
                    ));
                } else {
                    out.push("  (no relation clusters found)".into());
                }

                // Layer band breakdown
                let layers = index.loaded_layers();
                let syntax_features: usize = layers.iter()
                    .filter(|l| **l <= 13)
                    .map(|l| index.num_features(*l))
                    .sum();
                let knowledge_features: usize = layers.iter()
                    .filter(|l| **l >= 14 && **l <= 27)
                    .map(|l| index.num_features(*l))
                    .sum();
                let output_features: usize = layers.iter()
                    .filter(|l| **l >= 28)
                    .map(|l| index.num_features(*l))
                    .sum();

                out.push(String::new());
                out.push("  By layer band:".into());
                out.push(format!(
                    "    Syntax (L0-13):     {} features",
                    format_number(syntax_features),
                ));
                out.push(format!(
                    "    Knowledge (L14-27): {} features",
                    format_number(knowledge_features),
                ));
                out.push(format!(
                    "    Output (L28-33):    {} features",
                    format_number(output_features),
                ));

                // Coverage summary
                if let Some(rc) = relation_classifier {
                    let num_probes = rc.num_probe_labels();
                    let num_clusters = rc.num_clusters();

                    if num_clusters > 0 {
                        let mut mapped_clusters = 0;
                        for cluster_id in 0..num_clusters {
                            if let Some((label, _, _)) = rc.cluster_info(cluster_id) {
                                if !label.is_empty() {
                                    mapped_clusters += 1;
                                }
                            }
                        }

                        let probe_pct = if total_features > 0 {
                            (num_probes as f64 / total_features as f64) * 100.0
                        } else {
                            0.0
                        };
                        let cluster_pct = (mapped_clusters as f64 / num_clusters as f64) * 100.0;
                        let total_mapped_pct = ((mapped_clusters as f64 / num_clusters as f64) * 100.0)
                            .min(100.0);
                        let unmapped_pct = 100.0 - total_mapped_pct;

                        out.push(String::new());
                        out.push("  Coverage:".into());
                        out.push(format!(
                            "    Probe-confirmed:   {:.2}% of features ({} / {})",
                            probe_pct, num_probes, format_number(total_features),
                        ));
                        out.push(format!(
                            "    Cluster-labelled:  {:.0}% of clusters ({} / {})",
                            cluster_pct, mapped_clusters, num_clusters,
                        ));
                        out.push(format!(
                            "    Unmapped:          ~{:.0}% — the model knows more than we've labelled",
                            unmapped_pct,
                        ));
                    }
                }

                out.push(String::new());
                out.push(format!("Index size:      {}", format_bytes(file_size)));
                out.push(format!("Path:            {}", path.display()));
                Ok(out)
            }
            Backend::None => Err(LqlError::NoBackend),
        }
    }

    // ── EXTRACT ──

    pub(crate) fn exec_extract(
        &mut self,
        model: &str,
        output: &str,
        _components: Option<&[Component]>,
        _layers: Option<&Range>,
        _extract_level: ExtractLevel,
    ) -> Result<Vec<String>, LqlError> {
        let output_dir = PathBuf::from(output);

        let mut out = Vec::new();
        out.push(format!("Loading model: {model}..."));

        let inference_model = larql_inference::InferenceModel::load(model)
            .map_err(|e| LqlError::Execution(format!("failed to load model: {e}")))?;

        out.push(format!(
            "Model loaded ({} layers, hidden={}). Extracting to {}...",
            inference_model.num_layers(),
            inference_model.hidden_size(),
            output_dir.display()
        ));

        std::fs::create_dir_all(&output_dir)
            .map_err(|e| LqlError::Execution(format!("failed to create output dir: {e}")))?;

        let mut callbacks = LqlBuildCallbacks::new();
        larql_inference::vindex::build_vindex(
            inference_model.weights(),
            inference_model.tokenizer(),
            model,
            &output_dir,
            10,
            &mut callbacks,
        )
        .map_err(|e| LqlError::Execution(format!("extraction failed: {e}")))?;

        out.extend(callbacks.messages);
        out.push(format!("Extraction complete: {}", output_dir.display()));

        // Auto-load the newly created vindex
        let config = larql_vindex::load_vindex_config(&output_dir)
            .map_err(|e| LqlError::Execution(format!("failed to load vindex config: {e}")))?;
        let mut cb = larql_vindex::SilentLoadCallbacks;
        let index = larql_vindex::VectorIndex::load_vindex(&output_dir, &mut cb)
            .map_err(|e| LqlError::Execution(format!("failed to load vindex: {e}")))?;
        let relation_classifier = RelationClassifier::from_vindex(&output_dir);

        let total_features: usize = config.layers.iter().map(|l| l.num_features).sum();
        out.push(format!(
            "Using: {} ({} layers, {} features)",
            output_dir.display(),
            config.num_layers,
            format_number(total_features),
        ));

        self.backend = Backend::Vindex {
            path: output_dir,
            config,
            index,
            relation_classifier,
        };

        Ok(out)
    }

    // ── COMPILE ──

    pub(crate) fn exec_compile(
        &self,
        vindex: &VindexRef,
        output: &str,
        _format: Option<OutputFormat>,
    ) -> Result<Vec<String>, LqlError> {
        let vindex_path = match vindex {
            VindexRef::Current => {
                match &self.backend {
                    Backend::Vindex { path, .. } => path.clone(),
                    Backend::None => return Err(LqlError::NoBackend),
                }
            }
            VindexRef::Path(p) => PathBuf::from(p),
        };

        let config = larql_vindex::load_vindex_config(&vindex_path)
            .map_err(|e| LqlError::Execution(format!("failed to load vindex config: {e}")))?;

        if !config.has_model_weights {
            return Err(LqlError::Execution(format!(
                "COMPILE requires model weights in the vindex.\n\
                 This vindex was built without --include-weights.\n\
                 Rebuild: EXTRACT MODEL \"{}\" INTO \"{}\" (with weights)",
                config.model,
                vindex_path.display()
            )));
        }

        let output_dir = PathBuf::from(output);
        std::fs::create_dir_all(&output_dir)
            .map_err(|e| LqlError::Execution(format!("failed to create output dir: {e}")))?;

        let mut cb = larql_vindex::SilentLoadCallbacks;
        let weights = larql_inference::load_model_weights_from_vindex(&vindex_path, &mut cb)
            .map_err(|e| LqlError::Execution(format!("failed to load model weights: {e}")))?;

        let mut build_cb = larql_inference::vindex::SilentBuildCallbacks;
        larql_inference::write_model_weights(&weights, &output_dir, &mut build_cb)
            .map_err(|e| LqlError::Execution(format!("failed to write model: {e}")))?;

        // Copy tokenizer
        let tok_src = vindex_path.join("tokenizer.json");
        let tok_dst = output_dir.join("tokenizer.json");
        if tok_src.exists() {
            std::fs::copy(&tok_src, &tok_dst)
                .map_err(|e| LqlError::Execution(format!("failed to copy tokenizer: {e}")))?;
        }

        let mut out = Vec::new();
        out.push(format!(
            "Compiled {} → {}",
            vindex_path.display(),
            output_dir.display()
        ));
        out.push(format!("Model: {}", config.model));
        out.push(format!("Size: {}", format_bytes(dir_size(&output_dir))));
        Ok(out)
    }

    // ── DIFF ──

    pub(crate) fn exec_diff(
        &self,
        a: &VindexRef,
        b: &VindexRef,
        layer_filter: Option<u32>,
        _relation: Option<&str>,
        limit: Option<u32>,
    ) -> Result<Vec<String>, LqlError> {
        let path_a = self.resolve_vindex_ref(a)?;
        let path_b = self.resolve_vindex_ref(b)?;

        let mut cb = larql_vindex::SilentLoadCallbacks;
        let index_a = larql_vindex::VectorIndex::load_vindex(&path_a, &mut cb)
            .map_err(|e| LqlError::Execution(format!("failed to load {}: {e}", path_a.display())))?;
        let index_b = larql_vindex::VectorIndex::load_vindex(&path_b, &mut cb)
            .map_err(|e| LqlError::Execution(format!("failed to load {}: {e}", path_b.display())))?;

        let limit = limit.unwrap_or(20) as usize;

        let mut out = Vec::new();
        out.push(format!(
            "Diff: {} vs {}",
            path_a.display(),
            path_b.display()
        ));
        out.push(format!(
            "{:<8} {:<8} {:<20} {:<20} {:>10}",
            "Layer", "Feature", "A (token)", "B (token)", "Status"
        ));
        out.push("-".repeat(70));

        let layers_a = index_a.loaded_layers();
        let mut diff_count = 0;

        for layer in &layers_a {
            if let Some(l) = layer_filter {
                if *layer != l as usize {
                    continue;
                }
            }
            if diff_count >= limit {
                break;
            }

            let metas_a = index_a.down_meta_at(*layer);
            let metas_b = index_b.down_meta_at(*layer);

            let len_a = metas_a.map(|m| m.len()).unwrap_or(0);
            let len_b = metas_b.map(|m| m.len()).unwrap_or(0);
            let max_features = len_a.max(len_b);

            for feat in 0..max_features {
                if diff_count >= limit {
                    break;
                }

                let meta_a = metas_a
                    .and_then(|m| m.get(feat))
                    .and_then(|m| m.as_ref());
                let meta_b = metas_b
                    .and_then(|m| m.get(feat))
                    .and_then(|m| m.as_ref());

                let status = match (meta_a, meta_b) {
                    (Some(a), Some(b)) => {
                        if a.top_token != b.top_token || (a.c_score - b.c_score).abs() > 0.01 {
                            "modified"
                        } else {
                            continue;
                        }
                    }
                    (Some(_), None) => "removed",
                    (None, Some(_)) => "added",
                    (None, None) => continue,
                };

                let tok_a = meta_a.map(|m| m.top_token.as_str()).unwrap_or("-");
                let tok_b = meta_b.map(|m| m.top_token.as_str()).unwrap_or("-");

                out.push(format!(
                    "L{:<7} F{:<7} {:<20} {:<20} {:>10}",
                    layer, feat, tok_a, tok_b, status
                ));
                diff_count += 1;
            }
        }

        if diff_count == 0 {
            out.push("  (no differences found)".into());
        } else {
            out.push(format!("\n{} differences shown (limit {})", diff_count, limit));
        }

        Ok(out)
    }

    /// Resolve a VindexRef to a concrete path.
    fn resolve_vindex_ref(&self, vref: &VindexRef) -> Result<PathBuf, LqlError> {
        match vref {
            VindexRef::Current => match &self.backend {
                Backend::Vindex { path, .. } => Ok(path.clone()),
                Backend::None => Err(LqlError::NoBackend),
            },
            VindexRef::Path(p) => {
                let path = PathBuf::from(p);
                if !path.exists() {
                    return Err(LqlError::Execution(format!(
                        "vindex not found: {}",
                        path.display()
                    )));
                }
                Ok(path)
            }
        }
    }
}

/// Build callbacks that collect stage messages for LQL output.
struct LqlBuildCallbacks {
    messages: Vec<String>,
    current_stage: String,
}

impl LqlBuildCallbacks {
    fn new() -> Self {
        Self {
            messages: Vec::new(),
            current_stage: String::new(),
        }
    }
}

impl larql_inference::vindex::IndexBuildCallbacks for LqlBuildCallbacks {
    fn on_stage(&mut self, stage: &str) {
        self.current_stage = stage.to_string();
        self.messages.push(format!("  Stage: {stage}"));
    }

    fn on_stage_done(&mut self, stage: &str, elapsed_ms: f64) {
        self.messages.push(format!("  {stage}: {elapsed_ms:.0}ms"));
    }
}
