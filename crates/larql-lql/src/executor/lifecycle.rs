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
            UseTarget::Vindex(path_str) => {
                // Resolve hf:// paths to local cache
                let path = if larql_vindex::is_hf_path(path_str) {
                    larql_vindex::resolve_hf_vindex(path_str)
                        .map_err(|e| LqlError::Execution(format!("HuggingFace download failed: {e}")))?
                } else {
                    let p = PathBuf::from(path_str);
                    if !p.exists() {
                        return Err(LqlError::Execution(format!(
                            "vindex not found: {}",
                            p.display()
                        )));
                    }
                    p
                };

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

                let router = larql_vindex::RouterIndex::load(&path, &config);
                let patched = larql_vindex::PatchedVindex::new(index);
                self.backend = Backend::Vindex { path, config, patched, relation_classifier, router };
                // Reset any previous patch session
                self.patch_recording = None;
                self.auto_patch = false;
                Ok(out)
            }
            UseTarget::Model { id, auto_extract: _ } => {
                let mut out = Vec::new();
                out.push(format!("Loading model: {id}..."));

                let model_path = larql_inference::resolve_model_path(id)
                    .map_err(|e| LqlError::Execution(format!("failed to resolve model: {e}")))?;
                let weights = larql_inference::load_model_dir(&model_path)
                    .map_err(|e| LqlError::Execution(format!("failed to load model: {e}")))?;
                let tokenizer = larql_inference::load_tokenizer(&model_path)
                    .map_err(|e| LqlError::Execution(format!("failed to load tokenizer: {e}")))?;

                let size_gb = dir_size(&model_path) as f64 / (1024.0 * 1024.0 * 1024.0);
                out.push(format!(
                    "Using model: {} ({} layers, hidden={}, {:.1} GB, live weights)",
                    id,
                    weights.num_layers,
                    weights.hidden_size,
                    size_gb,
                ));
                out.push("Supported: INFER, EXPLAIN INFER, STATS. For WALK/DESCRIBE/SELECT, use EXTRACT first.".into());

                self.backend = Backend::Weight {
                    model_id: id.clone(),
                    weights,
                    tokenizer,
                };
                self.patch_recording = None;
                self.auto_patch = false;
                Ok(out)
            }
            UseTarget::Remote(url) => self.exec_use_remote(url),
        }
    }

    pub(crate) fn exec_stats(&self, _vindex_path: Option<&str>) -> Result<Vec<String>, LqlError> {
        match &self.backend {
            Backend::Vindex { path, config, patched, relation_classifier, .. } => {
                let index = patched.base();
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
            Backend::Weight { model_id, weights, .. } => {
                let mut out = Vec::new();
                out.push(format!("Model:           {}", model_id));
                out.push(format!("Backend:         live weights (no vindex)"));
                out.push(String::new());
                out.push(format!("Layers:          {}", weights.num_layers));
                out.push(format!("Hidden size:     {}", weights.hidden_size));
                out.push(format!("Intermediate:    {}", weights.intermediate_size));
                out.push(format!("Vocab size:      {}", format_number(weights.vocab_size)));
                out.push(String::new());
                out.push("Supported:       INFER, EXPLAIN INFER, STATS".into());
                out.push("For WALK/DESCRIBE/SELECT/INSERT: EXTRACT into a vindex first.".into());
                Ok(out)
            }
            Backend::Remote { .. } => self.remote_stats(),
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

        // Map AST ExtractLevel to vindex ExtractLevel
        let vindex_level = match _extract_level {
            ExtractLevel::Browse => larql_vindex::ExtractLevel::Browse,
            ExtractLevel::Inference => larql_vindex::ExtractLevel::Inference,
            ExtractLevel::All => larql_vindex::ExtractLevel::All,
        };

        let mut callbacks = LqlBuildCallbacks::new();
        larql_vindex::build_vindex(
            inference_model.weights(),
            inference_model.tokenizer(),
            model,
            &output_dir,
            10,
            vindex_level,
            larql_vindex::StorageDtype::F32,
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

        let router = larql_vindex::RouterIndex::load(&output_dir, &config);
        let patched = larql_vindex::PatchedVindex::new(index);
        self.backend = Backend::Vindex {
            path: output_dir,
            config,
            patched,
            relation_classifier,
            router,
        };

        Ok(out)
    }

    // ── COMPILE ──

    pub(crate) fn exec_compile(
        &self,
        vindex: &VindexRef,
        output: &str,
        _format: Option<OutputFormat>,
        target: CompileTarget,
    ) -> Result<Vec<String>, LqlError> {
        let vindex_path = match vindex {
            VindexRef::Current => {
                match &self.backend {
                    Backend::Vindex { path, .. } => path.clone(),
                    _ => return Err(LqlError::NoBackend),
                }
            }
            VindexRef::Path(p) => PathBuf::from(p),
        };

        match target {
            CompileTarget::Vindex => self.exec_compile_into_vindex(&vindex_path, output),
            CompileTarget::Model => self.exec_compile_into_model(&vindex_path, output),
        }
    }

    fn exec_compile_into_model(
        &self,
        vindex_path: &std::path::Path,
        output: &str,
    ) -> Result<Vec<String>, LqlError> {
        let config = larql_vindex::load_vindex_config(vindex_path)
            .map_err(|e| LqlError::Execution(format!("failed to load vindex config: {e}")))?;

        if !config.has_model_weights {
            return Err(LqlError::Execution(format!(
                "COMPILE INTO MODEL requires model weights in the vindex.\n\
                 This vindex was built without --include-weights.\n\
                 Rebuild: EXTRACT MODEL \"{}\" INTO \"{}\" WITH ALL",
                config.model, vindex_path.display()
            )));
        }

        let output_dir = PathBuf::from(output);
        std::fs::create_dir_all(&output_dir)
            .map_err(|e| LqlError::Execution(format!("failed to create output dir: {e}")))?;

        let mut cb = larql_vindex::SilentLoadCallbacks;
        let weights = larql_vindex::load_model_weights(vindex_path, &mut cb)
            .map_err(|e| LqlError::Execution(format!("failed to load model weights: {e}")))?;

        let mut build_cb = larql_vindex::SilentBuildCallbacks;
        larql_vindex::write_model_weights(&weights, &output_dir, &mut build_cb)
            .map_err(|e| LqlError::Execution(format!("failed to write model: {e}")))?;

        let tok_src = vindex_path.join("tokenizer.json");
        let tok_dst = output_dir.join("tokenizer.json");
        if tok_src.exists() {
            std::fs::copy(&tok_src, &tok_dst)
                .map_err(|e| LqlError::Execution(format!("failed to copy tokenizer: {e}")))?;
        }

        let mut out = Vec::new();
        out.push(format!("Compiled {} → {}", vindex_path.display(), output_dir.display()));
        out.push(format!("Model: {}", config.model));
        out.push(format!("Size: {}", format_bytes(dir_size(&output_dir))));
        Ok(out)
    }

    fn exec_compile_into_vindex(
        &self,
        source_path: &std::path::Path,
        output: &str,
    ) -> Result<Vec<String>, LqlError> {
        let output_dir = PathBuf::from(output);
        std::fs::create_dir_all(&output_dir)
            .map_err(|e| LqlError::Execution(format!("failed to create output dir: {e}")))?;

        // Load the current vindex with patches applied
        let (path, config, patched) = self.require_vindex()?;

        // Bake patches into a clean VectorIndex
        let baked = if patched.num_overrides() > 0 {
            patched.bake_down()
        } else {
            // No patches — use base directly (avoid unnecessary clone)
            patched.base().clone()
        };

        // Save gate vectors + down_meta + config to the new directory
        let layer_infos = baked.save_gate_vectors(&output_dir)
            .map_err(|e| LqlError::Execution(format!("failed to save gate vectors: {e}")))?;
        let dm_count = baked.save_down_meta(&output_dir)
            .map_err(|e| LqlError::Execution(format!("failed to save down_meta: {e}")))?;

        // Copy embeddings
        let embed_src = path.join("embeddings.bin");
        let embed_dst = output_dir.join("embeddings.bin");
        if embed_src.exists() {
            std::fs::copy(&embed_src, &embed_dst)
                .map_err(|e| LqlError::Execution(format!("failed to copy embeddings: {e}")))?;
        }

        // Copy tokenizer
        let tok_src = path.join("tokenizer.json");
        let tok_dst = output_dir.join("tokenizer.json");
        if tok_src.exists() {
            std::fs::copy(&tok_src, &tok_dst)
                .map_err(|e| LqlError::Execution(format!("failed to copy tokenizer: {e}")))?;
        }

        // Copy label files
        for label_file in &["relation_clusters.json", "feature_clusters.jsonl", "feature_labels.json"] {
            let src = path.join(label_file);
            let dst = output_dir.join(label_file);
            if src.exists() {
                let _ = std::fs::copy(&src, &dst);
            }
        }

        // Write updated config with new layer infos
        let mut new_config = config.clone();
        new_config.layers = layer_infos;
        new_config.checksums = larql_vindex::format::checksums::compute_checksums(&output_dir).ok();
        larql_vindex::VectorIndex::save_config(&new_config, &output_dir)
            .map_err(|e| LqlError::Execution(format!("failed to save config: {e}")))?;

        let mut out = Vec::new();
        out.push(format!("Compiled {} → {}", source_path.display(), output_dir.display()));
        out.push(format!("Features: {}", dm_count));
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
        into_patch: Option<&str>,
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

        // If INTO PATCH specified, extract diff as a .vlp file
        if let Some(patch_path) = into_patch {
            let mut operations = Vec::new();

            // Re-scan without limit for the full diff
            for layer in &layers_a {
                if let Some(l) = layer_filter {
                    if *layer != l as usize { continue; }
                }
                let metas_a = index_a.down_meta_at(*layer);
                let metas_b = index_b.down_meta_at(*layer);
                let len_a = metas_a.map(|m| m.len()).unwrap_or(0);
                let len_b = metas_b.map(|m| m.len()).unwrap_or(0);

                for feat in 0..len_a.max(len_b) {
                    let ma = metas_a.and_then(|m| m.get(feat)).and_then(|m| m.as_ref());
                    let mb = metas_b.and_then(|m| m.get(feat)).and_then(|m| m.as_ref());

                    match (ma, mb) {
                        (Some(_a), Some(b)) if _a.top_token != b.top_token || (_a.c_score - b.c_score).abs() > 0.01 => {
                            operations.push(larql_vindex::PatchOp::Update {
                                layer: *layer,
                                feature: feat,
                                gate_vector_b64: None,
                                down_meta: Some(larql_vindex::patch::core::PatchDownMeta {
                                    top_token: b.top_token.clone(),
                                    top_token_id: b.top_token_id,
                                    c_score: b.c_score,
                                }),
                            });
                        }
                        (Some(_), None) => {
                            operations.push(larql_vindex::PatchOp::Delete {
                                layer: *layer,
                                feature: feat,
                                reason: Some("removed in target".into()),
                            });
                        }
                        (None, Some(b)) => {
                            operations.push(larql_vindex::PatchOp::Insert {
                                layer: *layer,
                                feature: feat,
                                relation: None,
                                entity: String::new(),
                                target: b.top_token.clone(),
                                confidence: Some(b.c_score),
                                gate_vector_b64: None,
                                down_meta: Some(larql_vindex::patch::core::PatchDownMeta {
                                    top_token: b.top_token.clone(),
                                    top_token_id: b.top_token_id,
                                    c_score: b.c_score,
                                }),
                            });
                        }
                        _ => {}
                    }
                }
            }

            let model_name = match &self.backend {
                Backend::Vindex { config, .. } => config.model.clone(),
                Backend::Weight { model_id, .. } => model_id.clone(),
                _ => "unknown".into(),
            };

            let patch = larql_vindex::VindexPatch {
                version: 1,
                base_model: model_name,
                base_checksum: None,
                created_at: String::new(),
                description: Some(format!("Diff: {} vs {}", path_a.display(), path_b.display())),
                author: None,
                tags: vec![],
                operations,
            };

            let (ins, upd, del) = patch.counts();
            patch.save(std::path::Path::new(patch_path))
                .map_err(|e| LqlError::Execution(format!("failed to save patch: {e}")))?;
            out.push(format!(
                "Extracted: {} ({} ops: {} inserts, {} updates, {} deletes)",
                patch_path, patch.len(), ins, upd, del,
            ));
        }

        Ok(out)
    }

    /// Resolve a VindexRef to a concrete path.
    fn resolve_vindex_ref(&self, vref: &VindexRef) -> Result<PathBuf, LqlError> {
        match vref {
            VindexRef::Current => match &self.backend {
                Backend::Vindex { path, .. } => Ok(path.clone()),
                Backend::Weight { model_id, .. } => Err(LqlError::Execution(format!(
                    "CURRENT refers to a live model, not a vindex. Extract first:\n  \
                     EXTRACT MODEL \"{}\" INTO \"{}.vindex\"",
                    model_id,
                    model_id.split('/').last().unwrap_or(model_id),
                ))),
                _ => Err(LqlError::NoBackend),
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

impl larql_vindex::IndexBuildCallbacks for LqlBuildCallbacks {
    fn on_stage(&mut self, stage: &str) {
        self.current_stage = stage.to_string();
        self.messages.push(format!("  Stage: {stage}"));
    }

    fn on_stage_done(&mut self, stage: &str, elapsed_ms: f64) {
        self.messages.push(format!("  {stage}: {elapsed_ms:.0}ms"));
    }
}
