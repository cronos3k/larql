//! Lifecycle executor: USE, STATS, EXTRACT, COMPILE, DIFF

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
                        .map_err(|e| LqlError::exec("HuggingFace download failed", e))?
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
                    .map_err(|e| LqlError::exec("failed to load vindex config", e))?;

                let mut cb = larql_vindex::SilentLoadCallbacks;
                let index = larql_vindex::VectorIndex::load_vindex(&path, &mut cb)
                    .map_err(|e| LqlError::exec("failed to load vindex", e))?;

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
                let mut patched = larql_vindex::PatchedVindex::new(index);

                // Load KNN store if present (Architecture B)
                let knn_path = path.join("knn_store.bin");
                if knn_path.exists() {
                    match larql_vindex::KnnStore::load(&knn_path) {
                        Ok(store) => {
                            patched.knn_store = store;
                        }
                        Err(e) => {
                            eprintln!("warning: failed to load knn_store.bin: {e}");
                        }
                    }
                }

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
                    .map_err(|e| LqlError::exec("failed to resolve model", e))?;
                let weights = larql_inference::load_model_dir(&model_path)
                    .map_err(|e| LqlError::exec("failed to load model", e))?;
                let tokenizer = larql_inference::load_tokenizer(&model_path)
                    .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;

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
                out.push("Backend:         live weights (no vindex)".to_string());
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
            .map_err(|e| LqlError::exec("failed to load model", e))?;

        out.push(format!(
            "Model loaded ({} layers, hidden={}). Extracting to {}...",
            inference_model.num_layers(),
            inference_model.hidden_size(),
            output_dir.display()
        ));

        std::fs::create_dir_all(&output_dir)
            .map_err(|e| LqlError::exec("failed to create output dir", e))?;

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
        .map_err(|e| LqlError::exec("extraction failed", e))?;

        out.extend(callbacks.messages);
        out.push(format!("Extraction complete: {}", output_dir.display()));

        // Auto-load the newly created vindex
        let config = larql_vindex::load_vindex_config(&output_dir)
            .map_err(|e| LqlError::exec("failed to load vindex config", e))?;
        let mut cb = larql_vindex::SilentLoadCallbacks;
        let index = larql_vindex::VectorIndex::load_vindex(&output_dir, &mut cb)
            .map_err(|e| LqlError::exec("failed to load vindex", e))?;
        let relation_classifier = RelationClassifier::from_vindex(&output_dir);

        let total_features: usize = config.layers.iter().map(|l| l.num_features).sum();
        out.push(format!(
            "Using: {} ({} layers, {} features)",
            output_dir.display(),
            config.num_layers,
            format_number(total_features),
        ));

        let router = larql_vindex::RouterIndex::load(&output_dir, &config);
        let mut patched = larql_vindex::PatchedVindex::new(index);

        // Load KNN store if present (Architecture B)
        let knn_path = output_dir.join("knn_store.bin");
        if knn_path.exists() {
            if let Ok(store) = larql_vindex::KnnStore::load(&knn_path) {
                patched.knn_store = store;
            }
        }

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

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn exec_compile(
        &mut self,
        vindex: &VindexRef,
        output: &str,
        _format: Option<OutputFormat>,
        target: CompileTarget,
        on_conflict: Option<CompileConflict>,
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
            CompileTarget::Vindex => self.exec_compile_into_vindex(
                &vindex_path,
                output,
                on_conflict.unwrap_or(CompileConflict::LastWins),
            ),
            CompileTarget::Model => self.exec_compile_into_model(&vindex_path, output),
        }
    }

    fn exec_compile_into_model(
        &self,
        vindex_path: &std::path::Path,
        output: &str,
    ) -> Result<Vec<String>, LqlError> {
        let config = larql_vindex::load_vindex_config(vindex_path)
            .map_err(|e| LqlError::exec("failed to load vindex config", e))?;

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
            .map_err(|e| LqlError::exec("failed to create output dir", e))?;

        let mut cb = larql_vindex::SilentLoadCallbacks;
        let mut weights = larql_vindex::load_model_weights(vindex_path, &mut cb)
            .map_err(|e| LqlError::exec("failed to load model weights", e))?;

        // ── MEMIT: compile patch overlay into W_down edits ──
        //
        // Extract INSERT facts from the patch overlay, build MEMIT
        // fact descriptors, run the closed-form solve, and apply ΔW
        // to the loaded model weights before writing.
        let (_, _, patched) = self.require_vindex()?;
        let memit_facts = collect_memit_facts(patched, vindex_path)?;

        let mut out = Vec::new();
        if !memit_facts.is_empty() {
            let tokenizer = larql_vindex::load_vindex_tokenizer(vindex_path)
                .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;

            // MEMIT parameters — validated in Python reference
            // (experiments/15_v11_model/RESULTS.md §20).
            let ridge = 0.1;
            let target_alpha = 5.0;

            out.push(format!(
                "MEMIT: {} fact(s) across {} layer(s)",
                memit_facts.len(),
                memit_facts.iter()
                    .map(|f| f.layer)
                    .collect::<std::collections::HashSet<_>>()
                    .len(),
            ));

            let results = larql_inference::run_memit(
                &weights,
                &memit_facts,
                ridge,
                target_alpha,
                &tokenizer,
            ).map_err(|e| LqlError::Execution(format!("MEMIT failed: {e}")))?;

            for result in &results {
                let delta_norm: f32 = result.delta_w.iter()
                    .map(|v| v * v)
                    .sum::<f32>()
                    .sqrt();
                out.push(format!(
                    "  L{}: ΔW_down applied ({} facts, ‖ΔW‖={:.2})",
                    result.layer,
                    result.fact_results.len(),
                    delta_norm,
                ));

                // Apply ΔW to W_down at this layer.
                let down_key = weights.arch.ffn_down_key(result.layer);
                if let Some(w_down) = weights.tensors.get(&down_key) {
                    let updated = w_down.to_owned() + &result.delta_w;
                    weights.tensors.insert(
                        down_key,
                        larql_inference::ndarray::ArcArray::from(updated.into_shared()),
                    );
                }
            }
        }

        let mut build_cb = larql_vindex::SilentBuildCallbacks;
        larql_vindex::write_model_weights(&weights, &output_dir, &mut build_cb)
            .map_err(|e| LqlError::exec("failed to write model", e))?;

        let tok_src = vindex_path.join("tokenizer.json");
        let tok_dst = output_dir.join("tokenizer.json");
        if tok_src.exists() {
            std::fs::copy(&tok_src, &tok_dst)
                .map_err(|e| LqlError::exec("failed to copy tokenizer", e))?;
        }

        out.insert(0, format!("Compiled {} → {}", vindex_path.display(), output_dir.display()));
        out.push(format!("Model: {}", config.model));
        out.push(format!("Size: {}", format_bytes(dir_size(&output_dir))));
        Ok(out)
    }

    fn exec_compile_into_vindex(
        &mut self,
        source_path: &std::path::Path,
        output: &str,
        on_conflict: CompileConflict,
    ) -> Result<Vec<String>, LqlError> {
        let _ = source_path; // accepted for symmetry; current vindex is the source
        let output_dir = PathBuf::from(output);
        std::fs::create_dir_all(&output_dir)
            .map_err(|e| LqlError::exec("failed to create output dir", e))?;

        // Load the current vindex with patches applied
        let (path, config, patched) = self.require_vindex()?;

        // ── Conflict detection across applied patches ──
        //
        // The overlay maps in `PatchedVindex` are already collapsed under
        // last-wins semantics. To honour ON CONFLICT we re-scan the
        // ordered patch history and detect (layer, feature) slots that
        // are written by more than one patch.
        let collisions = collect_compile_collisions(&patched.patches);
        match on_conflict {
            CompileConflict::LastWins => {}
            CompileConflict::Fail => {
                if !collisions.is_empty() {
                    let preview = collisions.iter()
                        .take(5)
                        .map(|((l, f), n)| format!("L{l}/F{f} ({n} writes)"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    return Err(LqlError::Execution(format!(
                        "COMPILE INTO VINDEX ON CONFLICT FAIL: {} colliding slot(s): {}",
                        collisions.len(), preview
                    )));
                }
            }
            CompileConflict::HighestConfidence => {
                // Down vectors are baked at INSERT time and stored on the
                // base vindex collapsed under last-wins, so re-resolving
                // them from raw patches would require regenerating the
                // synthesised vectors. We do not currently do that — the
                // strategy is accepted for forward compatibility but
                // behaves like LAST_WINS today. This is reported in the
                // output below so callers know.
            }
        }

        // ── Step 1: gate_vectors.bin and down_meta.bin ──
        //
        // Both are written from a clone of the patched base. The clone path
        // produces byte-identical output to the source for unchanged
        // layers, and we deliberately do NOT bake any inserted gate
        // vectors into gate_vectors.bin (see comment further down).
        let baked = patched.base().clone();
        let layer_infos = baked.save_gate_vectors(&output_dir)
            .map_err(|e| LqlError::exec("failed to save gate vectors", e))?;
        // We hard-link down_meta.bin from source (in the unchanging-file
        // loop below) rather than calling save_down_meta, because the
        // cloned base is in mmap mode and its heap-side `down_meta` is
        // empty — saving it would produce a 152-byte file with zero
        // features and break WALK / DESCRIBE / SHOW.
        let dm_count: usize = config
            .layers
            .iter()
            .map(|l| l.num_features)
            .sum();

        // ── Step 2: hard-link unchanging weight files from the source ──
        //
        // These files are byte-identical to the source (model weights and
        // related artefacts that INSERT does not touch). Hard-linking is
        // free on APFS — same inode, no disk cost, no copy time.
        //
        // We deliberately do NOT bake the inserted gate vectors into
        // gate_vectors.bin. The dense FFN inference path
        // (`walk_ffn_exact` / `walk_ffn_full_mmap`) reads gate scores
        // from this file and feeds them into the GEGLU activation.
        // Baking a norm-matched (~typical-magnitude) gate at the
        // inserted slot makes its dense activation moderate-to-large,
        // which combined with the override down vector blows up the
        // residual stream. Keeping the source weak gate at the inserted
        // slot keeps the activation small — exactly matching the
        // patched-session math, where the small activation × override
        // down vector accumulates across layers into a meaningful
        // constellation effect.
        //
        // The override is instead baked into `down_weights.bin` further
        // down (see Step 3): the dense FFN reads `W_down[:, slot]` from
        // model weights, and replacing those columns with the override
        // values gives `small_activation × poseidon_vector` per layer,
        // which is the exact behaviour the runtime patch overlay
        // produces.
        const UNCHANGING: &[&str] = &[
            "attn_weights.bin",
            "up_weights.bin",
            "norms.bin",
            "weight_manifest.json",
            "embeddings.bin",
            "tokenizer.json",
            "up_features.bin",
            "down_meta.bin",
            "down_features.bin",
        ];
        for name in UNCHANGING {
            let src = path.join(name);
            let dst = output_dir.join(name);
            if !src.exists() {
                continue;
            }
            let _ = std::fs::remove_file(&dst);
            if std::fs::hard_link(&src, &dst).is_err() {
                std::fs::copy(&src, &dst)
                    .map_err(|e| LqlError::exec("failed to link/copy {name}", e))?;
            }
        }

        // Label files (small, copy is fine).
        for name in &["relation_clusters.json", "feature_clusters.jsonl", "feature_labels.json"] {
            let src = path.join(name);
            let dst = output_dir.join(name);
            if src.exists() {
                let _ = std::fs::remove_file(&dst);
                let _ = std::fs::copy(&src, &dst);
            }
        }

        // ── Step 3: bake down vector overrides into down_weights.bin ──
        //
        // The dense FFN inference path reads `W_down[:, slot]` from
        // `down_weights.bin` (via `load_model_weights` →
        // `walk_ffn_exact`). Replacing the column at the inserted slot
        // with the override down vector makes the inserted feature fire
        // through the standard FFN path with no runtime overlay needed.
        //
        // This is what makes the compiled vindex truly self-contained
        // and what unblocks `COMPILE INTO MODEL FORMAT safetensors|gguf`
        // — those exporters read the same `down_weights.bin` via
        // `weight_manifest.json` and emit it as the canonical down
        // projection, so the constellation is already in the exported
        // model.
        let down_overrides = patched.down_overrides();
        let up_overrides = patched.up_overrides();
        // Collect gate overrides from the patch overlay into an owned
        // HashMap matching the shape `patch_gate_vectors` expects.
        let gate_overrides: HashMap<(usize, usize), Vec<f32>> = patched
            .overrides_gate_iter()
            .map(|(l, f, g)| ((l, f), g.to_vec()))
            .collect();

        let mut overrides_applied = 0usize;
        if down_overrides.is_empty() {
            // Pure structural compile — hard-link down_weights.bin too.
            let src = path.join("down_weights.bin");
            let dst = output_dir.join("down_weights.bin");
            if src.exists() {
                let _ = std::fs::remove_file(&dst);
                if std::fs::hard_link(&src, &dst).is_err() {
                    std::fs::copy(&src, &dst)
                        .map_err(|e| LqlError::exec("copy down_weights", e))?;
                }
            }
        } else {
            patch_down_weights(path, &output_dir, config, down_overrides)?;
            overrides_applied = down_overrides.len();
        }

        // ── Step 3b/3c: bake gate + up overlays into the compiled vindex ──
        //
        // The dense FFN in a freshly-loaded compiled vindex reads
        // gate and up from `gate_vectors.bin` / `up_features.bin`
        // directly (no patch overlay present in a cold session). If
        // we only bake down, the compiled INFER path computes
        // `silu(weak_source_gate · x) * (weak_source_up · x) *
        // baked_down` at our installed slots — a tiny activation
        // times the right down direction — which is invisible on
        // prompts the model already knows (Gemma's Paris beats a
        // weak baked down in that direction).
        //
        // Baking gate + up into the source files produces the same
        // math the patched session's `sparse_ffn_forward_with_full_overrides`
        // runs, turning the compiled vindex into a self-contained
        // copy of the patched state. Validated by `refine_demo`:
        // patched session = 10/10; compiled = 8/10 pre-fix because
        // gate/up were never baked.
        patch_gate_vectors(path, &output_dir, config, &gate_overrides)?;
        patch_up_weights(path, &output_dir, config, up_overrides)?;

        // ── Step 4: write updated config ──
        let mut new_config = config.clone();
        new_config.layers = layer_infos;
        new_config.checksums = larql_vindex::format::checksums::compute_checksums(&output_dir).ok();
        larql_vindex::VectorIndex::save_config(&new_config, &output_dir)
            .map_err(|e| LqlError::exec("failed to save config", e))?;

        // ── Step 5: serialize KNN store (Architecture B) ──
        let knn_count = patched.knn_store.len();
        if knn_count > 0 {
            patched.knn_store.save(&output_dir.join("knn_store.bin"))
                .map_err(|e| LqlError::exec("failed to save knn_store", e))?;
        }

        let mut out = Vec::new();
        out.push(format!("Compiled {} → {}", source_path.display(), output_dir.display()));
        out.push(format!("Features: {}", dm_count));
        if !collisions.is_empty() {
            let strategy = match on_conflict {
                CompileConflict::LastWins => "LAST_WINS",
                CompileConflict::HighestConfidence => "HIGHEST_CONFIDENCE (resolves like LAST_WINS for down vectors — see docs)",
                CompileConflict::Fail => "FAIL",
            };
            out.push(format!(
                "Conflicts: {} slot(s) touched by multiple patches — strategy: {}",
                collisions.len(), strategy,
            ));
        }
        if overrides_applied > 0 {
            out.push(format!(
                "Down overrides baked: {} ({} layers touched)",
                overrides_applied,
                down_overrides.keys().map(|(l, _)| *l).collect::<std::collections::HashSet<_>>().len(),
            ));
        }
        if knn_count > 0 {
            out.push(format!("KNN store: {} entries", knn_count));
        }
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
            .map_err(|e| LqlError::exec(&format!("failed to load {}", path_a.display()), e))?;
        let index_b = larql_vindex::VectorIndex::load_vindex(&path_b, &mut cb)
            .map_err(|e| LqlError::exec(&format!("failed to load {}", path_b.display()), e))?;

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
                .map_err(|e| LqlError::exec("failed to save patch", e))?;
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
                    model_id.split('/').next_back().unwrap_or(model_id),
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

// ── COMPILE INTO MODEL: collect MEMIT facts from patch overlay ──

/// Extract INSERT operations from the patch overlay and convert them
/// into `MemitFact` descriptors for the MEMIT weight-edit solve.
///
/// Each INSERT carries entity, relation, target, and layer. We
/// synthesise a canonical prompt ("The {relation} of {entity} is"),
/// tokenise it (with BOS), and look up the target token ID — the
/// same approach as the INSERT executor in mutation.rs.
fn collect_memit_facts(
    patched: &larql_vindex::PatchedVindex,
    vindex_path: &std::path::Path,
) -> Result<Vec<larql_inference::MemitFact>, crate::error::LqlError> {
    let tokenizer = larql_vindex::load_vindex_tokenizer(vindex_path)
        .map_err(|e| crate::error::LqlError::exec("load tokenizer for MEMIT", e))?;

    let mut facts = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for patch in &patched.patches {
        for op in &patch.operations {
            if let larql_vindex::PatchOp::Insert { layer, entity, relation, target, .. } = op {
                let rel_str = relation.as_deref().unwrap_or("relation");
                let key = (entity.clone(), rel_str.to_string(), target.clone(), *layer);
                if !seen.insert(key) {
                    continue; // deduplicate
                }

                let rel_words = rel_str.replace(['-', '_'], " ");
                let prompt = format!("The {rel_words} of {entity} is");
                let encoding = tokenizer.encode(prompt.as_str(), true)
                    .map_err(|e| crate::error::LqlError::exec("tokenize MEMIT prompt", e))?;
                let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();

                // Target: first token of " " + target (matches INSERT semantics)
                let spaced = format!(" {target}");
                let target_encoding = tokenizer.encode(spaced.as_str(), false)
                    .map_err(|e| crate::error::LqlError::exec("tokenize MEMIT target", e))?;
                let target_token_id = target_encoding
                    .get_ids()
                    .first()
                    .copied()
                    .unwrap_or(0);

                facts.push(larql_inference::MemitFact {
                    prompt_tokens,
                    target_token_id,
                    layer: *layer,
                    label: format!("{entity} → {target} (L{layer})"),
                });
            }
        }
    }

    Ok(facts)
}

// ── COMPILE INTO VINDEX: bake down vector overrides into down_weights.bin ──
//
// The inserted features' down vectors live in
// `patched.base().down_overrides` (a HashMap populated by INSERT). To
// produce a self-contained vindex with no overlay needed, we copy the
// source `down_weights.bin` and rewrite the columns at the inserted
// feature slots with the override values.
//
// File layout: per layer `[hidden, intermediate]` row-major (f16 or f32).
// Feature `f`'s down vector is the *column* at index `f`, scattered
// across `hidden_size` rows. We read each affected layer slab into RAM,
// splice the override columns, and write the slab back. Each layer with
// overrides is one read + one write of `hidden * intermediate * dtype_bytes`
// (~100 MB for Gemma 4B).
//
// This approach is what makes the compiled vindex truly fresh: the
// dense FFN inference path reads `down_weights.bin` via
// `load_model_weights`, the bytes contain the override, and INFER works
// with no patch overlay. The same `down_weights.bin` is what
// `COMPILE INTO MODEL FORMAT safetensors|gguf` exports via
// `weight_manifest.json`, so the constellation is automatically present
// in the exported model file.
//
// We deliberately do NOT touch `gate_vectors.bin`. The dense FFN reads
// gate scores from that file and a norm-matched override gate would
// produce a moderate activation that — combined with the modified down
// column — blows up the residual. Keeping the source's weak free-slot
// gate at the inserted index keeps the activation small, exactly
// reproducing the patched-session math where small activation × override
// down vector accumulates across 8 layers into the constellation effect.

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};

/// Walk the ordered patch history and return the (layer, feature) slots
/// touched by more than one patch, along with the write count. Used by
/// `COMPILE INTO VINDEX ON CONFLICT` to detect ambiguous bakes.
pub(crate) fn collect_compile_collisions(
    patches: &[larql_vindex::VindexPatch],
) -> HashMap<(usize, usize), usize> {
    let mut counts: HashMap<(usize, usize), usize> = HashMap::new();
    for patch in patches {
        let mut seen_in_this_patch: std::collections::HashSet<(usize, usize)> =
            std::collections::HashSet::new();
        for op in &patch.operations {
            let key = match op.key() {
                Some(k) => k,
                None => continue, // KNN ops don't collide on (layer, feature)
            };
            if seen_in_this_patch.insert(key) {
                *counts.entry(key).or_insert(0) += 1;
            }
        }
    }
    counts.retain(|_, n| *n > 1);
    counts
}

fn copy_for_patch(src: &std::path::Path, dst: &std::path::Path) -> Result<(), LqlError> {
    let _ = std::fs::remove_file(dst);
    std::fs::copy(src, dst)
        .map_err(|e| LqlError::exec(&format!("failed to copy {}", src.display()), e))?;
    Ok(())
}

/// Bake down overrides into `down_weights.bin` (per-layer
/// `[hidden, intermediate]` row-major, may be f16 or f32).
fn patch_down_weights(
    source_dir: &std::path::Path,
    dest_dir: &std::path::Path,
    config: &larql_vindex::VindexConfig,
    overrides: &HashMap<(usize, usize), Vec<f32>>,
) -> Result<(), LqlError> {
    let src = source_dir.join("down_weights.bin");
    let dst = dest_dir.join("down_weights.bin");
    if !src.exists() {
        return Err(LqlError::Execution(
            "source vindex has no down_weights.bin — cannot bake overrides".into(),
        ));
    }

    copy_for_patch(&src, &dst)?;

    let total = std::fs::metadata(&dst)
        .map_err(|e| LqlError::exec("stat down_weights.bin", e))?
        .len() as usize;

    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let num_layers = config.num_layers;
    let elements_per_layer = hidden * intermediate;
    let total_elements = num_layers * elements_per_layer;

    let dtype_bytes: usize = if total == total_elements * 4 {
        4
    } else if total == total_elements * 2 {
        2
    } else {
        return Err(LqlError::Execution(format!(
            "down_weights.bin size {total} matches neither f32 ({}) nor f16 ({})",
            total_elements * 4,
            total_elements * 2
        )));
    };

    let layer_bytes = elements_per_layer * dtype_bytes;

    // Group overrides by layer so we only touch each layer's slab once.
    let mut by_layer: HashMap<usize, Vec<(usize, &Vec<f32>)>> = HashMap::new();
    for ((l, f), v) in overrides {
        by_layer.entry(*l).or_default().push((*f, v));
    }

    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&dst)
        .map_err(|e| LqlError::exec("open down_weights.bin", e))?;

    let mut buf = vec![0u8; layer_bytes];

    for (layer, layer_overrides) in by_layer {
        let layer_offset = (layer * layer_bytes) as u64;
        file.seek(SeekFrom::Start(layer_offset))
            .map_err(|e| LqlError::exec("seek down_weights", e))?;
        file.read_exact(&mut buf)
            .map_err(|e| LqlError::exec("read down_weights slab", e))?;

        for (feature, down_vec) in layer_overrides {
            if down_vec.len() != hidden {
                return Err(LqlError::Execution(format!(
                    "down override at L{layer} F{feature} has wrong shape: {} (expected {hidden})",
                    down_vec.len()
                )));
            }
            // Splice the column for `feature` across all `hidden` rows.
            for (row, val) in down_vec.iter().enumerate() {
                let cell = (row * intermediate + feature) * dtype_bytes;
                if dtype_bytes == 4 {
                    buf[cell..cell + 4].copy_from_slice(&val.to_le_bytes());
                } else {
                    let half_bits: u16 = larql_models::quant::half::f32_to_f16(*val);
                    buf[cell..cell + 2].copy_from_slice(&half_bits.to_le_bytes());
                }
            }
        }

        file.seek(SeekFrom::Start(layer_offset))
            .map_err(|e| LqlError::exec("seek down_weights", e))?;
        file.write_all(&buf)
            .map_err(|e| LqlError::exec("write down_weights slab", e))?;
    }
    Ok(())
}

/// Bake gate overlay entries into `gate_vectors.bin`. File layout
/// follows the per-layer `VindexLayerInfo` records in `config.layers`:
///
/// - dtype from `config.dtype` (may be f16 or f32)
/// - each layer has an explicit byte `offset` and `length` — layers
///   are NOT necessarily contiguous or in `layer` order within the
///   array. Writing at a naive `layer_index × layer_bytes` offset
///   lands in the wrong slice and corrupts whichever layer actually
///   lives at that byte position, which wrecks inference across the
///   whole file (validated by `refine_demo22`: the naive offsets
///   collapsed compiled-session retrieval from 8/10 to 0/10).
///
/// Within a layer, feature `f`'s gate is the row at
/// `info.offset + f × hidden × bpf` — contiguous per-feature.
fn patch_gate_vectors(
    source_dir: &std::path::Path,
    dest_dir: &std::path::Path,
    config: &larql_vindex::VindexConfig,
    gate_overrides: &HashMap<(usize, usize), Vec<f32>>,
) -> Result<(), LqlError> {
    if gate_overrides.is_empty() {
        return Ok(());
    }
    let src = source_dir.join("gate_vectors.bin");
    let dst = dest_dir.join("gate_vectors.bin");
    if !src.exists() {
        return Err(LqlError::Execution(
            "source vindex has no gate_vectors.bin — cannot bake gate overrides".into(),
        ));
    }

    // `dst` was hard-linked from the source earlier in the compile
    // bake's unchanging-files loop, so we need a real copy we own
    // before seek-writing into it.
    copy_for_patch(&src, &dst)?;

    let hidden = config.hidden_size;
    let bpf = larql_vindex::config::dtype::bytes_per_float(config.dtype);

    // Map layer → LayerInfo. Layers that don't appear in config.layers
    // have no gate data in the file (e.g. embedding-only layers) and
    // any override targeting them is a bug — we error out clearly.
    let mut layer_info: HashMap<usize, (u64, usize)> = HashMap::new();
    for info in &config.layers {
        layer_info.insert(info.layer, (info.offset, info.num_features));
    }

    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&dst)
        .map_err(|e| LqlError::exec("open gate_vectors.bin", e))?;

    let row_bytes = hidden * bpf;
    let mut row_buf = vec![0u8; row_bytes];

    for ((layer, feature), gate_vec) in gate_overrides {
        if gate_vec.len() != hidden {
            return Err(LqlError::Execution(format!(
                "gate override at L{layer} F{feature} has wrong shape: {} (expected {hidden})",
                gate_vec.len()
            )));
        }
        let Some(&(layer_offset, nf)) = layer_info.get(layer) else {
            return Err(LqlError::Execution(format!(
                "gate override at L{layer} F{feature}: layer {layer} not in config.layers \
                 (source vindex has no gate data for this layer)"
            )));
        };
        if *feature >= nf {
            return Err(LqlError::Execution(format!(
                "gate override at L{layer} F{feature} out of range (layer has {nf} features)"
            )));
        }

        // Encode the gate row to the file's native dtype.
        if bpf == 4 {
            for (i, v) in gate_vec.iter().enumerate() {
                row_buf[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
            }
        } else if bpf == 2 {
            for (i, v) in gate_vec.iter().enumerate() {
                let half_bits = larql_models::quant::half::f32_to_f16(*v);
                row_buf[i * 2..(i + 1) * 2].copy_from_slice(&half_bits.to_le_bytes());
            }
        } else {
            return Err(LqlError::Execution(format!(
                "unsupported gate_vectors.bin dtype: bpf={bpf}",
            )));
        }

        let feature_offset = layer_offset + (*feature * row_bytes) as u64;
        file.seek(SeekFrom::Start(feature_offset))
            .map_err(|e| LqlError::exec("seek gate_vectors", e))?;
        file.write_all(&row_buf)
            .map_err(|e| LqlError::exec("write gate_vectors row", e))?;
    }
    Ok(())
}

/// Bake up overlay entries into `up_weights.bin`. Dense FFN at
/// inference time reads this file via `load_model_weights`, which
/// consults `weight_manifest.json` to find each tensor's `(file,
/// offset, length, shape)` entry.
///
/// The layout is:
/// - the file the manifest points to (normally `up_weights.bin`, but
///   could be different if the extract pipeline changes)
/// - per-layer tensor at `entry.offset` with `entry.length` bytes
/// - dtype inferred from `byte_count / expected_floats` (4 = f32,
///   2 = f16), matching the loader at `weights.rs:534-541`
/// - shape is `[num_features, hidden_size]`, row-major; feature `f`'s
///   row starts at `entry.offset + f × hidden × bpf`
///
/// We DO NOT touch `up_features.bin` (which is a separate
/// feature-major f32 file used only by `walk_ffn_sparse`, typically
/// absent from vindexes that ship with `up_weights.bin`). Writing to
/// the wrong file was the root cause of `refine_demo22`'s regression
/// from 8/10 to 0/10 compiled retrieval.
fn patch_up_weights(
    source_dir: &std::path::Path,
    dest_dir: &std::path::Path,
    config: &larql_vindex::VindexConfig,
    up_overrides: &HashMap<(usize, usize), Vec<f32>>,
) -> Result<(), LqlError> {
    if up_overrides.is_empty() {
        return Ok(());
    }

    // Read the weight manifest from the SOURCE vindex — the dest copy
    // was hard-linked from source and we haven't modified the manifest.
    let manifest_path = source_dir.join("weight_manifest.json");
    if !manifest_path.exists() {
        // Manifestless vindex — we can't safely locate the up tensors.
        // Log and skip. The compiled vindex will still have baked
        // down_weights.bin and overlay gates in gate_vectors.bin, so
        // the install is at least partially live.
        return Ok(());
    }
    let manifest_text = std::fs::read_to_string(&manifest_path)
        .map_err(|e| LqlError::exec("read weight_manifest.json", e))?;
    let entries: Vec<serde_json::Value> = serde_json::from_str(&manifest_text)
        .map_err(|e| LqlError::exec("parse weight_manifest.json", e))?;

    // Build `layer → (file, offset, length)` lookup for the up_proj
    // tensor at each layer by pattern-matching the manifest key. We
    // don't resolve the full arch here — we just look for entries
    // whose key contains `layers.{L}.` AND `up_proj`, which works
    // for every Llama/Gemma/Mistral-family vindex that writes to
    // `up_weights.bin`. MoE experts or architectures with different
    // key conventions will simply not match and the overlay for
    // those layers is silently skipped.
    let mut layer_up_lookup: HashMap<usize, (String, u64, u64)> = HashMap::new();
    for entry in &entries {
        let Some(key) = entry.get("key").and_then(|v| v.as_str()) else { continue };
        if !key.contains("up_proj") {
            continue;
        }
        let Some(file) = entry.get("file").and_then(|v| v.as_str()) else { continue };
        let Some(offset) = entry.get("offset").and_then(|v| v.as_u64()) else { continue };
        let Some(length) = entry.get("length").and_then(|v| v.as_u64()) else { continue };
        // Extract the layer number from the key: the segment after
        // `layers.` and before the next `.`.
        let Some(rest) = key.split("layers.").nth(1) else { continue };
        let Some(layer_str) = rest.split('.').next() else { continue };
        let Ok(layer) = layer_str.parse::<usize>() else { continue };
        layer_up_lookup.insert(layer, (file.to_string(), offset, length));
    }

    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    // Row-major tensor is [num_features, hidden], so feature f starts
    // at `offset + f * hidden * bpf`. Expected per-tensor byte count
    // is `num_features * hidden * bpf` — detect bpf from that.
    let expected_floats = intermediate * hidden;

    // File handles are cached per file so we don't re-open for each
    // (layer, feature) write.
    let mut file_cache: HashMap<String, std::fs::File> = HashMap::new();

    for ((layer, feature), up_vec) in up_overrides {
        if up_vec.len() != hidden {
            return Err(LqlError::Execution(format!(
                "up override at L{layer} F{feature} has wrong shape: {} (expected {hidden})",
                up_vec.len()
            )));
        }
        if *feature >= intermediate {
            return Err(LqlError::Execution(format!(
                "up override at L{layer} F{feature} out of range (intermediate = {intermediate})"
            )));
        }

        let Some((file_name, offset, length)) = layer_up_lookup.get(layer) else {
            // No manifest entry for this layer's up projection —
            // skip silently, the layer's up is not materialised.
            continue;
        };

        let bpf = if *length as usize == expected_floats * 4 {
            4
        } else if *length as usize == expected_floats * 2 {
            2
        } else {
            return Err(LqlError::Execution(format!(
                "up weight for L{layer} has length {length} ≠ \
                 expected {} (f32) or {} (f16)",
                expected_floats * 4,
                expected_floats * 2,
            )));
        };

        // Lazily open + copy the file if we haven't touched it yet.
        if !file_cache.contains_key(file_name) {
            let src = source_dir.join(file_name);
            let dst = dest_dir.join(file_name);
            if !src.exists() {
                return Err(LqlError::Execution(format!(
                    "weight file {file_name} referenced by manifest but missing from source"
                )));
            }
            copy_for_patch(&src, &dst)?;
            let f = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&dst)
                .map_err(|e| LqlError::exec(&format!("open {file_name}"), e))?;
            file_cache.insert(file_name.clone(), f);
        }
        let file = file_cache.get_mut(file_name).unwrap();

        let row_bytes = hidden * bpf;
        let mut row_buf = vec![0u8; row_bytes];
        if bpf == 4 {
            for (i, v) in up_vec.iter().enumerate() {
                row_buf[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
            }
        } else {
            for (i, v) in up_vec.iter().enumerate() {
                let half_bits = larql_models::quant::half::f32_to_f16(*v);
                row_buf[i * 2..(i + 1) * 2].copy_from_slice(&half_bits.to_le_bytes());
            }
        }

        let feature_offset = offset + (*feature * row_bytes) as u64;
        file.seek(SeekFrom::Start(feature_offset))
            .map_err(|e| LqlError::exec(&format!("seek {file_name}"), e))?;
        file.write_all(&row_buf)
            .map_err(|e| LqlError::exec(&format!("write {file_name} row"), e))?;
    }
    Ok(())
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

#[cfg(test)]
mod compile_into_vindex_tests {
    //! Unit tests for the `COMPILE INTO VINDEX` byte-level baking helper.
    //!
    //! These build a tiny synthetic `down_weights.bin` file with known
    //! contents, run `patch_down_weights` against it, then verify that the
    //! override columns were spliced into the correct cells (and *only*
    //! those cells) without disturbing any other bytes.
    //!
    //! No real vindex required — these run in CI with no model on disk.
    use super::*;
    use std::collections::HashMap;
    use larql_vindex::{PatchOp, VindexPatch};

    fn make_patch(ops: Vec<PatchOp>) -> VindexPatch {
        VindexPatch {
            version: 1,
            base_model: String::new(),
            base_checksum: None,
            created_at: String::new(),
            description: None,
            author: None,
            tags: Vec::new(),
            operations: ops,
        }
    }

    fn insert_op(layer: usize, feature: usize) -> PatchOp {
        PatchOp::Insert {
            layer,
            feature,
            relation: None,
            entity: "e".into(),
            target: "t".into(),
            confidence: Some(0.9),
            gate_vector_b64: None,
            down_meta: None,
        }
    }

    #[test]
    fn collisions_empty_when_each_slot_unique() {
        let patches = vec![
            make_patch(vec![insert_op(1, 10)]),
            make_patch(vec![insert_op(2, 20)]),
        ];
        assert!(collect_compile_collisions(&patches).is_empty());
    }

    #[test]
    fn collisions_detect_same_slot_in_two_patches() {
        let patches = vec![
            make_patch(vec![insert_op(1, 10)]),
            make_patch(vec![insert_op(1, 10)]),
        ];
        let c = collect_compile_collisions(&patches);
        assert_eq!(c.get(&(1, 10)), Some(&2));
    }

    #[test]
    fn collisions_ignore_repeats_within_one_patch() {
        let patches = vec![
            make_patch(vec![insert_op(1, 10), insert_op(1, 10)]),
        ];
        assert!(collect_compile_collisions(&patches).is_empty());
    }


    /// Build a minimal `VindexConfig` shaped for these tests.
    /// Only the dimensions matter for `patch_down_weights`; everything
    /// else is dummy.
    fn mini_config(num_layers: usize, hidden: usize, intermediate: usize) -> larql_vindex::VindexConfig {
        larql_vindex::VindexConfig {
            version: 1,
            model: "test".into(),
            family: "test".into(),
            source: None,
            checksums: None,
            num_layers,
            hidden_size: hidden,
            intermediate_size: intermediate,
            vocab_size: 32,
            embed_scale: 1.0,
            extract_level: larql_vindex::ExtractLevel::All,
            dtype: larql_vindex::config::dtype::StorageDtype::F32,
            layer_bands: None,
            layers: Vec::new(),
            down_top_k: 10,
            has_model_weights: true,
            model_config: None,
        }
    }

    /// Write `num_layers * hidden * intermediate` floats to a fake
    /// `down_weights.bin` in the given directory. Each cell is set to a
    /// deterministic pattern so we can later assert which bytes the patch
    /// touched.
    fn write_synthetic_f32(
        dir: &std::path::Path,
        num_layers: usize,
        hidden: usize,
        intermediate: usize,
    ) {
        let total = num_layers * hidden * intermediate;
        let mut bytes: Vec<u8> = Vec::with_capacity(total * 4);
        for i in 0..total {
            // Distinctive sentinel: small positive floats indexed by element.
            let v = (i as f32) * 0.001;
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(dir.join("down_weights.bin"), &bytes).unwrap();
    }

    fn write_synthetic_f16(
        dir: &std::path::Path,
        num_layers: usize,
        hidden: usize,
        intermediate: usize,
    ) {
        let total = num_layers * hidden * intermediate;
        let mut bytes: Vec<u8> = Vec::with_capacity(total * 2);
        for i in 0..total {
            let v = (i as f32) * 0.001;
            let half_bits = larql_models::quant::half::f32_to_f16(v);
            bytes.extend_from_slice(&half_bits.to_le_bytes());
        }
        std::fs::write(dir.join("down_weights.bin"), &bytes).unwrap();
    }

    /// Read all elements at the column for `feature` in layer `layer` from
    /// an f32 down_weights.bin (the patched copy). Returns a Vec of length
    /// `hidden`.
    fn read_column_f32(
        dir: &std::path::Path,
        layer: usize,
        feature: usize,
        num_layers: usize,
        hidden: usize,
        intermediate: usize,
    ) -> Vec<f32> {
        let bytes = std::fs::read(dir.join("down_weights.bin")).unwrap();
        let layer_elems = hidden * intermediate;
        let mut out = Vec::with_capacity(hidden);
        for row in 0..hidden {
            let cell = (layer * layer_elems + row * intermediate + feature) * 4;
            out.push(f32::from_le_bytes(bytes[cell..cell + 4].try_into().unwrap()));
        }
        let _ = num_layers; // unused but documents the layout
        out
    }

    fn read_column_f16(
        dir: &std::path::Path,
        layer: usize,
        feature: usize,
        hidden: usize,
        intermediate: usize,
    ) -> Vec<f32> {
        let bytes = std::fs::read(dir.join("down_weights.bin")).unwrap();
        let layer_elems = hidden * intermediate;
        let mut out = Vec::with_capacity(hidden);
        for row in 0..hidden {
            let cell = (layer * layer_elems + row * intermediate + feature) * 2;
            let bits = u16::from_le_bytes(bytes[cell..cell + 2].try_into().unwrap());
            out.push(larql_models::quant::half::f16_to_f32(bits));
        }
        out
    }

    #[test]
    fn patch_down_weights_f32_writes_correct_columns() {
        let tmp = std::env::temp_dir().join("larql_pdw_f32");
        let _ = std::fs::remove_dir_all(&tmp);
        let src = tmp.join("src");
        let dst = tmp.join("dst");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&dst).unwrap();

        let num_layers = 4;
        let hidden = 8;
        let intermediate = 16;
        write_synthetic_f32(&src, num_layers, hidden, intermediate);
        let cfg = mini_config(num_layers, hidden, intermediate);

        // Build override down vectors with distinctive values per layer.
        let mut overrides: HashMap<(usize, usize), Vec<f32>> = HashMap::new();
        let layer = 2;
        let feature = 5;
        let down: Vec<f32> = (0..hidden).map(|r| 100.0 + r as f32).collect();
        overrides.insert((layer, feature), down.clone());

        patch_down_weights(&src, &dst, &cfg, &overrides).unwrap();

        // The patched column at L2 F5 must equal the override exactly.
        let read_back = read_column_f32(&dst, layer, feature, num_layers, hidden, intermediate);
        assert_eq!(read_back, down, "patched column doesn't match override");

        // Layer 0 column 5 must be untouched (offset = row*intermediate + feature
        // since layer 0 starts at element 0 of the file).
        let untouched = read_column_f32(&dst, 0, feature, num_layers, hidden, intermediate);
        for (row, val) in untouched.iter().enumerate() {
            let expected = ((row * intermediate + feature) as f32) * 0.001;
            assert!(
                (val - expected).abs() < 1e-6,
                "L0 F5 row {row}: got {val}, expected {expected}"
            );
        }

        // Adjacent column at L2 F4 must be untouched.
        let neighbour = read_column_f32(&dst, layer, feature - 1, num_layers, hidden, intermediate);
        for (row, val) in neighbour.iter().enumerate() {
            let expected =
                ((layer * hidden * intermediate + row * intermediate + (feature - 1)) as f32) * 0.001;
            assert!(
                (val - expected).abs() < 1e-6,
                "L2 F4 row {row}: got {val}, expected {expected}"
            );
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn patch_down_weights_f16_writes_correct_columns() {
        let tmp = std::env::temp_dir().join("larql_pdw_f16");
        let _ = std::fs::remove_dir_all(&tmp);
        let src = tmp.join("src");
        let dst = tmp.join("dst");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&dst).unwrap();

        let num_layers = 3;
        let hidden = 8;
        let intermediate = 16;
        write_synthetic_f16(&src, num_layers, hidden, intermediate);
        let cfg = mini_config(num_layers, hidden, intermediate);

        let mut overrides: HashMap<(usize, usize), Vec<f32>> = HashMap::new();
        let down: Vec<f32> = (0..hidden).map(|r| (r as f32) * 0.5 - 1.0).collect();
        overrides.insert((1, 7), down.clone());

        patch_down_weights(&src, &dst, &cfg, &overrides).unwrap();

        let read_back = read_column_f16(&dst, 1, 7, hidden, intermediate);
        // f16 round-trip tolerance — values like 0.5 round-trip cleanly.
        for (i, (got, want)) in read_back.iter().zip(down.iter()).enumerate() {
            assert!(
                (got - want).abs() < 0.01,
                "row {i}: got {got}, expected {want}"
            );
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn patch_down_weights_multiple_layers_and_features() {
        let tmp = std::env::temp_dir().join("larql_pdw_multi");
        let _ = std::fs::remove_dir_all(&tmp);
        let src = tmp.join("src");
        let dst = tmp.join("dst");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&dst).unwrap();

        let num_layers = 8;
        let hidden = 4;
        let intermediate = 8;
        write_synthetic_f32(&src, num_layers, hidden, intermediate);
        let cfg = mini_config(num_layers, hidden, intermediate);

        // 4 different (layer, feature) pairs with different override values.
        let mut overrides: HashMap<(usize, usize), Vec<f32>> = HashMap::new();
        let cases = [(0, 0), (3, 5), (5, 2), (7, 7)];
        for (layer, feature) in cases {
            let v: Vec<f32> = (0..hidden)
                .map(|r| 1000.0 + (layer * 100 + feature * 10 + r) as f32)
                .collect();
            overrides.insert((layer, feature), v);
        }

        patch_down_weights(&src, &dst, &cfg, &overrides).unwrap();

        for (layer, feature) in cases {
            let read_back = read_column_f32(&dst, layer, feature, num_layers, hidden, intermediate);
            let expected: Vec<f32> = (0..hidden)
                .map(|r| 1000.0 + (layer * 100 + feature * 10 + r) as f32)
                .collect();
            assert_eq!(
                read_back, expected,
                "L{layer} F{feature} doesn't match override"
            );
        }

        // Spot check a non-overridden cell at L3 F0 — must equal source.
        let untouched = read_column_f32(&dst, 3, 0, num_layers, hidden, intermediate);
        for (row, val) in untouched.iter().enumerate() {
            let expected = ((3 * hidden * intermediate + row * intermediate) as f32) * 0.001;
            assert!((val - expected).abs() < 1e-6, "L3 F0 row {row} disturbed");
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn patch_down_weights_rejects_wrong_shape() {
        let tmp = std::env::temp_dir().join("larql_pdw_bad");
        let _ = std::fs::remove_dir_all(&tmp);
        let src = tmp.join("src");
        let dst = tmp.join("dst");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&dst).unwrap();

        let cfg = mini_config(2, 8, 8);
        write_synthetic_f32(&src, 2, 8, 8);

        let mut overrides: HashMap<(usize, usize), Vec<f32>> = HashMap::new();
        // Wrong length: 4 instead of 8.
        overrides.insert((0, 0), vec![0.0; 4]);

        let result = patch_down_weights(&src, &dst, &cfg, &overrides);
        assert!(result.is_err(), "expected wrong-shape override to error");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("wrong shape"), "error message: {msg}");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn patch_down_weights_rejects_unrecognised_dtype_size() {
        let tmp = std::env::temp_dir().join("larql_pdw_dtype");
        let _ = std::fs::remove_dir_all(&tmp);
        let src = tmp.join("src");
        let dst = tmp.join("dst");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&dst).unwrap();

        let cfg = mini_config(2, 4, 4);
        // Write a file whose size matches neither f32 (128 bytes) nor f16 (64 bytes).
        std::fs::write(src.join("down_weights.bin"), vec![0u8; 100]).unwrap();

        let mut overrides: HashMap<(usize, usize), Vec<f32>> = HashMap::new();
        overrides.insert((0, 0), vec![1.0; 4]);

        let result = patch_down_weights(&src, &dst, &cfg, &overrides);
        assert!(result.is_err(), "expected mismatched dtype to error");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn patch_down_weights_missing_source_errors() {
        let tmp = std::env::temp_dir().join("larql_pdw_missing");
        let _ = std::fs::remove_dir_all(&tmp);
        let src = tmp.join("src");
        let dst = tmp.join("dst");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::create_dir_all(&dst).unwrap();

        // Note: src/down_weights.bin deliberately not created.

        let cfg = mini_config(2, 4, 4);
        let mut overrides: HashMap<(usize, usize), Vec<f32>> = HashMap::new();
        overrides.insert((0, 0), vec![1.0; 4]);

        let result = patch_down_weights(&src, &dst, &cfg, &overrides);
        assert!(result.is_err(), "expected missing source to error");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("no down_weights.bin"), "error message: {msg}");

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
