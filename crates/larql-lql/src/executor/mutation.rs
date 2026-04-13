//! Mutation executor: INSERT, DELETE, UPDATE, MERGE
//!
//! All mutations go through the PatchedVindex overlay.
//! Base vindex files on disk are never modified.

use std::path::PathBuf;

use crate::ast::*;
use crate::error::LqlError;
use super::{Backend, Session};

impl Session {
    // ── INSERT ──
    //
    // Adds an edge to the vindex via the patch overlay. Finds a free feature slot,
    // synthesises a gate vector from the entity embedding + relation cluster centre,
    // and records the operation for SAVE PATCH.

    pub(crate) fn exec_insert(
        &mut self,
        entity: &str,
        relation: &str,
        target: &str,
        layer_hint: Option<u32>,
        confidence: Option<f32>,
        _alpha_override: Option<f32>,
    ) -> Result<Vec<String>, LqlError> {
        // Architecture B: retrieval-override KNN store.
        //
        // Instead of synthesising gate/up/down vectors into an FFN slot
        // (Architecture A), we capture the model's residual at the
        // install layer for a canonical prompt and store it as a KNN
        // key alongside the target token. At inference time, the KNN
        // store is queried with the live residual — if cosine > threshold,
        // the target overrides the model's prediction.
        //
        // Port of Python `RetrievalVindex` from
        // experiments/15_v11_model/vindex_build_wordnet_b.py.
        // Validated at 25K edges, 87 edges/s, 100% same-prompt retrieval.

        // ── Phase 1: Read config, determine install layer ──
        let (install_layer, has_weights);
        {
            let (_path, config, _patched) = self.require_vindex()?;

            let bands = config.layer_bands.clone()
                .or_else(|| larql_vindex::LayerBands::for_family(&config.family, config.num_layers))
                .unwrap_or(larql_vindex::LayerBands {
                    syntax: (0, config.num_layers.saturating_sub(1)),
                    knowledge: (0, config.num_layers.saturating_sub(1)),
                    output: (0, config.num_layers.saturating_sub(1)),
                });

            install_layer = if let Some(l) = layer_hint {
                (l as usize).min(config.num_layers.saturating_sub(1))
            } else {
                bands.knowledge.1.saturating_sub(1)
                    .min(config.num_layers.saturating_sub(1))
            };

            has_weights = config.has_model_weights;
        }

        // ── Phase 2: Capture residual via forward pass ──
        let residual_key: Vec<f32>;
        let target_id: u32;

        if has_weights {
            let (path, _config, patched) = self.require_vindex()?;
            let mut cb = larql_vindex::SilentLoadCallbacks;
            let weights = larql_vindex::load_model_weights(path, &mut cb)
                .map_err(|e| LqlError::exec("failed to load weights", e))?;
            let tokenizer = larql_vindex::load_vindex_tokenizer(path)
                .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;

            // Encode target token (same " "+target first-token logic as before)
            let spaced_target = format!(" {target}");
            let target_encoding = tokenizer.encode(spaced_target.as_str(), false)
                .map_err(|e| LqlError::exec("tokenize error", e))?;
            target_id = target_encoding.get_ids().first().copied().unwrap_or(0);

            // Build canonical prompt and forward pass to capture residual
            let rel_words = relation.replace(['-', '_'], " ");
            let prompt = format!("The {rel_words} of {entity} is");
            let encoding = tokenizer.encode(prompt.as_str(), true)
                .map_err(|e| LqlError::exec("tokenize error", e))?;
            let token_ids: Vec<u32> = encoding.get_ids().to_vec();

            // Capture through BASE index with unlimited top_k (matches INFER)
            let walk_ffn = larql_inference::vindex::WalkFfn::new_unlimited_with_trace(
                &weights, patched.base(),
            );
            let _result = larql_inference::predict_with_ffn(
                &weights, &tokenizer, &token_ids, 1, &walk_ffn,
            );

            // Extract residual at install layer
            let residuals = walk_ffn.take_residuals();
            let captured = residuals.into_iter()
                .find(|(l, _)| *l == install_layer)
                .map(|(_, r)| r)
                .ok_or_else(|| LqlError::Execution(format!(
                    "no residual captured at layer {install_layer}"
                )))?;

            residual_key = captured;
        } else {
            // No model weights — use entity embedding as the key.
            // Less precise but allows INSERT on browse-only vindexes.
            let (path, _config, _patched) = self.require_vindex()?;
            let (embed, embed_scale) = larql_vindex::load_vindex_embeddings(path)
                .map_err(|e| LqlError::exec("failed to load embeddings", e))?;
            let tokenizer = larql_vindex::load_vindex_tokenizer(path)
                .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;

            let hidden = embed.shape()[1];

            // Target token
            let spaced_target = format!(" {target}");
            let target_encoding = tokenizer.encode(spaced_target.as_str(), false)
                .map_err(|e| LqlError::exec("tokenize error", e))?;
            target_id = target_encoding.get_ids().first().copied().unwrap_or(0);

            // Entity embedding as key
            let entity_encoding = tokenizer.encode(entity, false)
                .map_err(|e| LqlError::exec("tokenize error", e))?;
            let entity_ids: Vec<u32> = entity_encoding.get_ids().to_vec();
            let mut ev = vec![0.0f32; hidden];
            for &tok in &entity_ids {
                let row = embed.row(tok as usize);
                for j in 0..hidden { ev[j] += row[j] * embed_scale; }
            }
            let n = entity_ids.len().max(1) as f32;
            for v in &mut ev { *v /= n; }
            residual_key = ev;
        }

        // ── Phase 3: Store in KNN store ──
        let c_score = confidence.unwrap_or(1.0);
        let key_b64 = larql_vindex::patch::core::encode_gate_vector(&residual_key);

        {
            let (_path, _config, patched) = self.require_patched_mut()?;
            patched.knn_store.add(
                install_layer,
                residual_key,
                target_id,
                target.to_string(),
                entity.to_string(),
                relation.to_string(),
                c_score,
            );
        }

        // Record to patch session
        let patch_op = larql_vindex::PatchOp::InsertKnn {
            layer: install_layer,
            entity: entity.to_string(),
            relation: relation.to_string(),
            target: target.to_string(),
            target_id,
            confidence: Some(c_score),
            key_vector_b64: key_b64,
        };
        if let Some(ref mut recording) = self.patch_recording {
            recording.operations.push(patch_op);
        }

        let mut out = Vec::new();
        out.push(format!(
            "Inserted: {} —[{}]→ {} at L{} (KNN store)",
            entity, relation, target, install_layer,
        ));
        if has_weights {
            out.push("  mode: residual capture (Architecture B, retrieval-override)".into());
        } else {
            out.push("  mode: embedding key (no model weights)".into());
        }
        out.push(format!("  KNN store: {} entries total", {
            let (_, _, patched) = self.require_vindex()?;
            patched.knn_store.len()
        }));

        Ok(out)
    }

    // ── DELETE ──

    pub(crate) fn exec_delete(&mut self, conditions: &[Condition]) -> Result<Vec<String>, LqlError> {
        let layer_filter = conditions.iter().find(|c| c.field == "layer").and_then(|c| {
            if let Value::Integer(n) = c.value { Some(n as usize) } else { None }
        });
        let feature_filter = conditions.iter().find(|c| c.field == "feature").and_then(|c| {
            if let Value::Integer(n) = c.value { Some(n as usize) } else { None }
        });
        let entity_filter = conditions.iter().find(|c| c.field == "entity").and_then(|c| {
            if let Value::String(ref s) = c.value { Some(s.as_str()) } else { None }
        });

        // Collect deletions, then apply
        let deletes: Vec<(usize, usize)>;
        {
            let (_path, _config, patched) = self.require_patched_mut()?;

            if let (Some(layer), Some(feature)) = (layer_filter, feature_filter) {
                patched.delete_feature(layer, feature);
                deletes = vec![(layer, feature)];
            } else {
                let matches = patched.base().find_features(entity_filter, None, layer_filter);
                if matches.is_empty() {
                    return Ok(vec!["  (no matching features found)".into()]);
                }
                for &(layer, feature) in &matches {
                    patched.delete_feature(layer, feature);
                }
                deletes = matches;
            }
        }

        // Also remove from KNN store (Architecture B entries)
        let mut knn_removed = 0;
        if let Some(entity) = entity_filter {
            let (_path, _config, patched) = self.require_patched_mut()?;
            let before = patched.knn_store.len();
            patched.knn_store.remove_by_entity(entity);
            knn_removed = before - patched.knn_store.len();

            if knn_removed > 0 {
                if let Some(ref mut recording) = self.patch_recording {
                    recording.operations.push(larql_vindex::PatchOp::DeleteKnn {
                        entity: entity.to_string(),
                    });
                }
            }
        }

        // Record to patch session
        for &(layer, feature) in &deletes {
            if let Some(ref mut recording) = self.patch_recording {
                recording.operations.push(larql_vindex::PatchOp::Delete {
                    layer,
                    feature,
                    reason: None,
                });
            }
        }

        let _total = deletes.len() + knn_removed;
        let knn_note = if knn_removed > 0 {
            format!(" + {} KNN entries", knn_removed)
        } else {
            String::new()
        };
        Ok(vec![format!("Deleted {} features{} (patch overlay)", deletes.len(), knn_note)])
    }

    // ── UPDATE ──

    pub(crate) fn exec_update(
        &mut self,
        set: &[Assignment],
        conditions: &[Condition],
    ) -> Result<Vec<String>, LqlError> {
        let entity_filter = conditions.iter().find(|c| c.field == "entity").and_then(|c| {
            if let Value::String(ref s) = c.value { Some(s.as_str()) } else { None }
        });
        let layer_filter = conditions.iter().find(|c| c.field == "layer").and_then(|c| {
            if let Value::Integer(n) = c.value { Some(n as usize) } else { None }
        });
        let feature_filter = conditions.iter().find(|c| c.field == "feature").and_then(|c| {
            if let Value::Integer(n) = c.value { Some(n as usize) } else { None }
        });

        // Collect updates, then record
        let mut update_ops: Vec<(usize, usize, larql_vindex::FeatureMeta)> = Vec::new();
        {
            let (_path, _config, patched) = self.require_patched_mut()?;

            // Fast path: explicit (layer, feature) — same shape as DELETE.
            // Bypasses `find_features` so the caller can target a single
            // slot directly without needing to match by entity/relation.
            let matches: Vec<(usize, usize)> = if let (Some(layer), Some(feature)) = (layer_filter, feature_filter) {
                vec![(layer, feature)]
            } else {
                patched.base().find_features(entity_filter, None, layer_filter)
            };

            if matches.is_empty() {
                return Ok(vec!["  (no matching features found)".into()]);
            }

            for &(layer, feature) in &matches {
                if let Some(meta) = patched.feature_meta(layer, feature) {
                    let mut new_meta = meta;
                    for assignment in set {
                        match assignment.field.as_str() {
                            "target" | "top_token" => {
                                if let Value::String(ref s) = assignment.value {
                                    new_meta.top_token = s.clone();
                                }
                            }
                            "confidence" | "c_score" => {
                                if let Value::Number(n) = assignment.value {
                                    new_meta.c_score = n as f32;
                                } else if let Value::Integer(n) = assignment.value {
                                    new_meta.c_score = n as f32;
                                }
                            }
                            _ => {}
                        }
                    }
                    patched.update_feature_meta(layer, feature, new_meta.clone());
                    update_ops.push((layer, feature, new_meta));
                }
            }
        }

        // Record to patch session
        for (layer, feature, meta) in &update_ops {
            if let Some(ref mut recording) = self.patch_recording {
                recording.operations.push(larql_vindex::PatchOp::Update {
                    layer: *layer,
                    feature: *feature,
                    gate_vector_b64: None,
                    down_meta: Some(larql_vindex::patch::core::PatchDownMeta {
                        top_token: meta.top_token.clone(),
                        top_token_id: meta.top_token_id,
                        c_score: meta.c_score,
                    }),
                });
            }
        }

        Ok(vec![format!("Updated {} features (patch overlay)", update_ops.len())])
    }

    // ── MERGE ──

    pub(crate) fn exec_merge(
        &mut self,
        source: &str,
        target: Option<&str>,
        conflict: Option<ConflictStrategy>,
    ) -> Result<Vec<String>, LqlError> {
        let source_path = PathBuf::from(source);
        if !source_path.exists() {
            return Err(LqlError::Execution(format!(
                "source vindex not found: {}",
                source_path.display()
            )));
        }

        let target_path = if let Some(t) = target {
            let p = PathBuf::from(t);
            if !p.exists() {
                return Err(LqlError::Execution(format!(
                    "target vindex not found: {}",
                    p.display()
                )));
            }
            p
        } else {
            match &self.backend {
                Backend::Vindex { path, .. } => path.clone(),
                _ => return Err(LqlError::NoBackend),
            }
        };

        let strategy = conflict.unwrap_or(ConflictStrategy::KeepSource);

        // Load source
        let mut cb = larql_vindex::SilentLoadCallbacks;
        let source_index = larql_vindex::VectorIndex::load_vindex(&source_path, &mut cb)
            .map_err(|e| LqlError::exec("failed to load source", e))?;

        // Merge into the patch overlay
        let (_path, _config, patched) = self.require_patched_mut()?;

        let mut merged = 0;
        let mut skipped = 0;

        let source_layers = source_index.loaded_layers();
        for layer in source_layers {
            if let Some(source_metas) = source_index.down_meta_at(layer) {
                for (feature, meta_opt) in source_metas.iter().enumerate() {
                    if let Some(source_meta) = meta_opt {
                        let existing = patched.feature_meta(layer, feature);

                        let should_write = match (existing, &strategy) {
                            (None, _) => true,
                            (Some(_), ConflictStrategy::KeepSource) => true,
                            (Some(_), ConflictStrategy::KeepTarget) => false,
                            (Some(existing), ConflictStrategy::HighestConfidence) => {
                                source_meta.c_score > existing.c_score
                            }
                        };

                        if should_write {
                            patched.update_feature_meta(layer, feature, source_meta.clone());
                            merged += 1;
                        } else {
                            skipped += 1;
                        }
                    }
                }
            }
        }

        let mut out = Vec::new();
        out.push(format!(
            "Merged {} → {} (patch overlay)",
            source_path.display(),
            target_path.display()
        ));
        out.push(format!(
            "  {} features merged, {} skipped (strategy: {:?})",
            merged, skipped, strategy
        ));
        Ok(out)
    }
}

/// Architecture A helpers (kept for backward compatibility with existing patches).
#[allow(dead_code)]
/// Median per-feature norms at a layer for the gate / up / down matrices.
struct LayerMedianNorms {
    gate: f32,
    up: f32,
    down: f32,
}

/// Sample up to `sample_size` features at `layer` and compute the median
/// per-feature L2 norm for each of gate / up / down. Falls back to a
/// reasonable default (1.0) for any matrix the index doesn't carry.
///
/// We use median rather than mean to match the Python pipeline; mean is
/// pulled by outliers and produces a slightly different scale that
/// breaks reproduction of the validated install behaviour.
#[allow(dead_code)]
fn compute_layer_median_norms(
    base: &larql_vindex::VectorIndex,
    layer: usize,
    sample_size: usize,
) -> LayerMedianNorms {
    let n_features = base.num_features(layer);
    let sample_n = n_features.min(sample_size);

    let mut gate_norms = Vec::with_capacity(sample_n);
    let mut up_norms = Vec::with_capacity(sample_n);
    let mut down_norms = Vec::with_capacity(sample_n);

    let up_view = base.up_layer_matrix(layer);
    let down_view = base.down_layer_matrix(layer);

    for i in 0..sample_n {
        if let Some(g) = base.gate_vector(layer, i) {
            let n: f32 = g.iter().map(|v| v * v).sum::<f32>().sqrt();
            if n.is_finite() && n > 0.0 {
                gate_norms.push(n);
            }
        }
        if let Some(view) = up_view {
            if i < view.shape()[0] {
                let n: f32 = view.row(i).iter().map(|v| v * v).sum::<f32>().sqrt();
                if n.is_finite() && n > 0.0 {
                    up_norms.push(n);
                }
            }
        }
        if let Some(view) = down_view {
            if i < view.shape()[0] {
                let n: f32 = view.row(i).iter().map(|v| v * v).sum::<f32>().sqrt();
                if n.is_finite() && n > 0.0 {
                    down_norms.push(n);
                }
            }
        }
    }

    LayerMedianNorms {
        gate: median_or(&mut gate_norms, 1.0),
        up: median_or(&mut up_norms, 1.0),
        down: median_or(&mut down_norms, 1.0),
    }
}

#[allow(dead_code)]
fn median_or(xs: &mut [f32], default: f32) -> f32 {
    if xs.is_empty() {
        return default;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    xs[xs.len() / 2]
}

/// L2-normalise a vector. Returns the input unchanged if its norm is
/// effectively zero (degenerate case — embedding for an unknown token).
#[allow(dead_code)]
fn unit_vector(v: &[f32]) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n < 1e-8 {
        return v.to_vec();
    }
    v.iter().map(|x| x / n).collect()
}

#[cfg(test)]
mod install_helpers_tests {
    //! Unit tests for the install_compiled_slot helpers. These are the
    //! load-bearing math primitives for INSERT — getting any of them
    //! wrong silently weakens the install (validated in
    //! `experiments/14_vindex_compilation`: pre-fix retrieval was 6/10,
    //! post-fix should be 10/10). Test them in isolation so a future
    //! refactor can't drift the math without a red light.
    use super::*;

    #[test]
    fn unit_vector_normalises_to_length_one() {
        let v = vec![3.0_f32, 4.0]; // norm = 5
        let u = unit_vector(&v);
        let n: f32 = u.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-6, "unit norm; got {n}");
        assert!((u[0] - 0.6).abs() < 1e-6);
        assert!((u[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn unit_vector_passthrough_on_zero() {
        let v = vec![0.0_f32, 0.0, 0.0];
        let u = unit_vector(&v);
        assert_eq!(u, v, "zero vector should pass through unchanged");
    }

    #[test]
    fn unit_vector_handles_already_unit() {
        let v = vec![1.0_f32, 0.0, 0.0];
        let u = unit_vector(&v);
        for (a, b) in v.iter().zip(u.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn median_or_picks_middle() {
        let mut xs = vec![3.0_f32, 1.0, 2.0, 5.0, 4.0];
        // Sorted: [1, 2, 3, 4, 5], middle = index 2 = 3.0
        assert_eq!(median_or(&mut xs, 0.0), 3.0);
    }

    #[test]
    fn median_or_uses_default_when_empty() {
        let mut xs: Vec<f32> = Vec::new();
        assert_eq!(median_or(&mut xs, 1.5), 1.5);
    }

    #[test]
    fn median_or_handles_single_element() {
        let mut xs = vec![7.0_f32];
        assert_eq!(median_or(&mut xs, 0.0), 7.0);
    }

    #[test]
    fn median_or_sorts_input_in_place() {
        // Median sorts the slice as a side effect — this test exists
        // so a future refactor that switches to a non-sorting median
        // implementation can't accidentally break callers that rely on
        // the post-sort order. (Currently: nobody does, but the
        // contract is documented for safety.)
        let mut xs = vec![5.0_f32, 1.0, 3.0];
        let _ = median_or(&mut xs, 0.0);
        assert_eq!(xs, vec![1.0, 3.0, 5.0]);
    }

    /// End-to-end install math: synthesise gate / up / down at the
    /// magnitudes the install_compiled_slot pipeline would produce,
    /// and check the resulting activation is in the right ballpark for
    /// a slot that's expected to fire. This is a bench-mark
    /// sanity-check, not a precise test — the FFN nonlinearity
    /// (silu) means we can only assert orders of magnitude.
    #[test]
    fn install_math_produces_competing_activation() {
        const GATE_SCALE: f32 = 30.0;
        const ALPHA_MUL: f32 = 0.1;

        // A toy 4-dim layer.
        let g_ref = 2.0_f32;
        let u_ref = 1.5_f32;
        let d_ref = 3.0_f32;

        // Captured residual (gate direction).
        let residual = vec![0.6_f32, 0.0, 0.8, 0.0]; // norm = 1
        let gate_dir = unit_vector(&residual);

        // Install math (mirrors mutation.rs INSERT body).
        let gate_vec: Vec<f32> = gate_dir.iter().map(|v| v * g_ref * GATE_SCALE).collect();
        let up_vec: Vec<f32> = gate_dir.iter().map(|v| v * u_ref).collect();

        let gate_norm: f32 = gate_vec.iter().map(|v| v * v).sum::<f32>().sqrt();
        let up_norm: f32 = up_vec.iter().map(|v| v * v).sum::<f32>().sqrt();

        // Without GATE_SCALE the gate's norm would just be g_ref * 1 = 2.
        // With GATE_SCALE it should be 30× that = 60. The 30× is what
        // makes silu(gate · x) compete with trained slots at the layer.
        assert!((gate_norm - 60.0).abs() < 1e-3,
                "gate norm should be g_ref * 30 = 60, got {gate_norm}");
        assert!((up_norm - 1.5).abs() < 1e-3,
                "up norm should be u_ref = 1.5, got {up_norm}");

        // Down vector: target_embed_unit * d_ref * alpha_mul
        let target_embed = vec![0.0_f32, 0.5, 0.0, 0.866]; // norm ~1
        let target_norm: f32 = target_embed.iter().map(|v| v * v).sum::<f32>().sqrt();
        let payload = d_ref * ALPHA_MUL;
        let down_vec: Vec<f32> = target_embed.iter().map(|v| (v / target_norm) * payload).collect();
        let down_norm: f32 = down_vec.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((down_norm - payload).abs() < 1e-3,
                "down norm should be d_ref * alpha_mul = 0.3, got {down_norm}");

        // Sanity: the activation through this slot for an input
        // exactly aligned with the residual direction is huge — that's
        // what makes it compete.
        let x = gate_dir.clone();
        let gate_x: f32 = gate_vec.iter().zip(x.iter()).map(|(g, xi)| g * xi).sum();
        let up_x: f32 = up_vec.iter().zip(x.iter()).map(|(u, xi)| u * xi).sum();
        // gate · x = 60 (norm × cos = 60 × 1)
        // up · x = 1.5
        // silu(60) ≈ 60
        // activation ≈ 60 * 1.5 = 90
        let activation = silu(gate_x) * up_x;
        assert!(activation > 50.0,
                "activation along the install direction should be large; got {activation}");
    }

    fn silu(x: f32) -> f32 {
        x * (1.0 / (1.0 + (-x).exp()))
    }
}
