//! Query executor: WALK, INFER, SELECT, DESCRIBE, EXPLAIN.

use std::collections::HashMap;

use crate::ast::*;
use crate::error::LqlError;
use super::Session;
use super::helpers::is_content_token;

impl Session {
    // ── WALK ──
    //
    // Pure vindex feature scan. No attention. Shows what gate features fire
    // for the last token's embedding. This is a knowledge browser, not inference.

    pub(crate) fn exec_walk(
        &self,
        prompt: &str,
        top: Option<u32>,
        layers: Option<&Range>,
        mode: Option<WalkMode>,
        compare: bool,
    ) -> Result<Vec<String>, LqlError> {
        let (path, _config, patched) = self.require_vindex()?;
        let top_k = top.unwrap_or(10) as usize;

        let (embed, embed_scale) = larql_vindex::load_vindex_embeddings(path)
            .map_err(|e| LqlError::exec("failed to load embeddings", e))?;
        let tokenizer = larql_vindex::load_vindex_tokenizer(path)
            .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;

        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| LqlError::exec("tokenize error", e))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        if token_ids.is_empty() {
            return Err(LqlError::Execution("empty prompt".into()));
        }

        let last_tok = *token_ids.last().unwrap();
        let token_str = tokenizer
            .decode(&[last_tok], true)
            .unwrap_or_else(|_| format!("T{last_tok}"));

        let embed_row = embed.row(last_tok as usize);
        let query: larql_vindex::ndarray::Array1<f32> =
            embed_row.mapv(|v| v * embed_scale);

        let all_layers = patched.loaded_layers();
        let walk_layers: Vec<usize> = if let Some(range) = layers {
            (range.start as usize..=range.end as usize)
                .filter(|l| all_layers.contains(l))
                .collect()
        } else {
            all_layers
        };

        let start = std::time::Instant::now();
        let trace = patched.walk(&query, &walk_layers, top_k);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        let mode_str = match mode {
            Some(WalkMode::Pure) => "pure (sparse KNN only)",
            Some(WalkMode::Dense) => "dense (full matmul)",
            Some(WalkMode::Hybrid) | None => "hybrid (default)",
        };

        let mut out = Vec::new();
        out.push(format!(
            "Feature scan for {:?} (token {:?}, {} layers, mode={})",
            prompt,
            token_str.trim(),
            walk_layers.len(),
            mode_str,
        ));
        out.push(String::new());

        let show_per_layer = if compare { 5 } else { 3 };
        for (layer, hits) in &trace.layers {
            if hits.is_empty() {
                continue;
            }
            for hit in hits.iter().take(show_per_layer) {
                let down_top: String = hit
                    .meta
                    .top_k
                    .iter()
                    .take(3)
                    .map(|t| t.token.clone())
                    .collect::<Vec<_>>()
                    .join(", ");
                out.push(format!(
                    "  L{:2}: F{:<5} gate={:+.1}  top={:15}  down=[{}]",
                    layer, hit.feature, hit.gate_score,
                    format!("{:?}", hit.meta.top_token), down_top,
                ));
            }
        }

        out.push(format!("\n{:.1}ms", elapsed_ms));
        if compare {
            out.push(String::new());
            out.push("Note: COMPARE shows more features per layer. For inference use INFER.".into());
        } else {
            out.push(String::new());
            out.push("Note: pure vindex scan (no attention). For inference use INFER.".into());
        }

        Ok(out)
    }

    // ── INFER ──
    //
    // Full forward pass with attention. Requires model weights.

    pub(crate) fn exec_infer(
        &mut self,
        prompt: &str,
        top: Option<u32>,
        compare: bool,
    ) -> Result<Vec<String>, LqlError> {
        let top_k = top.unwrap_or(5) as usize;

        // Weight backend: dense inference (no vindex needed)
        if let super::Backend::Weight { weights, tokenizer, .. } = &self.backend {
            let encoding = tokenizer
                .encode(prompt, true)
                .map_err(|e| LqlError::exec("tokenize error", e))?;
            let token_ids: Vec<u32> = encoding.get_ids().to_vec();

            let start = std::time::Instant::now();
            let result = larql_inference::predict(weights, tokenizer, &token_ids, top_k);
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

            let mut out = Vec::new();
            out.push("Predictions (dense — no vindex):".into());
            for (i, (tok, prob)) in result.predictions.iter().enumerate() {
                out.push(format!(
                    "  {:2}. {:20} ({:.2}%)",
                    i + 1, tok, prob * 100.0
                ));
            }
            out.push(format!("  {:.0}ms", elapsed_ms));
            if !compare {
                out.push(String::new());
                out.push("Tip: EXTRACT into a vindex for walk FFN (sparse, faster, editable).".into());
            }
            return Ok(out);
        }

        // Vindex backend: walk FFN with optional dense comparison
        let (path, config, patched) = self.require_vindex()?;

        if !config.has_model_weights {
            return Err(LqlError::Execution(format!(
                "INFER requires model weights. This vindex was built without --include-weights.\n\
                 Rebuild: EXTRACT MODEL \"{}\" INTO \"{}\" WITH INFERENCE",
                config.model,
                path.display(),
            )));
        }

        let mut cb = larql_vindex::SilentLoadCallbacks;
        let weights = larql_vindex::load_model_weights(path, &mut cb)
            .map_err(|e| LqlError::exec("failed to load model weights", e))?;
        let tokenizer = larql_vindex::load_vindex_tokenizer(path)
            .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;

        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| LqlError::exec("tokenize error", e))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        // Unlimited top_k: use every feature at each layer, matching
        // the dense FFN path exactly. The 8092 default dropped half
        // of Gemma's 16384 features from the activation sum, which is
        // fine for a clean model (the discarded features have very
        // small activations) but becomes catastrophic once an INSERT
        // lands a strong (×30 gate scale) slot. The slot's activation
        // then dominates a half-weakened baseline, producing
        // whichever installed target has the largest lm_head alignment
        // on every prompt. Matching Python's dense forward pass by
        // using every feature preserves the baseline and keeps the
        // installed slot proportional.
        let walk_ffn = larql_inference::vindex::WalkFfn::new_unlimited_with_trace(&weights, patched);
        let start = std::time::Instant::now();
        let result = larql_inference::predict_with_ffn(
            &weights,
            &tokenizer,
            &token_ids,
            top_k,
            &walk_ffn,
        );
        let walk_ms = start.elapsed().as_secs_f64() * 1000.0;

        // ── KNN override check ──
        // Must call take_residuals BEFORE take_trace (both drain the same RefCell).
        let residuals = walk_ffn.take_residuals();

        // Check KNN store for retrieval override
        const KNN_COSINE_THRESHOLD: f32 = 0.75;
        let knn_layers = patched.knn_store.layers();
        let mut knn_override: Option<(String, f32, usize)> = None; // (token, cosine, layer)

        if !knn_layers.is_empty() {
            for &(ref layer, ref residual) in &residuals {
                if !knn_layers.contains(layer) { continue; }
                if let Some((entry, cosine)) = patched.knn_store.query_top1(*layer, residual) {
                    if cosine > KNN_COSINE_THRESHOLD {
                        knn_override = Some((entry.target_token.clone(), cosine, *layer));
                        break;
                    }
                }
            }
        }

        // Build trace from residuals (same logic as take_trace but inline)
        let mut trace_layers = Vec::with_capacity(residuals.len());
        for (layer, residual) in &residuals {
            let r = larql_vindex::ndarray::Array1::from_vec(residual.clone());
            let hits = patched.gate_knn(*layer, &r, 20);
            let walk_hits: Vec<larql_vindex::WalkHit> = hits
                .into_iter()
                .filter_map(|(feature, gate_score)| {
                    let meta = patched.feature_meta(*layer, feature)?;
                    Some(larql_vindex::WalkHit { layer: *layer, feature, gate_score, meta })
                })
                .collect();
            trace_layers.push((*layer, walk_hits));
        }

        let mut out = Vec::new();
        out.push("Predictions (walk FFN):".into());

        // If KNN override fired, show it first
        if let Some((ref token, cosine, knn_layer)) = knn_override {
            out.push(format!(
                "   1. {:20} (KNN override, cos={:.2}, L{})",
                token, cosine, knn_layer,
            ));
            for (i, (tok, prob)) in result.predictions.iter().enumerate() {
                out.push(format!(
                    "  {:2}. {:20} ({:.2}%)",
                    i + 2,
                    tok,
                    prob * 100.0
                ));
            }
        } else {
            for (i, (tok, prob)) in result.predictions.iter().enumerate() {
                out.push(format!(
                    "  {:2}. {:20} ({:.2}%)",
                    i + 1,
                    tok,
                    prob * 100.0
                ));
            }
        }
        out.push(format!("  {:.0}ms", walk_ms));

        out.push(String::new());
        out.push("Inference trace (features that fired with attention):".into());
        let classifier = self.relation_classifier();
        for (layer, hits) in &trace_layers {
            if hits.is_empty() {
                continue;
            }
            for hit in hits.iter().take(3) {
                let label = classifier
                    .and_then(|rc| rc.label_for_feature(*layer, hit.feature))
                    .unwrap_or("");
                let label_str = if label.is_empty() {
                    String::new()
                } else {
                    format!("{:<14}", label)
                };
                let top_token = hit.meta.top_token.trim();
                let down_top: String = hit
                    .meta
                    .top_k
                    .iter()
                    .take(3)
                    .map(|t| t.token.clone())
                    .collect::<Vec<_>>()
                    .join(", ");
                out.push(format!(
                    "  L{:2}: {} F{:<5} gate={:+.1}  → {:15} [{}]",
                    layer, label_str, hit.feature, hit.gate_score, top_token, down_top,
                ));
            }
        }

        if compare {
            let start = std::time::Instant::now();
            let dense = larql_inference::predict(&weights, &tokenizer, &token_ids, top_k);
            let dense_ms = start.elapsed().as_secs_f64() * 1000.0;

            out.push(String::new());
            out.push("Predictions (dense):".into());
            for (i, (tok, prob)) in dense.predictions.iter().enumerate() {
                out.push(format!(
                    "  {:2}. {:20} ({:.2}%)",
                    i + 1,
                    tok,
                    prob * 100.0
                ));
            }
            out.push(format!("  {:.0}ms", dense_ms));
        }

        Ok(out)
    }

    // ── DESCRIBE ──

    pub(crate) fn exec_describe(
        &self,
        entity: &str,
        band: Option<crate::ast::LayerBand>,
        layer: Option<u32>,
        relations_only: bool,
        mode: crate::ast::DescribeMode,
    ) -> Result<Vec<String>, LqlError> {
        let verbose = mode != crate::ast::DescribeMode::Brief;

        // MoE router-based DESCRIBE if available
        if let Some(router_result) = self.try_moe_describe(entity, band, layer, verbose)? {
            return Ok(router_result);
        }

        // ── Phase 1: load embeddings + tokenizer, build query vector ──
        let (path, config, patched) = self.require_vindex()?;
        let query = describe_build_query(entity, path)?;

        if query.is_none() {
            return Ok(vec![format!("{entity}\n  (not found)")]);
        }
        let query = query.unwrap();

        // ── Phase 2: pick scan layers from band/layer filter ──
        let bands = describe_resolve_bands(config);
        let scan_layers = describe_scan_layers(&bands, &patched.loaded_layers(), band, layer);

        // ── Phase 3: walk + collect edges ──
        let trace = patched.walk(&query, &scan_layers, 20);
        let mut edges = describe_collect_edges(&trace, entity);

        // ── Phase 3b: append KNN store entries for this entity ──
        let knn_hits = patched.knn_store.entries_for_entity(entity);
        for (knn_layer, entry) in knn_hits {
            edges.push(DescribeEdge {
                gate: entry.confidence * 10.0, // scale to match gate score range
                layers: vec![knn_layer],
                count: 1,
                original: entry.target_token.clone(),
                also: vec![format!("[knn:{}]", entry.relation)],
                best_layer: knn_layer,
                best_feature: 0,
            });
        }

        // ── Phase 4: format ──
        let mut out = vec![entity.to_string()];
        if edges.is_empty() {
            out.push("  (no edges found)".into());
            return Ok(out);
        }

        let formatted = describe_format_and_split(
            &edges,
            self.relation_classifier(),
            relations_only,
            &bands,
        );

        let max_edges = if mode == crate::ast::DescribeMode::Brief { 10 } else { 30 };

        if !formatted.syntax.is_empty() {
            out.push(format!("  Syntax (L{}-{}):", bands.syntax.0, bands.syntax.1));
            for edge in formatted.syntax.iter().take(max_edges) {
                out.push(format_describe_edge(edge, mode));
            }
        }
        if !formatted.knowledge.is_empty() {
            out.push(format!("  Edges (L{}-{}):", bands.knowledge.0, bands.knowledge.1));
            for edge in formatted.knowledge.iter().take(max_edges) {
                out.push(format_describe_edge(edge, mode));
            }
        }
        if !formatted.output_band.is_empty() {
            out.push(format!("  Output (L{}-{}):", bands.output.0, bands.output.1));
            let cap = if mode == crate::ast::DescribeMode::Brief { 5 } else { max_edges };
            for edge in formatted.output_band.iter().take(cap) {
                out.push(format_describe_edge(edge, mode));
            }
        }

        Ok(out)
    }

    // ── SELECT ──

    pub(crate) fn exec_select(
        &self,
        _fields: &[Field],
        conditions: &[Condition],
        nearest: Option<&NearestClause>,
        order: Option<&OrderBy>,
        limit: Option<u32>,
    ) -> Result<Vec<String>, LqlError> {
        let (path, _config, patched) = self.require_vindex()?;

        // Handle NEAREST TO clause — KNN lookup
        if let Some(nc) = nearest {
            return self.exec_select_nearest(patched, path, nc, limit);
        }

        let all_layers = patched.loaded_layers();
        // Default limit: num_layers when filtering by feature (user
        // expects to see the feature across all layers), otherwise 20.
        let feature_filter_present = conditions.iter().any(|c| c.field == "feature");
        let default_limit = if feature_filter_present {
            patched.num_layers()
        } else {
            20
        };
        let limit = limit.unwrap_or(default_limit as u32) as usize;

        let entity_filter = conditions.iter().find(|c| c.field == "entity").and_then(|c| {
            if let Value::String(ref s) = c.value { Some(s.as_str()) } else { None }
        });
        let relation_filter = conditions.iter().find(|c| c.field == "relation").and_then(|c| {
            if let Value::String(ref s) = c.value { Some(s.as_str()) } else { None }
        });
        let layer_filter = conditions.iter().find(|c| c.field == "layer").and_then(|c| {
            if let Value::Integer(n) = c.value { Some(n as usize) } else { None }
        });
        let feature_filter = conditions.iter().find(|c| c.field == "feature").and_then(|c| {
            if let Value::Integer(n) = c.value { Some(n as usize) } else { None }
        });
        let score_filter = conditions.iter().find(|c| c.field == "score" || c.field == "confidence").and_then(|c| {
            let val = match &c.value {
                Value::Number(n) => Some(*n as f32),
                Value::Integer(n) => Some(*n as f32),
                _ => None,
            };
            val.map(|v| (c.op.clone(), v))
        });

        struct Row {
            layer: usize,
            feature: usize,
            top_token: String,
            also: String,
            relation: String,
            c_score: f32,
        }

        let mut rows: Vec<Row> = Vec::new();
        let classifier = self.relation_classifier();

        let scan_layers: Vec<usize> = if let Some(l) = layer_filter {
            vec![l]
        } else {
            all_layers.clone()
        };

        // When entity + relation are both specified, use walk-based lookup:
        // embed the entity, walk all layers, find features that fire,
        // then filter by relation label. This finds "capital features that
        // activate for France" rather than "capital features whose top token
        // contains France".
        if let (Some(entity), Some(rel)) = (entity_filter, relation_filter) {

            let (embed, embed_scale) = larql_vindex::load_vindex_embeddings(path)
                .map_err(|e| LqlError::exec("failed to load embeddings", e))?;
            let tokenizer = larql_vindex::load_vindex_tokenizer(path)
                .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;

            let encoding = tokenizer
                .encode(entity, false)
                .map_err(|e| LqlError::exec("tokenize error", e))?;
            let token_ids: Vec<u32> = encoding.get_ids().to_vec();

            if !token_ids.is_empty() {
                let hidden = embed.shape()[1];
                let query = if token_ids.len() == 1 {
                    embed.row(token_ids[0] as usize).mapv(|v| v * embed_scale)
                } else {
                    let mut avg = larql_vindex::ndarray::Array1::<f32>::zeros(hidden);
                    for &tok in &token_ids {
                        avg += &embed.row(tok as usize).mapv(|v| v * embed_scale);
                    }
                    avg /= token_ids.len() as f32;
                    avg
                };

                // Use a large top_k because the raw embedding query
                // has low cosine with deep-layer gate directions (the
                // residual stream has been transformed by N layers of
                // attention+FFN). We need to scan widely to find the
                // relation-labeled features that fire on this entity.
                let trace = patched.walk(&query, &scan_layers, 500);

                for (layer_idx, hits) in &trace.layers {
                    for hit in hits {
                        if let Some(feature_f) = feature_filter {
                            if hit.feature != feature_f {
                                continue;
                            }
                        }
                        let rel_label = classifier
                            .and_then(|rc| rc.label_for_feature(*layer_idx, hit.feature))
                            .unwrap_or("")
                            .to_string();
                        if rel_label.is_empty() {
                            continue;
                        }
                        let rel_norm = rel.to_lowercase();
                        let label_norm = rel_label.to_lowercase();
                        if !label_norm.contains(&rel_norm) && !rel_norm.contains(&label_norm) {
                            continue;
                        }
                        let also = hit.meta.top_k.iter()
                            .skip(1)
                            .take(3)
                            .map(|e| e.token.clone())
                            .collect::<Vec<_>>()
                            .join(", ");
                        rows.push(Row {
                            layer: *layer_idx,
                            feature: hit.feature,
                            top_token: hit.meta.top_token.clone(),
                            also,
                            relation: rel_label,
                            c_score: hit.gate_score,
                        });
                    }
                }
            }
        } else {
            // Standard scan: iterate features via feature_meta() which
            // handles both heap and mmap modes. Earlier versions used
            // down_meta_at() which only reads heap-side metadata and
            // returned empty results on mmap-mode vindexes.
            for layer in &scan_layers {
                let nf = patched.num_features(*layer);
                for feat_idx in 0..nf {
                    if let Some(feature_f) = feature_filter {
                        if feat_idx != feature_f {
                            continue;
                        }
                    }
                    if let Some(meta) = patched.feature_meta(*layer, feat_idx) {
                        if let Some(ent) = entity_filter {
                            if !meta.top_token.to_lowercase().contains(&ent.to_lowercase()) {
                                continue;
                            }
                        }
                        let rel_label = classifier
                            .and_then(|rc| rc.label_for_feature(*layer, feat_idx))
                            .unwrap_or("")
                            .to_string();
                        if let Some(rel) = relation_filter {
                            if rel_label.is_empty() {
                                continue;
                            }
                            let rel_norm = rel.to_lowercase();
                            let label_norm = rel_label.to_lowercase();
                            if !label_norm.contains(&rel_norm) && !rel_norm.contains(&label_norm) {
                                continue;
                            }
                        }
                        let also = meta.top_k.iter()
                            .skip(1)
                            .take(3)
                            .map(|e| e.token.clone())
                            .collect::<Vec<_>>()
                            .join(", ");
                        rows.push(Row {
                            layer: *layer,
                            feature: feat_idx,
                            top_token: meta.top_token.clone(),
                            also,
                            relation: rel_label,
                            c_score: meta.c_score,
                        });
                    }
                }
            }
        }

        if let Some(ord) = order {
            match ord.field.as_str() {
                "confidence" | "c_score" => {
                    rows.sort_by(|a, b| {
                        let cmp = a.c_score.partial_cmp(&b.c_score).unwrap_or(std::cmp::Ordering::Equal);
                        if ord.descending { cmp.reverse() } else { cmp }
                    });
                }
                "layer" => {
                    rows.sort_by(|a, b| {
                        let cmp = a.layer.cmp(&b.layer);
                        if ord.descending { cmp.reverse() } else { cmp }
                    });
                }
                _ => {}
            }
        }

        // Apply score filter (WHERE score > N / score < N).
        if let Some((ref op, threshold)) = score_filter {
            rows.retain(|r| {
                match op {
                    CompareOp::Gt => r.c_score > threshold,
                    CompareOp::Lt => r.c_score < threshold,
                    CompareOp::Gte => r.c_score >= threshold,
                    CompareOp::Lte => r.c_score <= threshold,
                    CompareOp::Eq => (r.c_score - threshold).abs() < 0.001,
                    _ => true,
                }
            });
        }

        rows.truncate(limit);

        let show_relation = relation_filter.is_some()
            || rows.iter().any(|r| !r.relation.is_empty());
        let show_also = rows.iter().any(|r| !r.also.is_empty());

        let mut out = Vec::new();
        if show_relation {
            if show_also {
                out.push(format!(
                    "{:<8} {:<8} {:<16} {:<28} {:<14} {:>8}",
                    "Layer", "Feature", "Token", "Also", "Relation", "Score"
                ));
                out.push("-".repeat(86));
            } else {
                out.push(format!(
                    "{:<8} {:<8} {:<20} {:<20} {:>10}",
                    "Layer", "Feature", "Token", "Relation", "Score"
                ));
                out.push("-".repeat(70));
            }
        } else if show_also {
            out.push(format!(
                "{:<8} {:<8} {:<16} {:<28} {:>8}",
                "Layer", "Feature", "Token", "Also", "Score"
            ));
            out.push("-".repeat(72));
        } else {
            out.push(format!(
                "{:<8} {:<8} {:<20} {:>10}",
                "Layer", "Feature", "Token", "Score"
            ));
            out.push("-".repeat(50));
        }

        for row in &rows {
            let also_display = if row.also.is_empty() {
                String::new()
            } else {
                format!("[{}]", row.also)
            };
            if show_relation {
                if show_also {
                    out.push(format!(
                        "L{:<7} F{:<7} {:16} {:28} {:14} {:>8.4}",
                        row.layer, row.feature, row.top_token, also_display, row.relation, row.c_score
                    ));
                } else {
                    out.push(format!(
                        "L{:<7} F{:<7} {:20} {:20} {:>10.4}",
                        row.layer, row.feature, row.top_token, row.relation, row.c_score
                    ));
                }
            } else if show_also {
                out.push(format!(
                    "L{:<7} F{:<7} {:16} {:28} {:>8.4}",
                    row.layer, row.feature, row.top_token, also_display, row.c_score
                ));
            } else {
                out.push(format!(
                    "L{:<7} F{:<7} {:20} {:>10.4}",
                    row.layer, row.feature, row.top_token, row.c_score
                ));
            }
        }

        if rows.is_empty() {
            out.push("  (no matching edges)".into());
        }

        Ok(out)
    }

    /// SELECT NEAREST TO — KNN lookup at a specific layer.
    fn exec_select_nearest(
        &self,
        index: &larql_vindex::PatchedVindex,
        path: &std::path::Path,
        nc: &NearestClause,
        limit: Option<u32>,
    ) -> Result<Vec<String>, LqlError> {
        let limit = limit.unwrap_or(20) as usize;

        let (embed, embed_scale) = larql_vindex::load_vindex_embeddings(path)
            .map_err(|e| LqlError::exec("failed to load embeddings", e))?;
        let tokenizer = larql_vindex::load_vindex_tokenizer(path)
            .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;

        let encoding = tokenizer
            .encode(nc.entity.as_str(), false)
            .map_err(|e| LqlError::exec("tokenize error", e))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        if token_ids.is_empty() {
            return Ok(vec!["  (entity not found)".into()]);
        }

        // Build query from entity embedding
        let hidden = embed.shape()[1];
        let query = if token_ids.len() == 1 {
            embed.row(token_ids[0] as usize).mapv(|v| v * embed_scale)
        } else {
            let mut avg = larql_vindex::ndarray::Array1::<f32>::zeros(hidden);
            for &tok in &token_ids {
                avg += &embed.row(tok as usize).mapv(|v| v * embed_scale);
            }
            avg /= token_ids.len() as f32;
            avg
        };

        // KNN at the specified layer
        let hits = index.gate_knn(nc.layer as usize, &query, limit);

        let classifier = self.relation_classifier();

        let mut out = Vec::new();
        out.push(format!(
            "{:<8} {:<8} {:<16} {:<28} {:<14} {:>8}",
            "Layer", "Feature", "Token", "Also", "Relation", "Score"
        ));
        out.push("-".repeat(86));

        for (feat, score) in &hits {
            let meta = index.feature_meta(nc.layer as usize, *feat);
            let tok = meta.as_ref()
                .map(|m| m.top_token.clone())
                .unwrap_or_else(|| "-".into());
            let also = meta.as_ref()
                .map(|m| {
                    let items: Vec<_> = m.top_k.iter()
                        .skip(1)
                        .take(3)
                        .map(|e| e.token.clone())
                        .collect();
                    if items.is_empty() { String::new() } else { format!("[{}]", items.join(", ")) }
                })
                .unwrap_or_default();
            let rel = classifier
                .and_then(|rc| rc.label_for_feature(nc.layer as usize, *feat))
                .unwrap_or("");
            out.push(format!(
                "L{:<7} F{:<7} {:16} {:28} {:14} {:>8.4}",
                nc.layer, feat, tok, also, rel, score
            ));
        }

        if hits.is_empty() {
            out.push("  (no matching features)".into());
        }

        Ok(out)
    }

    // ── SELECT * FROM FEATURES ──

    pub(crate) fn exec_select_features(
        &self,
        conditions: &[Condition],
        limit: Option<u32>,
    ) -> Result<Vec<String>, LqlError> {
        let (_path, config, patched) = self.require_vindex()?;
        let classifier = self.relation_classifier();

        let layer_filter = conditions.iter().find(|c| c.field == "layer").and_then(|c| {
            if let Value::Integer(n) = c.value { Some(n as usize) } else { None }
        });
        let feature_filter = conditions.iter().find(|c| c.field == "feature").and_then(|c| {
            if let Value::Integer(n) = c.value { Some(n as usize) } else { None }
        });
        let token_filter = conditions.iter().find(|c| c.field == "token" || c.field == "entity").and_then(|c| {
            if let Value::String(ref s) = c.value { Some(s.as_str()) } else { None }
        });

        let default_limit = if feature_filter.is_some() {
            config.num_layers
        } else if layer_filter.is_some() {
            config.intermediate_size
        } else {
            34
        };
        let limit = limit.unwrap_or(default_limit as u32) as usize;

        let scan_layers: Vec<usize> = if let Some(l) = layer_filter {
            vec![l]
        } else {
            (0..config.num_layers).collect()
        };

        let mut out = Vec::new();
        out.push(format!(
            "{:<8} {:<8} {:<16} {:<28} {:<14} {:>8}",
            "Layer", "Feature", "Token", "Also", "Relation", "Score"
        ));
        out.push("-".repeat(86));

        let mut count = 0;
        for layer in &scan_layers {
            let nf = patched.num_features(*layer);
            for feat in 0..nf {
                if count >= limit { break; }
                if let Some(ff) = feature_filter {
                    if feat != ff { continue; }
                }
                if let Some(meta) = patched.feature_meta(*layer, feat) {
                    if let Some(tf) = token_filter {
                        if !meta.top_token.to_lowercase().contains(&tf.to_lowercase()) {
                            continue;
                        }
                    }
                    let also: String = meta.top_k.iter()
                        .skip(1).take(3)
                        .map(|e| e.token.clone())
                        .collect::<Vec<_>>()
                        .join(", ");
                    let also_display = if also.is_empty() { String::new() } else { format!("[{}]", also) };
                    let rel = classifier
                        .and_then(|rc| rc.label_for_feature(*layer, feat))
                        .unwrap_or("");
                    out.push(format!(
                        "L{:<7} F{:<7} {:16} {:28} {:14} {:>8.4}",
                        layer, feat, meta.top_token, also_display, rel, meta.c_score
                    ));
                    count += 1;
                }
            }
            if count >= limit { break; }
        }

        if count == 0 {
            out.push("  (no matching features)".into());
        }

        Ok(out)
    }

    // ── SELECT * FROM ENTITIES ──

    pub(crate) fn exec_select_entities(
        &self,
        conditions: &[Condition],
        limit: Option<u32>,
    ) -> Result<Vec<String>, LqlError> {
        let (_path, config, patched) = self.require_vindex()?;

        let layer_filter = conditions.iter().find(|c| c.field == "layer").and_then(|c| {
            if let Value::Integer(n) = c.value { Some(n as usize) } else { None }
        });
        let entity_filter = conditions.iter().find(|c| c.field == "entity" || c.field == "token").and_then(|c| {
            if let Value::String(ref s) = c.value { Some(s.as_str()) } else { None }
        });
        let limit = limit.unwrap_or(50) as usize;

        let scan_layers: Vec<usize> = if let Some(l) = layer_filter {
            vec![l]
        } else {
            (0..config.num_layers).collect()
        };

        // Common English stop words to filter out — these are capitalized
        // at sentence starts but aren't named entities.
        const STOP_WORDS: &[&str] = &[
            "The", "For", "And", "But", "Not", "This", "That", "With",
            "From", "Into", "Will", "Can", "One", "All", "Any", "Has",
            "Had", "Was", "Are", "Were", "Been", "His", "Her", "Its",
            "Our", "Who", "How", "Why", "When", "What", "Where", "Which",
            "Each", "Both", "Some", "Most", "Many", "Much", "More", "Such",
            "Than", "Then", "Also", "Just", "Now", "May", "Per", "Pre",
            "Pro", "Con", "Dis", "Via", "Yet", "Nor", "Should", "Would",
            "Could", "Did", "Does", "Too", "Very", "Instead", "Mon",
            "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
            "First", "Second", "Third", "Fourth", "Fifth", "Sixth",
            "Forty", "Fifty", "Only", "Over", "Under", "After", "Before",
            "About", "Above", "Below", "Between", "Through",
        ];

        // Collect distinct entity-like tokens.
        let mut entity_counts: std::collections::HashMap<String, (usize, f32)> =
            std::collections::HashMap::new();

        for layer in &scan_layers {
            let nf = patched.num_features(*layer);
            for feat in 0..nf {
                if let Some(meta) = patched.feature_meta(*layer, feat) {
                    let tok = meta.top_token.trim().to_string();
                    // Named entities: uppercase start, 3+ chars, all alphabetic.
                    if tok.len() < 3 { continue; }
                    let first = tok.chars().next().unwrap_or(' ');
                    if !first.is_ascii_uppercase() { continue; }
                    if !tok.chars().all(|c| c.is_alphabetic()) { continue; }
                    if STOP_WORDS.contains(&tok.as_str()) { continue; }
                    // Entity name filter (WHERE entity = "X").
                    if let Some(ef) = entity_filter {
                        if !tok.to_lowercase().contains(&ef.to_lowercase()) { continue; }
                    }
                    let entry = entity_counts.entry(tok).or_insert((0, 0.0));
                    entry.0 += 1;
                    if meta.c_score > entry.1 { entry.1 = meta.c_score; }
                }
            }
        }

        let mut entities: Vec<(String, usize, f32)> = entity_counts
            .into_iter()
            .map(|(tok, (count, max_score))| (tok, count, max_score))
            .collect();
        entities.sort_by(|a, b| b.1.cmp(&a.1));
        entities.truncate(limit);

        let mut out = Vec::new();
        out.push(format!(
            "{:<24} {:>10} {:>10}",
            "Entity", "Features", "Max Score"
        ));
        out.push("-".repeat(48));

        for (tok, count, max_score) in &entities {
            out.push(format!(
                "{:<24} {:>10} {:>10.4}",
                tok, count, max_score
            ));
        }

        if entities.is_empty() {
            out.push("  (no entities found)".into());
        }

        Ok(out)
    }

    // ── EXPLAIN ──

    pub(crate) fn exec_explain(
        &self,
        prompt: &str,
        layers: Option<&Range>,
        verbose: bool,
    ) -> Result<Vec<String>, LqlError> {
        let (path, _config, patched) = self.require_vindex()?;

        let (embed, embed_scale) = larql_vindex::load_vindex_embeddings(path)
            .map_err(|e| LqlError::exec("failed to load embeddings", e))?;
        let tokenizer = larql_vindex::load_vindex_tokenizer(path)
            .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;

        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| LqlError::exec("tokenize error", e))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        if token_ids.is_empty() {
            return Err(LqlError::Execution("empty prompt".into()));
        }

        let last_tok = *token_ids.last().unwrap();
        let embed_row = embed.row(last_tok as usize);
        let query: larql_vindex::ndarray::Array1<f32> =
            embed_row.mapv(|v| v * embed_scale);

        let all_layers = patched.loaded_layers();
        let walk_layers: Vec<usize> = if let Some(range) = layers {
            (range.start as usize..=range.end as usize)
                .filter(|l| all_layers.contains(l))
                .collect()
        } else {
            all_layers
        };

        let top_k = if verbose { 10 } else { 5 };
        let trace = patched.walk(&query, &walk_layers, top_k);

        let mut out = Vec::new();
        for (layer, hits) in &trace.layers {
            let show_count = if verbose { hits.len() } else { hits.len().min(5) };
            for hit in hits.iter().take(show_count) {
                let down_count = if verbose { 5 } else { 3 };
                let down_tokens: String = hit
                    .meta
                    .top_k
                    .iter()
                    .take(down_count)
                    .map(|t| t.token.clone())
                    .collect::<Vec<_>>()
                    .join(", ");

                out.push(format!(
                    "L{}: F{} → {} (gate={:.1}, down=[{}])",
                    layer, hit.feature, hit.meta.top_token, hit.gate_score, down_tokens
                ));
            }
        }

        Ok(out)
    }

    // ── EXPLAIN INFER (with attention) ──

    pub(crate) fn exec_infer_trace(
        &self,
        prompt: &str,
        top: Option<u32>,
        band: Option<crate::ast::LayerBand>,
        relations_only: bool,
        with_attention: bool,
    ) -> Result<Vec<String>, LqlError> {
        let top_k = top.unwrap_or(5) as usize;
        let per_layer = top.unwrap_or(3) as usize;

        // Weight backend has no feature labels — short-circuit to a
        // dense-only summary.
        if let super::Backend::Weight { weights, tokenizer, .. } = &self.backend {
            return self.exec_infer_trace_dense(weights, tokenizer, prompt, top_k);
        }

        // ── Phase 1: load model weights and tokenise ──
        let (path, config, patched) = self.require_vindex()?;
        if !config.has_model_weights {
            return Err(LqlError::Execution(
                "EXPLAIN INFER requires model weights. Rebuild with WITH INFERENCE.".into(),
            ));
        }
        let mut cb = larql_vindex::SilentLoadCallbacks;
        let weights = larql_vindex::load_model_weights(path, &mut cb)
            .map_err(|e| LqlError::exec("failed to load model weights", e))?;
        let tokenizer = larql_vindex::load_vindex_tokenizer(path)
            .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;
        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| LqlError::exec("tokenize error", e))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        let token_strs: Vec<Option<String>> = if with_attention {
            token_ids
                .iter()
                .map(|&id| larql_inference::decode_token(&tokenizer, id))
                .collect()
        } else {
            Vec::new()
        };

        // ── Phase 2: forward pass (with optional attention capture) ──
        //
        // Unlimited top_k so EXPLAIN INFER's activation sum matches
        // what `exec_infer` uses. Otherwise a user who runs INFER
        // then EXPLAIN INFER on the same prompt sees a half-power
        // baseline in the trace while production inference uses
        // full power — silent divergence.
        let walk_ffn = larql_inference::vindex::WalkFfn::new_unlimited_with_trace(&weights, patched);
        let start = std::time::Instant::now();
        let (predictions, attention_captures, lens_residuals) = if with_attention {
            let r = larql_inference::predict_with_ffn_attention(
                &weights, &tokenizer, &token_ids, top_k, &walk_ffn,
            );
            (r.predictions, r.attention, r.residuals)
        } else {
            let r = larql_inference::predict_with_ffn(
                &weights, &tokenizer, &token_ids, top_k, &walk_ffn,
            );
            (r.predictions, Vec::new(), Vec::new())
        };
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        // ── Phase 3: side-tables for the rendering loop ──
        let attention_map = build_attention_map(&attention_captures, &token_strs, with_attention);
        let lens_map = build_lens_map(&lens_residuals, &weights, &tokenizer, with_attention);

        let trace = walk_ffn.take_trace();
        let classifier = self.relation_classifier();
        let bands = describe_resolve_bands(config);
        let layer_range = band_to_layer_range(band, &bands);

        // ── Phase 4: format header ──
        let band_label = match band {
            Some(crate::ast::LayerBand::Syntax) => " (syntax)",
            Some(crate::ast::LayerBand::Knowledge) => " (knowledge)",
            Some(crate::ast::LayerBand::Output) => " (output)",
            _ => "",
        };

        let mut out = Vec::new();
        out.push(format!("Inference trace for {:?}{}:", prompt, band_label));
        out.push(format!(
            "Prediction: {} ({:.2}%) in {:.0}ms",
            predictions.first().map(|(t, _)| t.as_str()).unwrap_or("?"),
            predictions.first().map(|(_, p)| p * 100.0).unwrap_or(0.0),
            elapsed_ms
        ));
        out.push(String::new());

        // ── Phase 5: per-layer rendering ──
        for (layer, hits) in &trace.layers {
            if hits.is_empty() {
                continue;
            }
            if let Some((lo, hi)) = layer_range {
                if *layer < lo || *layer > hi {
                    continue;
                }
            }
            render_trace_layer(
                &mut out,
                *layer,
                hits,
                classifier,
                relations_only,
                per_layer,
                with_attention,
                &attention_map,
                &lens_map,
            );
        }

        Ok(out)
    }

    /// EXPLAIN INFER on a `Backend::Weight` (no vindex): produces a dense
    /// inference summary with no feature trace, since there are no
    /// gate vectors / down meta to attribute.
    fn exec_infer_trace_dense(
        &self,
        weights: &larql_inference::ModelWeights,
        tokenizer: &larql_inference::tokenizers::Tokenizer,
        prompt: &str,
        top_k: usize,
    ) -> Result<Vec<String>, LqlError> {
        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| LqlError::exec("tokenize error", e))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        let start = std::time::Instant::now();
        let result = larql_inference::predict(weights, tokenizer, &token_ids, top_k);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        let mut out = Vec::new();
        out.push(format!("Inference trace for {:?} (dense — no vindex):", prompt));
        out.push(format!(
            "Prediction: {} ({:.2}%) in {:.0}ms",
            result.predictions.first().map(|(t, _)| t.as_str()).unwrap_or("?"),
            result.predictions.first().map(|(_, p)| p * 100.0).unwrap_or(0.0),
            elapsed_ms,
        ));
        out.push(String::new());
        out.push("Note: no per-feature trace without a vindex. EXTRACT for full trace.".into());
        Ok(out)
    }

    // ── MoE Router-guided DESCRIBE ──

    /// For MoE models: use the router to select experts, then gate KNN within
    /// only the selected experts' features. Same output format as dense DESCRIBE.
    /// Returns None if no router (dense model — falls through to standard gate KNN).
    fn try_moe_describe(
        &self,
        entity: &str,
        _band: Option<crate::ast::LayerBand>,
        _layer: Option<u32>,
        verbose: bool,
    ) -> Result<Option<Vec<String>>, LqlError> {
        let router = match &self.backend {
            super::Backend::Vindex { router: Some(r), config, .. } => {
                if config.model_config.as_ref().and_then(|mc| mc.moe.as_ref()).is_none() {
                    return Ok(None);
                }
                r
            }
            _ => return Ok(None),
        };

        let (path, config, _) = self.require_vindex()?;

        let (embed, embed_scale) = larql_vindex::load_vindex_embeddings(path)
            .map_err(|e| LqlError::exec("failed to load embeddings", e))?;
        let tokenizer = larql_vindex::load_vindex_tokenizer(path)
            .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;

        let encoding = tokenizer.encode(entity, false)
            .map_err(|e| LqlError::exec("tokenize error", e))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();
        if token_ids.is_empty() {
            return Ok(Some(vec![format!("{entity}\n  (not found)")]));
        }

        let hidden = embed.shape()[1];
        let query = if token_ids.len() == 1 {
            embed.row(token_ids[0] as usize).mapv(|v| v * embed_scale)
        } else {
            let mut avg = larql_vindex::ndarray::Array1::<f32>::zeros(hidden);
            for &tok in &token_ids {
                avg += &embed.row(tok as usize).mapv(|v| v * embed_scale);
            }
            avg /= token_ids.len() as f32;
            avg
        };

        let last = config.num_layers.saturating_sub(1);
        let bands = config.layer_bands.clone()
            .or_else(|| larql_vindex::LayerBands::for_family(&config.family, config.num_layers))
            .unwrap_or(larql_vindex::LayerBands {
                syntax: (0, last), knowledge: (0, last), output: (0, last),
            });

        let start = std::time::Instant::now();

        // ── Per-layer expert routing ──
        let mut out = vec![entity.to_string()];

        // Aggregate: which experts are most active across the knowledge band?
        let knowledge_range = bands.knowledge.0..=bands.knowledge.1;
        let expert_summary = router.route_all_layers(&query, knowledge_range.clone());

        // Show per-layer routing in verbose mode
        if verbose {
            out.push(format!("  Routing (L{}-{}):", bands.knowledge.0, bands.knowledge.1));
            for l in knowledge_range.clone() {
                if let Some(result) = router.route(l, &query) {
                    let experts_str: String = result.experts.iter().enumerate()
                        .map(|(i, e)| format!("E{} ({:.0}%)", e, result.probs[i] * 100.0))
                        .collect::<Vec<_>>()
                        .join(", ");
                    out.push(format!("    L{:2}: {}", l, experts_str));
                }
            }
            out.push(String::new());
        }

        // ── Expert summary ──
        let layers_total = bands.knowledge.1 - bands.knowledge.0 + 1;
        out.push(format!("  Experts (L{}-{}):", bands.knowledge.0, bands.knowledge.1));
        let max_experts = if verbose { 15 } else { 6 };
        for (eid, count, avg_prob) in expert_summary.iter().take(max_experts) {
            out.push(format!(
                "    E{:<4} {}/{} layers  ({:.0}% avg)",
                eid, count, layers_total, avg_prob * 100.0,
            ));
        }

        // ── Co-routed entities: what else routes to the same experts? ──
        let top_experts: Vec<usize> = expert_summary.iter()
            .take(3)
            .map(|(e, _, _)| *e)
            .collect();

        if !top_experts.is_empty() {
            out.push(String::new());
            out.push("  Similar (shares experts):".into());

            let mid_layer = (bands.knowledge.0 + bands.knowledge.1) / 2;

            // Sample vocab and find entities that route to the same experts
            let sample_step = (embed.shape()[0] / 2000).max(1);
            let mut corouted_all: HashMap<usize, Vec<(String, f32)>> = HashMap::new();

            for tid in (0..embed.shape()[0]).step_by(sample_step) {
                let tok_emb = embed.row(tid).mapv(|v| v * embed_scale);
                if let Some(result) = router.route(mid_layer, &tok_emb) {
                    for (i, &eid) in result.experts.iter().enumerate() {
                        if top_experts.contains(&eid) {
                            let tok_str = tokenizer.decode(&[tid as u32], true)
                                .unwrap_or_default().trim().to_string();
                            if is_content_token(&tok_str) && tok_str.len() > 1
                                && tok_str.to_lowercase() != entity.to_lowercase()
                            {
                                corouted_all.entry(eid)
                                    .or_default()
                                    .push((tok_str, result.probs[i]));
                            }
                        }
                    }
                }
            }

            for &eid in &top_experts {
                if let Some(tokens) = corouted_all.get_mut(&eid) {
                    tokens.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    tokens.dedup_by(|a, b| a.0.to_lowercase() == b.0.to_lowercase());
                    let display: String = tokens.iter()
                        .take(10)
                        .map(|(t, _)| t.as_str())
                        .collect::<Vec<_>>()
                        .join(", ");
                    out.push(format!("    E{}: {}", eid, display));
                }
            }
        }

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        out.push(format!("\n  {:.0}ms", elapsed_ms));

        Ok(Some(out))
    }
}

// ── DESCRIBE helpers ────────────────────────────────────────────────────
//
// `exec_describe` is a five-phase pipeline (load query → resolve bands →
// walk → collect edges → format). The helpers below split each phase out
// of the main function so the orchestration reads top-down.

/// Tokenise `entity` and build a query vector by averaging its token
/// embeddings (single tokens get their embed row directly). Returns
/// `Ok(None)` when the entity tokenises to nothing — the caller emits
/// the "(not found)" line.
fn describe_build_query(
    entity: &str,
    path: &std::path::Path,
) -> Result<Option<larql_vindex::ndarray::Array1<f32>>, LqlError> {
    let (embed, embed_scale) = larql_vindex::load_vindex_embeddings(path)
        .map_err(|e| LqlError::exec("failed to load embeddings", e))?;
    let tokenizer = larql_vindex::load_vindex_tokenizer(path)
        .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;

    let encoding = tokenizer
        .encode(entity, false)
        .map_err(|e| LqlError::exec("tokenize error", e))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    if token_ids.is_empty() {
        return Ok(None);
    }

    let hidden = embed.shape()[1];
    let query = if token_ids.len() == 1 {
        let tok = token_ids[0];
        embed.row(tok as usize).mapv(|v| v * embed_scale)
    } else {
        let mut avg = larql_vindex::ndarray::Array1::<f32>::zeros(hidden);
        for &tok in &token_ids {
            let row = embed.row(tok as usize);
            avg += &row.mapv(|v| v * embed_scale);
        }
        avg /= token_ids.len() as f32;
        avg
    };
    Ok(Some(query))
}

/// Resolve the layer-band boundaries from the vindex config, with a
/// family-based default and a final whole-range fallback.
fn describe_resolve_bands(config: &larql_vindex::VindexConfig) -> larql_vindex::LayerBands {
    let last = config.num_layers.saturating_sub(1);
    config
        .layer_bands
        .clone()
        .or_else(|| larql_vindex::LayerBands::for_family(&config.family, config.num_layers))
        .unwrap_or(larql_vindex::LayerBands {
            syntax: (0, last),
            knowledge: (0, last),
            output: (0, last),
        })
}

/// Filter `all_layers` down to those covered by the requested band /
/// explicit layer.
fn describe_scan_layers(
    bands: &larql_vindex::LayerBands,
    all_layers: &[usize],
    band: Option<crate::ast::LayerBand>,
    layer: Option<u32>,
) -> Vec<usize> {
    if let Some(l) = layer {
        return vec![l as usize];
    }
    match band {
        Some(crate::ast::LayerBand::Syntax) => all_layers
            .iter()
            .copied()
            .filter(|l| *l >= bands.syntax.0 && *l <= bands.syntax.1)
            .collect(),
        Some(crate::ast::LayerBand::Knowledge) => all_layers
            .iter()
            .copied()
            .filter(|l| *l >= bands.knowledge.0 && *l <= bands.knowledge.1)
            .collect(),
        Some(crate::ast::LayerBand::Output) => all_layers
            .iter()
            .copied()
            .filter(|l| *l >= bands.output.0 && *l <= bands.output.1)
            .collect(),
        Some(crate::ast::LayerBand::All) | None => all_layers.to_vec(),
    }
}

/// Per-target accumulator for the walk-collected edges.
struct DescribeEdge {
    gate: f32,
    layers: Vec<usize>,
    count: usize,
    original: String,
    also: Vec<String>,
    best_layer: usize,
    best_feature: usize,
}

/// A formatted edge ready to be rendered into the output buffer. Built
/// from a `DescribeEdge` by `describe_format_and_split` after label
/// resolution and the RELATIONS ONLY filter.
struct FormattedEdge {
    /// Probe label, raw cluster label, or empty when no label is known.
    label: String,
    is_probe: bool,
    is_cluster: bool,
    target: String,
    gate: f32,
    primary_layer: usize,
    layers: Vec<usize>,
    count: usize,
    also: Vec<String>,
}

/// The three formatted-edge buckets returned by
/// `describe_format_and_split`, one per layer band.
struct DescribeBands {
    syntax: Vec<FormattedEdge>,
    knowledge: Vec<FormattedEdge>,
    output_band: Vec<FormattedEdge>,
}

/// Walk the trace, deduplicate by lowercased target token, and apply
/// content / coherence filters. The output is sorted descending by gate.
fn describe_collect_edges(
    trace: &larql_vindex::WalkTrace,
    entity: &str,
) -> Vec<DescribeEdge> {
    let entity_lower = entity.to_lowercase();
    let gate_threshold = 5.0_f32;
    let mut edges: HashMap<String, DescribeEdge> = HashMap::new();

    for (layer_idx, hits) in &trace.layers {
        for hit in hits {
            if hit.gate_score < gate_threshold {
                continue;
            }
            let tok = &hit.meta.top_token;
            if !is_content_token(tok) {
                continue;
            }
            if tok.to_lowercase() == entity_lower {
                continue;
            }

            let also_readable: Vec<String> = hit
                .meta
                .top_k
                .iter()
                .filter(|t| {
                    t.token.to_lowercase() != tok.to_lowercase()
                        && t.token.to_lowercase() != entity_lower
                        && super::helpers::is_readable_token(&t.token)
                        && t.logit > 0.0
                })
                .take(5)
                .map(|t| t.token.clone())
                .collect();

            let also: Vec<String> = also_readable
                .iter()
                .filter(|t| is_content_token(t))
                .take(3)
                .cloned()
                .collect();

            // Coherence filter: skip weak edges with no content secondaries
            if also.is_empty() && !also_readable.is_empty() && hit.gate_score < 20.0 {
                continue;
            }

            let key = tok.to_lowercase();
            let entry = edges.entry(key).or_insert_with(|| DescribeEdge {
                gate: 0.0,
                layers: Vec::new(),
                count: 0,
                original: tok.to_string(),
                also,
                best_layer: *layer_idx,
                best_feature: hit.feature,
            });

            if hit.gate_score > entry.gate {
                entry.gate = hit.gate_score;
                entry.best_layer = *layer_idx;
                entry.best_feature = hit.feature;
            }
            if !entry.layers.contains(layer_idx) {
                entry.layers.push(*layer_idx);
            }
            entry.count += 1;
        }
    }

    let mut ranked: Vec<DescribeEdge> = edges.into_values().collect();
    ranked.sort_by(|a, b| {
        b.gate
            .partial_cmp(&a.gate)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked
}

/// Resolve relation labels from the optional `RelationClassifier`, apply
/// the RELATIONS ONLY filter, and split the resulting `FormattedEdge`s
/// into syntax / knowledge / output buckets according to which band the
/// edge's primary layer falls in.
fn describe_format_and_split(
    edges: &[DescribeEdge],
    classifier: Option<&crate::relations::RelationClassifier>,
    relations_only: bool,
    bands: &larql_vindex::LayerBands,
) -> DescribeBands {
    let formatted: Vec<FormattedEdge> = edges
        .iter()
        .map(|info| {
            let (label, is_probe, is_cluster) = if let Some(rc) = classifier {
                if let Some(lbl) = rc.label_for_feature(info.best_layer, info.best_feature) {
                    let probe = rc.is_probe_label(info.best_layer, info.best_feature);
                    (lbl.to_string(), probe, !probe)
                } else {
                    (String::new(), false, false)
                }
            } else {
                (String::new(), false, false)
            };
            FormattedEdge {
                label,
                is_probe,
                is_cluster,
                target: info.original.clone(),
                gate: info.gate,
                primary_layer: info.best_layer,
                layers: info.layers.clone(),
                count: info.count,
                also: info.also.clone(),
            }
        })
        .filter(|e| !relations_only || e.is_probe || e.is_cluster)
        .collect();

    let mut out = DescribeBands {
        syntax: Vec::new(),
        knowledge: Vec::new(),
        output_band: Vec::new(),
    };
    for edge in formatted {
        let primary = edge.primary_layer;
        if primary >= bands.syntax.0 && primary <= bands.syntax.1 {
            out.syntax.push(edge);
        } else if primary >= bands.knowledge.0 && primary <= bands.knowledge.1 {
            out.knowledge.push(edge);
        } else if primary >= bands.output.0 && primary <= bands.output.1 {
            out.output_band.push(edge);
        } else {
            // Layer outside any band — fall back to knowledge.
            out.knowledge.push(edge);
        }
    }
    out
}

/// Render a single `FormattedEdge` into a single line of DESCRIBE output.
/// The three modes share the same shape:
///
///   - **Verbose** (default): `[relation]    → target  gate  L20-L27  Nx  also: ...`
///   - **Brief**: compact `relation    → target  gate  L26`, no also-tokens
///   - **Raw**: no labels, otherwise like Verbose
fn format_describe_edge(edge: &FormattedEdge, mode: crate::ast::DescribeMode) -> String {
    match mode {
        crate::ast::DescribeMode::Verbose => {
            let bracket_label = if edge.label.is_empty() {
                format!("{:<14}", "[—]")
            } else {
                let tag = format!("[{}]", edge.label);
                format!("{:<14}", tag)
            };
            let (min_l, max_l) = layer_range(&edge.layers);
            let layer_str = if min_l == max_l {
                format!("L{min_l}")
            } else {
                format!("L{min_l}-{max_l}")
            };
            let also = format_also(&edge.also);
            format!(
                "    {} → {:20} {:>7.1}  {:<8} {}x{}",
                bracket_label, edge.target, edge.gate, layer_str, edge.count, also,
            )
        }
        crate::ast::DescribeMode::Brief => {
            let label = if edge.is_probe {
                format!("{:<12}", edge.label)
            } else {
                format!("{:<12}", "")
            };
            format!(
                "    {} → {:20} {:>7.1}  L{:<3}",
                label, edge.target, edge.gate, edge.primary_layer,
            )
        }
        crate::ast::DescribeMode::Raw => {
            let (min_l, max_l) = layer_range(&edge.layers);
            let layer_str = if min_l == max_l {
                format!("L{min_l}")
            } else {
                format!("L{min_l}-{max_l}")
            };
            let also = format_also(&edge.also);
            format!(
                "                 → {:20} {:>7.1}  {:<8} {}x{}",
                edge.target, edge.gate, layer_str, edge.count, also,
            )
        }
    }
}

fn layer_range(layers: &[usize]) -> (usize, usize) {
    let min_l = *layers.iter().min().unwrap_or(&0);
    let max_l = *layers.iter().max().unwrap_or(&0);
    (min_l, max_l)
}

fn format_also(also: &[String]) -> String {
    if also.is_empty() {
        String::new()
    } else {
        format!("  also: {}", also.join(", "))
    }
}

// ── EXPLAIN INFER helpers ───────────────────────────────────────────────
//
// `exec_infer_trace` is a five-phase pipeline (load → forward → side
// tables → header → render). The helpers below split the side-table
// builders and the per-layer rendering loop out of the main function.

/// Build a `layer → top-3 attended (token, weight)` map from the
/// captured attention weights. Returns an empty map when
/// `with_attention` is false. Averages across all heads, drops special
/// tokens (BOS/EOS) by skipping `None` entries from `decode_token`, and
/// truncates to the top 3 by weight.
fn build_attention_map(
    captures: &[larql_inference::LayerAttentionCapture],
    token_strs: &[Option<String>],
    with_attention: bool,
) -> std::collections::HashMap<usize, Vec<(String, f32)>> {
    if !with_attention {
        return std::collections::HashMap::new();
    }
    let mut map = std::collections::HashMap::new();
    for cap in captures {
        let n_heads = cap.weights.heads.len();
        if n_heads == 0 || token_strs.is_empty() {
            continue;
        }
        let seq_len = cap.weights.heads[0].len();
        let mut avg = vec![0.0f32; seq_len];
        for head in &cap.weights.heads {
            for (j, &w) in head.iter().enumerate() {
                avg[j] += w;
            }
        }
        for v in avg.iter_mut() {
            *v /= n_heads as f32;
        }
        let mut pairs: Vec<(String, f32)> = avg
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(j, w)| {
                let tok = token_strs.get(j)?.as_ref()?;
                Some((tok.trim().to_string(), w))
            })
            .collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs.truncate(3);
        map.insert(cap.layer, pairs);
    }
    map
}

/// Build a `layer → (top_token, probability)` map by running the logit
/// lens on each captured residual. Returns empty when `with_attention`
/// is false (only the attention path captures intermediate residuals).
fn build_lens_map(
    lens_residuals: &[(usize, Vec<f32>)],
    weights: &larql_inference::ModelWeights,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    with_attention: bool,
) -> std::collections::HashMap<usize, (String, f64)> {
    if !with_attention {
        return std::collections::HashMap::new();
    }
    lens_residuals
        .iter()
        .filter_map(|(layer, residual)| {
            let pred = larql_inference::logit_lens_top1(weights, tokenizer, residual.as_slice())?;
            Some((*layer, pred))
        })
        .collect()
}

/// Resolve a `LayerBand` to a `(lo, hi)` filter on the trace layers.
/// Returns `None` for `All` / no band — the caller treats that as
/// "include every layer".
fn band_to_layer_range(
    band: Option<crate::ast::LayerBand>,
    bands: &larql_vindex::LayerBands,
) -> Option<(usize, usize)> {
    match band {
        Some(crate::ast::LayerBand::Syntax) => Some(bands.syntax),
        Some(crate::ast::LayerBand::Knowledge) => Some(bands.knowledge),
        Some(crate::ast::LayerBand::Output) => Some(bands.output),
        Some(crate::ast::LayerBand::All) | None => None,
    }
}

/// Render one layer's worth of trace hits, in either the compact
/// `with_attention` single-line format (top hit + attention + lens) or
/// the standard multi-line format (top-N hits with relation labels).
#[allow(clippy::too_many_arguments)]
fn render_trace_layer(
    out: &mut Vec<String>,
    layer: usize,
    hits: &[larql_vindex::WalkHit],
    classifier: Option<&crate::relations::RelationClassifier>,
    relations_only: bool,
    per_layer: usize,
    with_attention: bool,
    attention_map: &std::collections::HashMap<usize, Vec<(String, f32)>>,
    lens_map: &std::collections::HashMap<usize, (String, f64)>,
) {
    // When filtering to relations only, re-sort so positive gates rank
    // above negative gates of equal magnitude (positive gates correlate
    // with the prediction; negative gates with the opposite).
    let labelled_hits: Vec<&larql_vindex::WalkHit> = if relations_only {
        let mut lh: Vec<_> = hits
            .iter()
            .filter(|hit| {
                classifier
                    .and_then(|rc| rc.label_for_feature(layer, hit.feature))
                    .map(|l| !l.is_empty())
                    .unwrap_or(false)
            })
            .collect();
        lh.sort_by(|a, b| {
            let a_pos = a.gate_score > 0.0;
            let b_pos = b.gate_score > 0.0;
            match (a_pos, b_pos) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => b
                    .gate_score
                    .abs()
                    .partial_cmp(&a.gate_score.abs())
                    .unwrap_or(std::cmp::Ordering::Equal),
            }
        });
        lh
    } else {
        hits.iter().collect()
    };

    if with_attention {
        // Compact single-line format: feature + attention + logit lens.
        let hit = labelled_hits.first();
        let feature_part = if let Some(hit) = hit {
            let label = classifier
                .and_then(|rc| rc.label_for_feature(layer, hit.feature))
                .unwrap_or("");
            if relations_only && label.is_empty() {
                None
            } else {
                let top_token = hit.meta.top_token.trim();
                let name = if !label.is_empty() { label } else { top_token };
                Some(format!("{:<14} {:+.1}", name, hit.gate_score))
            }
        } else {
            None
        };
        let empty = format!("{:19}", "");
        let feature_str = feature_part.as_deref().unwrap_or(&empty);

        let attn_part = attention_map
            .get(&layer)
            .and_then(|attn| attn.first())
            .map(|(tok, w)| format!("{}({:.0}%)", tok, w * 100.0))
            .unwrap_or_default();

        let lens_part = lens_map
            .get(&layer)
            .map(|(tok, prob)| format!("{} ({:.1}%)", tok, prob * 100.0))
            .unwrap_or_default();

        if feature_part.is_some() || !lens_part.is_empty() {
            out.push(format!(
                "  L{:2}  {:<19}  {:<16} → {}",
                layer, feature_str, attn_part, lens_part,
            ));
        }
    } else {
        // Standard multi-line format without attention.
        let mut shown = 0;
        for hit in &labelled_hits {
            if shown >= per_layer {
                break;
            }
            let label = classifier
                .and_then(|rc| rc.label_for_feature(layer, hit.feature))
                .unwrap_or("");
            if relations_only && label.is_empty() {
                continue;
            }
            shown += 1;
            let label_str = if label.is_empty() {
                format!("{:14}", "")
            } else {
                format!("{:<14}", label)
            };
            let top_token = hit.meta.top_token.trim();
            let down_top: String = hit
                .meta
                .top_k
                .iter()
                .take(3)
                .map(|t| t.token.clone())
                .collect::<Vec<_>>()
                .join(", ");
            out.push(format!(
                "  L{:2}: {} F{:<5} gate={:+.1}  → {:15} [{}]",
                layer, label_str, hit.feature, hit.gate_score, top_token, down_top,
            ));
        }
    }
}
