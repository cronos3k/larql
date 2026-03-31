/// Query executor: WALK, INFER, SELECT, DESCRIBE, EXPLAIN.

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
        _mode: Option<WalkMode>,
        _compare: bool,
    ) -> Result<Vec<String>, LqlError> {
        let (path, _config, index) = self.require_vindex()?;
        let top_k = top.unwrap_or(10) as usize;

        let (embed, embed_scale) = larql_vindex::load_vindex_embeddings(path)
            .map_err(|e| LqlError::Execution(format!("failed to load embeddings: {e}")))?;
        let tokenizer = larql_vindex::load_vindex_tokenizer(path)
            .map_err(|e| LqlError::Execution(format!("failed to load tokenizer: {e}")))?;

        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| LqlError::Execution(format!("tokenize error: {e}")))?;
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

        let all_layers = index.loaded_layers();
        let walk_layers: Vec<usize> = if let Some(range) = layers {
            (range.start as usize..=range.end as usize)
                .filter(|l| all_layers.contains(l))
                .collect()
        } else {
            all_layers
        };

        let start = std::time::Instant::now();
        let trace = index.walk(&query, &walk_layers, top_k);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        let mut out = Vec::new();
        out.push(format!(
            "Feature scan for {:?} (token {:?}, {} layers)",
            prompt,
            token_str.trim(),
            walk_layers.len(),
        ));
        out.push(String::new());

        for (layer, hits) in &trace.layers {
            if hits.is_empty() {
                continue;
            }
            for hit in hits.iter().take(3) {
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
        out.push(String::new());
        out.push("Note: pure vindex scan (no attention). For inference use INFER.".into());

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
        let (path, config, index) = self.require_vindex()?;
        let top_k = top.unwrap_or(5) as usize;

        if !config.has_model_weights {
            return Err(LqlError::Execution(format!(
                "INFER requires model weights. This vindex was built without --include-weights.\n\
                 Options:\n\
                 1. Rebuild: larql extract-index --model \"{}\" --output \"{}\" --include-weights\n\
                 2. Use the CLI: larql walk --index \"{}\" --predict --model \"{}\"",
                config.model,
                path.display(),
                path.display(),
                config.model,
            )));
        }

        let mut cb = larql_vindex::SilentLoadCallbacks;
        let weights = larql_inference::load_model_weights_from_vindex(path, &mut cb)
            .map_err(|e| LqlError::Execution(format!("failed to load model weights: {e}")))?;
        let tokenizer = larql_vindex::load_vindex_tokenizer(path)
            .map_err(|e| LqlError::Execution(format!("failed to load tokenizer: {e}")))?;

        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| LqlError::Execution(format!("tokenize error: {e}")))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        let walk_ffn = larql_inference::vindex::WalkFfn::new(&weights, index, 10);
        let start = std::time::Instant::now();
        let result = larql_inference::predict_with_ffn(
            &weights,
            &tokenizer,
            &token_ids,
            top_k,
            &walk_ffn,
        );
        let walk_ms = start.elapsed().as_secs_f64() * 1000.0;

        let trace = walk_ffn.take_trace();

        let mut out = Vec::new();
        out.push("Predictions (walk FFN):".into());
        for (i, (tok, prob)) in result.predictions.iter().enumerate() {
            out.push(format!(
                "  {:2}. {:20} ({:.2}%)",
                i + 1,
                tok,
                prob * 100.0
            ));
        }
        out.push(format!("  {:.0}ms", walk_ms));

        out.push(String::new());
        out.push("Inference trace (features that fired with attention):".into());
        let classifier = self.relation_classifier();
        for (layer, hits) in &trace.layers {
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
                let down_top: String = hit
                    .meta
                    .top_k
                    .iter()
                    .take(3)
                    .map(|t| t.token.clone())
                    .collect::<Vec<_>>()
                    .join(", ");
                out.push(format!(
                    "  L{:2}: {} F{:<5} gate={:+.1}  → [{}]",
                    layer, label_str, hit.feature, hit.gate_score, down_top,
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
        _relations_only: bool,
        verbose: bool,
    ) -> Result<Vec<String>, LqlError> {
        let (path, _config, index) = self.require_vindex()?;

        let (embed, embed_scale) = larql_vindex::load_vindex_embeddings(path)
            .map_err(|e| LqlError::Execution(format!("failed to load embeddings: {e}")))?;
        let tokenizer = larql_vindex::load_vindex_tokenizer(path)
            .map_err(|e| LqlError::Execution(format!("failed to load tokenizer: {e}")))?;

        let encoding = tokenizer
            .encode(entity, false)
            .map_err(|e| LqlError::Execution(format!("tokenize error: {e}")))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        if token_ids.is_empty() {
            return Ok(vec![format!("{entity}\n  (not found)")]);
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

        let all_layers = index.loaded_layers();

        // Apply band + layer filter
        let scan_layers: Vec<usize> = if let Some(l) = layer {
            vec![l as usize]
        } else {
            match band {
                Some(crate::ast::LayerBand::Syntax) => {
                    all_layers.iter().copied().filter(|l| *l <= 13).collect()
                }
                Some(crate::ast::LayerBand::Knowledge) => {
                    all_layers.iter().copied().filter(|l| *l >= 14 && *l <= 27).collect()
                }
                Some(crate::ast::LayerBand::Output) => {
                    all_layers.iter().copied().filter(|l| *l >= 28).collect()
                }
                Some(crate::ast::LayerBand::All) | None => all_layers,
            }
        };

        let trace = index.walk(&query, &scan_layers, 20);

        struct EdgeInfo {
            gate: f32,
            layers: Vec<usize>,
            count: usize,
            original: String,
            also: Vec<String>,
            best_layer: usize,
            best_feature: usize,
        }

        let entity_lower = entity.to_lowercase();
        let mut edges: HashMap<String, EdgeInfo> = HashMap::new();
        let gate_threshold = 5.0_f32;

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

                let also_readable: Vec<String> = hit.meta.top_k.iter()
                    .filter(|t| {
                        t.token.to_lowercase() != tok.to_lowercase()
                            && t.token.to_lowercase() != entity_lower
                            && super::helpers::is_readable_token(&t.token)
                            && t.logit > 0.0
                    })
                    .take(5)
                    .map(|t| t.token.clone())
                    .collect();

                let also: Vec<String> = also_readable.iter()
                    .filter(|t| is_content_token(t))
                    .take(3)
                    .cloned()
                    .collect();

                // Coherence filter: skip weak edges with no content secondaries
                if also.is_empty() && !also_readable.is_empty() && hit.gate_score < 20.0 {
                    continue;
                }

                let key = tok.to_lowercase();
                let entry = edges.entry(key).or_insert_with(|| {
                    EdgeInfo {
                        gate: 0.0,
                        layers: Vec::new(),
                        count: 0,
                        original: tok.to_string(),
                        also,
                        best_layer: *layer_idx,
                        best_feature: hit.feature,
                    }
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

        let mut ranked: Vec<&EdgeInfo> = edges.values().collect();
        ranked.sort_by(|a, b| b.gate.partial_cmp(&a.gate).unwrap_or(std::cmp::Ordering::Equal));

        let mut out = vec![entity.to_string()];

        if ranked.is_empty() {
            out.push("  (no edges found)".into());
            return Ok(out);
        }

        let classifier = self.relation_classifier();

        // Resolve labels for all edges
        struct FormattedEdge {
            label: String,       // clean probe label, raw cluster label, or empty
            is_probe: bool,
            is_cluster: bool,
            target: String,
            gate: f32,
            primary_layer: usize,
            layers: Vec<usize>,
            count: usize,
            also: Vec<String>,
        }

        let formatted: Vec<FormattedEdge> = ranked.iter().map(|info| {
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
        }).collect();

        // Split into bands
        let mut syntax = Vec::new();
        let mut knowledge = Vec::new();
        let mut output_band = Vec::new();

        for edge in &formatted {
            let primary = edge.primary_layer;
            if primary <= 13 {
                syntax.push(edge);
            } else if primary <= 27 {
                knowledge.push(edge);
            } else {
                output_band.push(edge);
            }
        }

        // ── Format edges ──
        // Default mode: clean, scannable, demo-ready. Top 10 per band.
        //   Probe labels shown. Cluster labels hidden (blank). No also-tokens.
        //   Single primary layer. Fixed-width columns.
        // Verbose mode: everything. Raw cluster labels. Also-tokens. Layer ranges.

        let max_edges = if verbose { 30 } else { 10 };

        let format_edge = |edge: &FormattedEdge| -> String {
            if verbose {
                // Verbose: show all labels, also-tokens, layer ranges, counts
                let label = if edge.label.is_empty() {
                    format!("{:<14}", "")
                } else {
                    format!("{:<14}", edge.label)
                };

                let tag = if edge.is_probe {
                    "  (probe)"
                } else if edge.is_cluster {
                    "  (cluster)"
                } else {
                    ""
                };

                let min_l = *edge.layers.iter().min().unwrap_or(&0);
                let max_l = *edge.layers.iter().max().unwrap_or(&0);
                let layer_str = if min_l == max_l {
                    format!("L{}", min_l)
                } else {
                    format!("L{}-{}", min_l, max_l)
                };

                let also = if edge.also.is_empty() {
                    String::new()
                } else {
                    format!("  also: {}", edge.also.join(", "))
                };

                format!(
                    "    {} → {:20} {:>7.1}  {:<8} {}x{}{}",
                    label, edge.target, edge.gate, layer_str,
                    edge.count, tag, also,
                )
            } else {
                // Default: clean columns, probe labels only, primary layer
                let label = if edge.is_probe {
                    format!("{:<12}", edge.label)
                } else {
                    format!("{:<12}", "")
                };

                let tag = if edge.is_probe { "  (probe)" } else { "" };

                format!(
                    "    {} → {:20} {:>7.1}  L{:<3}{}",
                    label, edge.target, edge.gate, edge.primary_layer, tag,
                )
            }
        };

        if !syntax.is_empty() {
            out.push("  Syntax (L0-13):".into());
            for edge in syntax.iter().take(max_edges) {
                out.push(format_edge(edge));
            }
        }
        if !knowledge.is_empty() {
            out.push("  Edges (L14-27):".into());
            for edge in knowledge.iter().take(max_edges) {
                out.push(format_edge(edge));
            }
        }
        if !output_band.is_empty() {
            out.push("  Output (L28-33):".into());
            for edge in output_band.iter().take(if verbose { max_edges } else { 5 }) {
                out.push(format_edge(edge));
            }
        }

        Ok(out)
    }

    // ── SELECT ──

    pub(crate) fn exec_select(
        &self,
        _fields: &[Field],
        conditions: &[Condition],
        _nearest: Option<&NearestClause>,
        order: Option<&OrderBy>,
        limit: Option<u32>,
    ) -> Result<Vec<String>, LqlError> {
        let (_path, _config, index) = self.require_vindex()?;

        let all_layers = index.loaded_layers();
        let limit = limit.unwrap_or(20) as usize;

        let entity_filter = conditions.iter().find(|c| c.field == "entity").and_then(|c| {
            if let Value::String(ref s) = c.value { Some(s.as_str()) } else { None }
        });
        let _relation_filter = conditions.iter().find(|c| c.field == "relation").and_then(|c| {
            if let Value::String(ref s) = c.value { Some(s.as_str()) } else { None }
        });
        let layer_filter = conditions.iter().find(|c| c.field == "layer").and_then(|c| {
            if let Value::Integer(n) = c.value { Some(n as usize) } else { None }
        });
        let feature_filter = conditions.iter().find(|c| c.field == "feature").and_then(|c| {
            if let Value::Integer(n) = c.value { Some(n as usize) } else { None }
        });

        struct Row {
            layer: usize,
            feature: usize,
            top_token: String,
            c_score: f32,
        }

        let mut rows: Vec<Row> = Vec::new();

        let scan_layers: Vec<usize> = if let Some(l) = layer_filter {
            vec![l]
        } else {
            all_layers
        };

        for layer in &scan_layers {
            if let Some(metas) = index.down_meta_at(*layer) {
                for (feat_idx, meta_opt) in metas.iter().enumerate() {
                    if let Some(feature_f) = feature_filter {
                        if feat_idx != feature_f {
                            continue;
                        }
                    }
                    if let Some(meta) = meta_opt {
                        if let Some(ent) = entity_filter {
                            if !meta.top_token.to_lowercase().contains(&ent.to_lowercase()) {
                                continue;
                            }
                        }
                        rows.push(Row {
                            layer: *layer,
                            feature: feat_idx,
                            top_token: meta.top_token.clone(),
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

        rows.truncate(limit);

        let mut out = Vec::new();
        out.push(format!(
            "{:<8} {:<8} {:<20} {:>10}",
            "Layer", "Feature", "Token", "Score"
        ));
        out.push("-".repeat(50));

        for row in &rows {
            out.push(format!(
                "L{:<7} F{:<7} {:20} {:>10.4}",
                row.layer, row.feature, row.top_token, row.c_score
            ));
        }

        if rows.is_empty() {
            out.push("  (no matching edges)".into());
        }

        Ok(out)
    }

    // ── EXPLAIN ──

    pub(crate) fn exec_explain(
        &self,
        prompt: &str,
        layers: Option<&Range>,
        _verbose: bool,
    ) -> Result<Vec<String>, LqlError> {
        let (path, _config, index) = self.require_vindex()?;

        let (embed, embed_scale) = larql_vindex::load_vindex_embeddings(path)
            .map_err(|e| LqlError::Execution(format!("failed to load embeddings: {e}")))?;
        let tokenizer = larql_vindex::load_vindex_tokenizer(path)
            .map_err(|e| LqlError::Execution(format!("failed to load tokenizer: {e}")))?;

        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| LqlError::Execution(format!("tokenize error: {e}")))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        if token_ids.is_empty() {
            return Err(LqlError::Execution("empty prompt".into()));
        }

        let last_tok = *token_ids.last().unwrap();
        let embed_row = embed.row(last_tok as usize);
        let query: larql_vindex::ndarray::Array1<f32> =
            embed_row.mapv(|v| v * embed_scale);

        let all_layers = index.loaded_layers();
        let walk_layers: Vec<usize> = if let Some(range) = layers {
            (range.start as usize..=range.end as usize)
                .filter(|l| all_layers.contains(l))
                .collect()
        } else {
            all_layers
        };

        let trace = index.walk(&query, &walk_layers, 5);

        let mut out = Vec::new();
        for (layer, hits) in &trace.layers {
            for hit in hits {
                let down_tokens: String = hit
                    .meta
                    .top_k
                    .iter()
                    .take(3)
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
    ) -> Result<Vec<String>, LqlError> {
        let (path, config, index) = self.require_vindex()?;
        let top_k = top.unwrap_or(5) as usize;

        if !config.has_model_weights {
            return Err(LqlError::Execution(
                "EXPLAIN INFER requires model weights. Rebuild with --include-weights.".into(),
            ));
        }

        let mut cb = larql_vindex::SilentLoadCallbacks;
        let weights = larql_inference::load_model_weights_from_vindex(path, &mut cb)
            .map_err(|e| LqlError::Execution(format!("failed to load model weights: {e}")))?;
        let tokenizer = larql_vindex::load_vindex_tokenizer(path)
            .map_err(|e| LqlError::Execution(format!("failed to load tokenizer: {e}")))?;

        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| LqlError::Execution(format!("tokenize error: {e}")))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        let walk_ffn = larql_inference::vindex::WalkFfn::new(&weights, index, 10);
        let start = std::time::Instant::now();
        let result = larql_inference::predict_with_ffn(
            &weights, &tokenizer, &token_ids, top_k, &walk_ffn,
        );
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        let trace = walk_ffn.take_trace();
        let classifier = self.relation_classifier();

        let mut out = Vec::new();
        out.push(format!("Inference trace for {:?}:", prompt));
        out.push(format!(
            "Prediction: {} ({:.2}%) in {:.0}ms",
            result.predictions.first().map(|(t, _)| t.as_str()).unwrap_or("?"),
            result.predictions.first().map(|(_, p)| p * 100.0).unwrap_or(0.0),
            elapsed_ms
        ));
        out.push(String::new());

        for (layer, hits) in &trace.layers {
            if hits.is_empty() {
                continue;
            }
            for hit in hits.iter().take(3) {
                let label = classifier
                    .and_then(|rc| rc.label_for_feature(*layer, hit.feature))
                    .unwrap_or("");
                let label_str = if label.is_empty() {
                    format!("{:14}", "")
                } else {
                    format!("{:<14}", label)
                };
                let down_top: String = hit
                    .meta
                    .top_k
                    .iter()
                    .take(3)
                    .map(|t| t.token.clone())
                    .collect::<Vec<_>>()
                    .join(", ");
                out.push(format!(
                    "  L{:2}: {} F{:<5} gate={:+.1}  → [{}]",
                    layer, label_str, hit.feature, hit.gate_score, down_top,
                ));
            }
        }

        Ok(out)
    }
}
