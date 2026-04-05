/// Remote executor — forwards LQL queries to a larql-server via HTTP.

use crate::ast::*;
use crate::error::LqlError;
use super::Session;
use super::Backend;

impl Session {
    /// Connect to a remote larql-server.
    pub(crate) fn exec_use_remote(&mut self, url: &str) -> Result<Vec<String>, LqlError> {
        let url = url.trim_end_matches('/').to_string();

        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| LqlError::Execution(format!("failed to create HTTP client: {e}")))?;

        // Verify the server is reachable by hitting /v1/stats.
        let stats_url = format!("{url}/v1/stats");
        let resp = client
            .get(&stats_url)
            .send()
            .map_err(|e| LqlError::Execution(format!("failed to connect to {url}: {e}")))?;

        if !resp.status().is_success() {
            return Err(LqlError::Execution(format!(
                "server returned {}: {}",
                resp.status(),
                resp.text().unwrap_or_default()
            )));
        }

        let stats: serde_json::Value = resp
            .json()
            .map_err(|e| LqlError::Execution(format!("invalid response from server: {e}")))?;

        let model = stats["model"].as_str().unwrap_or("unknown");
        let layers = stats["layers"].as_u64().unwrap_or(0);
        let features = stats["features"].as_u64().unwrap_or(0);

        // Generate a unique session ID for this connection
        let session_id = format!("larql-{}-{}", std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis());

        self.backend = Backend::Remote {
            url: url.clone(),
            client,
            local_patches: Vec::new(),
            session_id,
        };
        self.patch_recording = None;
        self.auto_patch = false;

        Ok(vec![format!(
            "Connected: {} ({} layers, {} features)\n  Remote: {}",
            model, layers, features, url,
        )])
    }

    /// Check if the backend is remote.
    pub(crate) fn is_remote(&self) -> bool {
        matches!(&self.backend, Backend::Remote { .. })
    }

    /// Get the remote URL, client, and session ID, or error.
    fn require_remote(&self) -> Result<(&str, &reqwest::blocking::Client, &str), LqlError> {
        match &self.backend {
            Backend::Remote { url, client, session_id, .. } => Ok((url, client, session_id)),
            _ => Err(LqlError::Execution("not connected to a remote server".into())),
        }
    }


    // ── Remote query forwarding ──

    pub(crate) fn remote_describe(
        &self,
        entity: &str,
        band: Option<LayerBand>,
        mode: crate::ast::DescribeMode,
    ) -> Result<Vec<String>, LqlError> {
        let (url, client, _sid) = self.require_remote()?;
        let verbose = mode == crate::ast::DescribeMode::Verbose;
        let show_also = matches!(mode, crate::ast::DescribeMode::Verbose | crate::ast::DescribeMode::Raw);

        let band_str = match band {
            Some(LayerBand::Syntax) => "syntax",
            Some(LayerBand::Knowledge) => "knowledge",
            Some(LayerBand::Output) => "output",
            Some(LayerBand::All) => "all",
            None => "knowledge",
        };

        let resp = client
            .get(format!("{url}/v1/describe"))
            .query(&[("entity", entity), ("band", band_str), ("verbose", if verbose { "true" } else { "false" })])
            .send()
            .map_err(|e| LqlError::Execution(format!("request failed: {e}")))?;

        let body: serde_json::Value = resp
            .json()
            .map_err(|e| LqlError::Execution(format!("invalid response: {e}")))?;

        let mut out = vec![entity.to_string()];

        if let Some(edges) = body["edges"].as_array() {
            if edges.is_empty() {
                out.push("  (no edges found)".into());
            } else {
                for edge in edges {
                    let target = edge["target"].as_str().unwrap_or("?");
                    let gate = edge["gate_score"].as_f64().unwrap_or(0.0);
                    let layer = edge["layer"].as_u64().unwrap_or(0);
                    let relation = edge["relation"].as_str().unwrap_or("");
                    let source = edge["source"].as_str().unwrap_or("");

                    let show_labels = mode != crate::ast::DescribeMode::Raw;
                    let label = if show_labels && !relation.is_empty() {
                        format!("{:<12}", relation)
                    } else {
                        format!("{:<12}", "")
                    };

                    let tag = if show_labels && source == "probe" { "  (probe)" } else { "" };

                    let also_str = if show_also {
                        edge["also"].as_array()
                            .map(|arr| arr.iter()
                                .filter_map(|v| v.as_str())
                                .collect::<Vec<_>>()
                                .join(", "))
                            .filter(|s| !s.is_empty())
                            .map(|s| format!("  also: {s}"))
                            .unwrap_or_default()
                    } else {
                        String::new()
                    };

                    out.push(format!(
                        "    {} → {:20} {:>7.1}  L{:<3}{}{}",
                        label, target, gate, layer, tag, also_str,
                    ));
                }
            }
        }

        if let Some(ms) = body["latency_ms"].as_f64() {
            out.push(format!("\n{:.1}ms (remote)", ms));
        }

        // Overlay local patch edges.
        if let Backend::Remote { local_patches, .. } = &self.backend {
            if !local_patches.is_empty() {
                let entity_lower = entity.to_lowercase();
                let mut local_edges = Vec::new();
                for patch in local_patches {
                    for op in &patch.operations {
                        if let larql_vindex::PatchOp::Insert {
                            entity: ent,
                            target,
                            relation,
                            layer,
                            confidence,
                            ..
                        } = op
                        {
                            if ent.to_lowercase() == entity_lower {
                                local_edges.push((
                                    relation.as_deref().unwrap_or(""),
                                    target.as_str(),
                                    *layer,
                                    confidence.unwrap_or(0.9),
                                ));
                            }
                        }
                    }
                }
                if !local_edges.is_empty() {
                    out.push("  Local patch edges:".into());
                    for (relation, target, layer, conf) in &local_edges {
                        let label = if relation.is_empty() {
                            format!("{:<12}", "")
                        } else {
                            format!("{:<12}", relation)
                        };
                        out.push(format!(
                            "    {} → {:20} {:>7.2}  L{:<3}  (local)",
                            label, target, conf, layer,
                        ));
                    }
                }
            }
        }

        Ok(out)
    }

    pub(crate) fn remote_walk(
        &self,
        prompt: &str,
        top: Option<u32>,
        layers: Option<&Range>,
    ) -> Result<Vec<String>, LqlError> {
        let (url, client, _sid) = self.require_remote()?;

        let top_k = top.unwrap_or(10);
        let mut params = vec![
            ("prompt".to_string(), prompt.to_string()),
            ("top".to_string(), top_k.to_string()),
        ];
        if let Some(r) = layers {
            params.push(("layers".to_string(), format!("{}-{}", r.start, r.end)));
        }

        let resp = client
            .get(format!("{url}/v1/walk"))
            .query(&params)
            .send()
            .map_err(|e| LqlError::Execution(format!("request failed: {e}")))?;

        let body: serde_json::Value = resp
            .json()
            .map_err(|e| LqlError::Execution(format!("invalid response: {e}")))?;

        let mut out = Vec::new();
        out.push(format!("Feature scan for {:?}", prompt));
        out.push(String::new());

        if let Some(hits) = body["hits"].as_array() {
            for hit in hits {
                let layer = hit["layer"].as_u64().unwrap_or(0);
                let feature = hit["feature"].as_u64().unwrap_or(0);
                let gate = hit["gate_score"].as_f64().unwrap_or(0.0);
                let target = hit["target"].as_str().unwrap_or("?");

                out.push(format!(
                    "  L{:2}: F{:<5} gate={:+.1}  top={:?}",
                    layer, feature, gate, target,
                ));
            }
        }

        if let Some(ms) = body["latency_ms"].as_f64() {
            out.push(format!("\n{:.1}ms (remote)", ms));
        }

        Ok(out)
    }

    pub(crate) fn remote_infer(
        &self,
        prompt: &str,
        top: Option<u32>,
        compare: bool,
    ) -> Result<Vec<String>, LqlError> {
        let (url, client, sid) = self.require_remote()?;

        let mode = if compare { "compare" } else { "walk" };
        let body = serde_json::json!({
            "prompt": prompt,
            "top": top.unwrap_or(5),
            "mode": mode,
        });

        let resp = client
            .post(format!("{url}/v1/infer"))
            .header("x-session-id", sid)
            .json(&body)
            .send()
            .map_err(|e| LqlError::Execution(format!("request failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().unwrap_or_default();
            return Err(LqlError::Execution(format!("infer failed ({}): {}", status, text)));
        }

        let result: serde_json::Value = resp
            .json()
            .map_err(|e| LqlError::Execution(format!("invalid response: {e}")))?;

        let mut out = Vec::new();

        if compare {
            for mode in &["walk", "dense"] {
                if let Some(preds) = result[mode].as_array() {
                    out.push(format!("Predictions ({mode}):"));
                    for (i, p) in preds.iter().enumerate() {
                        let tok = p["token"].as_str().unwrap_or("?");
                        let prob = p["probability"].as_f64().unwrap_or(0.0);
                        out.push(format!("  {:2}. {:20} ({:.2}%)", i + 1, tok, prob * 100.0));
                    }
                    if let Some(ms) = result[format!("{mode}_ms")].as_f64() {
                        out.push(format!("  {:.0}ms", ms));
                    }
                    out.push(String::new());
                }
            }
        } else if let Some(preds) = result["predictions"].as_array() {
            out.push("Predictions (walk FFN):".into());
            for (i, p) in preds.iter().enumerate() {
                let tok = p["token"].as_str().unwrap_or("?");
                let prob = p["probability"].as_f64().unwrap_or(0.0);
                out.push(format!("  {:2}. {:20} ({:.2}%)", i + 1, tok, prob * 100.0));
            }
        }

        if let Some(ms) = result["latency_ms"].as_f64() {
            out.push(format!("{:.0}ms (remote)", ms));
        }

        Ok(out)
    }

    pub(crate) fn remote_explain_infer(
        &self,
        prompt: &str,
        top: Option<u32>,
        band: Option<LayerBand>,
        relations_only: bool,
        with_attention: bool,
    ) -> Result<Vec<String>, LqlError> {
        let (url, client, _sid) = self.require_remote()?;

        let per_layer = top.unwrap_or(3);
        let band_str = match band {
            Some(LayerBand::Syntax) => "syntax",
            Some(LayerBand::Knowledge) => "knowledge",
            Some(LayerBand::Output) => "output",
            Some(LayerBand::All) => "all",
            None => "all",
        };

        let body = serde_json::json!({
            "prompt": prompt,
            "top": top.unwrap_or(5),
            "per_layer": per_layer,
            "band": band_str,
            "relations_only": relations_only,
            "with_attention": with_attention,
        });

        let resp = client
            .post(format!("{url}/v1/explain-infer"))
            .json(&body)
            .send()
            .map_err(|e| LqlError::Execution(format!("request failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().unwrap_or_default();
            return Err(LqlError::Execution(format!("explain-infer failed ({}): {}", status, text)));
        }

        let result: serde_json::Value = resp
            .json()
            .map_err(|e| LqlError::Execution(format!("invalid response: {e}")))?;

        let band_label = match band {
            Some(LayerBand::Syntax) => " (syntax)",
            Some(LayerBand::Knowledge) => " (knowledge)",
            Some(LayerBand::Output) => " (output)",
            _ => "",
        };

        let mut out = Vec::new();
        out.push(format!("Inference trace for {:?}{}:", prompt, band_label));

        if let Some(preds) = result["predictions"].as_array() {
            if let Some(first) = preds.first() {
                let tok = first["token"].as_str().unwrap_or("?");
                let prob = first["probability"].as_f64().unwrap_or(0.0);
                out.push(format!("Prediction: {} ({:.2}%)", tok, prob * 100.0));
            }
        }
        out.push(String::new());

        if let Some(layers) = result["trace"].as_array() {
            for layer_obj in layers {
                let layer = layer_obj["layer"].as_u64().unwrap_or(0);
                let features = layer_obj["features"].as_array();

                if with_attention {
                    // Compact single-line format
                    let feat = features.and_then(|f| f.first());
                    let feature_str = if let Some(feat) = feat {
                        let relation = feat["relation"].as_str()
                            .or_else(|| feat["relation"].as_null().map(|_| ""))
                            .unwrap_or("");
                        if relations_only && relation.is_empty() {
                            None
                        } else {
                            let gate = feat["gate_score"].as_f64().unwrap_or(0.0);
                            let top_token = feat["top_token"].as_str().unwrap_or("?");
                            let name = if !relation.is_empty() { relation } else { top_token };
                            Some(format!("{:<14} {:+.1}", name, gate))
                        }
                    } else {
                        None
                    };
                    let empty = format!("{:19}", "");
                    let feature_part = feature_str.as_deref().unwrap_or(&empty);

                    let attn_part = layer_obj.get("attention")
                        .and_then(|a| a.as_array())
                        .and_then(|arr| arr.first())
                        .and_then(|v| {
                            let tok = v["token"].as_str()?;
                            let w = v["weight"].as_f64()?;
                            Some(format!("{}({:.0}%)", tok, w * 100.0))
                        })
                        .unwrap_or_default();

                    let lens_part = layer_obj.get("lens")
                        .and_then(|l| {
                            let tok = l["token"].as_str()?;
                            let prob = l["probability"].as_f64()?;
                            Some(format!("{} ({:.1}%)", tok, prob * 100.0))
                        })
                        .unwrap_or_default();

                    if feature_str.is_some() || !lens_part.is_empty() {
                        out.push(format!(
                            "  L{:2}  {:<19}  {:<16} → {}",
                            layer, feature_part, attn_part, lens_part,
                        ));
                    }
                } else {
                    // Standard multi-line format
                    if let Some(features) = features {
                        for feat in features {
                            let feature = feat["feature"].as_u64().unwrap_or(0);
                            let gate = feat["gate_score"].as_f64().unwrap_or(0.0);
                            let relation = feat["relation"].as_str()
                                .or_else(|| feat["relation"].as_null().map(|_| ""))
                                .unwrap_or("");
                            if relations_only && relation.is_empty() {
                                continue;
                            }
                            let label_str = if relation.is_empty() {
                                format!("{:14}", "")
                            } else {
                                format!("{:<14}", relation)
                            };
                            let top_token = feat["top_token"].as_str().unwrap_or("?");
                            let top_tokens: String = feat["top_tokens"]
                                .as_array()
                                .map(|arr| {
                                    arr.iter()
                                        .filter_map(|v| v.as_str())
                                        .collect::<Vec<_>>()
                                        .join(", ")
                                })
                                .unwrap_or_default();
                            out.push(format!(
                                "  L{:2}: {} F{:<5} gate={:+.1}  → {:15} [{}]",
                                layer, label_str, feature, gate, top_token, top_tokens,
                            ));
                        }
                    }
                }
            }
        }

        if let Some(ms) = result["latency_ms"].as_f64() {
            out.push(format!("\n{:.0}ms (remote)", ms));
        }

        Ok(out)
    }

    pub(crate) fn remote_stats(&self) -> Result<Vec<String>, LqlError> {
        let (url, client, _sid) = self.require_remote()?;

        let resp = client
            .get(format!("{url}/v1/stats"))
            .send()
            .map_err(|e| LqlError::Execution(format!("request failed: {e}")))?;

        let body: serde_json::Value = resp
            .json()
            .map_err(|e| LqlError::Execution(format!("invalid response: {e}")))?;

        let mut out = Vec::new();
        out.push(format!("Model: {}", body["model"].as_str().unwrap_or("?")));
        out.push(format!("Family: {}", body["family"].as_str().unwrap_or("?")));
        out.push(format!("Layers: {}", body["layers"].as_u64().unwrap_or(0)));
        out.push(format!("Features: {}", body["features"].as_u64().unwrap_or(0)));
        out.push(format!("Hidden: {}", body["hidden_size"].as_u64().unwrap_or(0)));
        out.push(format!("Dtype: {}", body["dtype"].as_str().unwrap_or("?")));
        out.push(format!("Extract level: {}", body["extract_level"].as_str().unwrap_or("?")));

        if let Some(bands) = body.get("layer_bands") {
            if let (Some(s), Some(k), Some(o)) = (
                bands.get("syntax"),
                bands.get("knowledge"),
                bands.get("output"),
            ) {
                out.push(format!(
                    "Bands: syntax {}-{}, knowledge {}-{}, output {}-{}",
                    s[0], s[1], k[0], k[1], o[0], o[1]
                ));
            }
        }

        if let Some(loaded) = body.get("loaded") {
            out.push(format!(
                "Loaded: browse={}, inference={}",
                loaded["browse"].as_bool().unwrap_or(false),
                loaded["inference"].as_bool().unwrap_or(false),
            ));
        }

        out.push(format!("Remote: {url}"));

        Ok(out)
    }

    pub(crate) fn remote_show_relations(&self, mode: crate::ast::DescribeMode, with_examples: bool) -> Result<Vec<String>, LqlError> {
        use crate::ast::DescribeMode;
        let (url, client, _sid) = self.require_remote()?;

        let resp = client
            .get(format!("{url}/v1/relations"))
            .send()
            .map_err(|e| LqlError::Execution(format!("request failed: {e}")))?;

        let body: serde_json::Value = resp
            .json()
            .map_err(|e| LqlError::Execution(format!("invalid response: {e}")))?;

        let mut out = Vec::new();

        // Probe-confirmed relations (skip for Raw mode)
        if mode != DescribeMode::Raw {
            if let Some(probes) = body["probe_relations"].as_array() {
                if !probes.is_empty() {
                    let probe_count = body["probe_count"].as_u64().unwrap_or(0);
                    out.push(format!("Probe-confirmed relations ({probe_count} labels):"));
                    out.push(format!("{:<25} {:>8}", "Relation", "Features"));
                    out.push("-".repeat(35));
                    for rel in probes {
                        let name = rel["name"].as_str().unwrap_or("?");
                        let count = rel["count"].as_u64().unwrap_or(0);
                        out.push(format!("{:<25} {:>8}", name, count));
                    }
                    out.push(String::new());
                }
            }
        }

        // Raw token relations (show for Verbose, Raw, or when no probes)
        let show_raw = mode == DescribeMode::Raw
            || mode == DescribeMode::Verbose
            || out.is_empty();

        if show_raw {
            if let Some(rels) = body["relations"].as_array() {
                if !rels.is_empty() {
                    out.push("Top output tokens:".to_string());
                    out.push(format!(
                        "{:<25} {:>8} {:>8} {:>10}",
                        "Token", "Count", "Score", "Layers"
                    ));
                    out.push("-".repeat(55));
                    for rel in rels {
                        let name = rel["name"].as_str().unwrap_or("?");
                        let count = rel["count"].as_u64().unwrap_or(0);
                        let score = rel["max_score"].as_f64().unwrap_or(0.0);
                        let min_l = rel["min_layer"].as_u64().unwrap_or(0);
                        let max_l = rel["max_layer"].as_u64().unwrap_or(0);
                        let examples_str = if with_examples {
                            if let Some(arr) = rel["examples"].as_array() {
                                let ex: Vec<&str> = arr.iter()
                                    .filter_map(|v| v.as_str())
                                    .collect();
                                if ex.is_empty() { String::new() }
                                else { format!("  e.g. {}", ex.join(", ")) }
                            } else {
                                String::new()
                            }
                        } else {
                            String::new()
                        };
                        out.push(format!(
                            "{:<25} {:>8} {:>8.2} {:>5}-{}{}",
                            name, count, score, min_l, max_l, examples_str,
                        ));
                    }
                }
            }
        }

        Ok(out)
    }

    // ── Remote mutations (forwarded to server as patches) ──

    pub(crate) fn remote_insert(
        &self,
        entity: &str,
        relation: &str,
        target: &str,
        layer: Option<u32>,
        confidence: Option<f32>,
    ) -> Result<Vec<String>, LqlError> {
        let (url, client, sid) = self.require_remote()?;

        let body = serde_json::json!({
            "entity": entity,
            "relation": relation,
            "target": target,
            "layer": layer,
            "confidence": confidence.unwrap_or(0.9),
        });

        let resp = client
            .post(format!("{url}/v1/insert"))
            .header("x-session-id", sid)
            .json(&body)
            .send()
            .map_err(|e| LqlError::Execution(format!("request failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().unwrap_or_default();
            return Err(LqlError::Execution(format!("INSERT failed ({}): {}", status, text)));
        }

        let result: serde_json::Value = resp.json()
            .map_err(|e| LqlError::Execution(format!("invalid response: {e}")))?;

        let inserted = result["inserted"].as_u64().unwrap_or(0);
        let mode = result["mode"].as_str().unwrap_or("unknown");
        let ms = result["latency_ms"].as_f64().unwrap_or(0.0);

        let mut out = Vec::new();
        out.push(format!(
            "Inserted: {} —[{}]→ {} ({} layers, mode: {})",
            entity, relation, target, inserted, mode,
        ));
        out.push(format!("{:.0}ms (remote)", ms));

        Ok(out)
    }

    pub(crate) fn remote_delete(
        &self,
        conditions: &[crate::ast::Condition],
    ) -> Result<Vec<String>, LqlError> {
        let (url, client, _sid) = self.require_remote()?;

        // Build delete operations from conditions.
        let mut ops = Vec::new();
        let layer = conditions
            .iter()
            .find(|c| c.field == "layer")
            .and_then(|c| match &c.value {
                crate::ast::Value::Integer(n) => Some(*n as usize),
                _ => None,
            })
            .unwrap_or(0);
        let feature = conditions
            .iter()
            .find(|c| c.field == "feature")
            .and_then(|c| match &c.value {
                crate::ast::Value::Integer(n) => Some(*n as usize),
                _ => None,
            })
            .unwrap_or(0);

        ops.push(larql_vindex::PatchOp::Delete {
            layer,
            feature,
            reason: Some("remote DELETE".into()),
        });

        let patch = larql_vindex::VindexPatch {
            version: 1,
            base_model: String::new(),
            base_checksum: None,
            created_at: String::new(),
            description: Some(format!("DELETE L{layer} F{feature}")),
            author: None,
            tags: vec![],
            operations: ops,
        };

        let resp = client
            .post(format!("{url}/v1/patches/apply"))
            .json(&serde_json::json!({"patch": patch}))
            .send()
            .map_err(|e| LqlError::Execution(format!("request failed: {e}")))?;

        if !resp.status().is_success() {
            let text = resp.text().unwrap_or_default();
            return Err(LqlError::Execution(format!("DELETE failed: {text}")));
        }

        Ok(vec![format!("Deleted: L{layer} F{feature} → remote server")])
    }

    pub(crate) fn remote_update(
        &self,
        set: &[crate::ast::Assignment],
        conditions: &[crate::ast::Condition],
    ) -> Result<Vec<String>, LqlError> {
        let (url, client, _sid) = self.require_remote()?;

        let layer = conditions
            .iter()
            .find(|c| c.field == "layer")
            .and_then(|c| match &c.value {
                crate::ast::Value::Integer(n) => Some(*n as usize),
                _ => None,
            })
            .unwrap_or(0);
        let feature = conditions
            .iter()
            .find(|c| c.field == "feature")
            .and_then(|c| match &c.value {
                crate::ast::Value::Integer(n) => Some(*n as usize),
                _ => None,
            })
            .unwrap_or(0);

        // Build down_meta from SET assignments.
        let target = set
            .iter()
            .find(|a| a.field == "target" || a.field == "top_token")
            .and_then(|a| match &a.value {
                crate::ast::Value::String(s) => Some(s.clone()),
                _ => None,
            });
        let confidence = set
            .iter()
            .find(|a| a.field == "confidence" || a.field == "c_score")
            .and_then(|a| match &a.value {
                crate::ast::Value::Number(n) => Some(*n as f32),
                crate::ast::Value::Integer(n) => Some(*n as f32),
                _ => None,
            });

        let down_meta = target.as_ref().map(|t| {
            larql_vindex::patch::core::PatchDownMeta {
                top_token: t.clone(),
                top_token_id: 0,
                c_score: confidence.unwrap_or(0.9),
            }
        });

        let op = larql_vindex::PatchOp::Update {
            layer,
            feature,
            gate_vector_b64: None,
            down_meta,
        };

        let patch = larql_vindex::VindexPatch {
            version: 1,
            base_model: String::new(),
            base_checksum: None,
            created_at: String::new(),
            description: Some(format!("UPDATE L{layer} F{feature}")),
            author: None,
            tags: vec![],
            operations: vec![op],
        };

        let resp = client
            .post(format!("{url}/v1/patches/apply"))
            .json(&serde_json::json!({"patch": patch}))
            .send()
            .map_err(|e| LqlError::Execution(format!("request failed: {e}")))?;

        if !resp.status().is_success() {
            let text = resp.text().unwrap_or_default();
            return Err(LqlError::Execution(format!("UPDATE failed: {text}")));
        }

        let desc = target
            .as_deref()
            .map(|t| format!(" target={t}"))
            .unwrap_or_default();
        Ok(vec![format!("Updated: L{layer} F{feature}{desc} → remote server")])
    }

    // ── Remote SELECT ──

    pub(crate) fn remote_select(
        &self,
        conditions: &[crate::ast::Condition],
        limit: Option<u32>,
    ) -> Result<Vec<String>, LqlError> {
        let (url, client, _sid) = self.require_remote()?;

        let mut body = serde_json::Map::new();
        body.insert("limit".into(), serde_json::json!(limit.unwrap_or(20)));

        for cond in conditions {
            match cond.field.as_str() {
                "entity" => {
                    if let crate::ast::Value::String(s) = &cond.value {
                        body.insert("entity".into(), serde_json::json!(s));
                    }
                }
                "relation" => {
                    if let crate::ast::Value::String(s) = &cond.value {
                        body.insert("relation".into(), serde_json::json!(s));
                    }
                }
                "layer" => {
                    if let crate::ast::Value::Integer(n) = &cond.value {
                        body.insert("layer".into(), serde_json::json!(n));
                    }
                }
                "confidence" | "c_score" => {
                    match &cond.value {
                        crate::ast::Value::Number(n) => {
                            body.insert("min_confidence".into(), serde_json::json!(n));
                        }
                        crate::ast::Value::Integer(n) => {
                            body.insert("min_confidence".into(), serde_json::json!(n));
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        let resp = client
            .post(format!("{url}/v1/select"))
            .json(&serde_json::Value::Object(body))
            .send()
            .map_err(|e| LqlError::Execution(format!("request failed: {e}")))?;

        let result: serde_json::Value = resp
            .json()
            .map_err(|e| LqlError::Execution(format!("invalid response: {e}")))?;

        let mut out = Vec::new();

        if let Some(edges) = result["edges"].as_array() {
            if edges.is_empty() {
                out.push("  (no matching edges)".into());
            } else {
                out.push(format!(
                    "  {:<20} {:<15} {:>6}  {:<6} {}",
                    "Target", "Relation", "Score", "Layer", "Feature"
                ));
                out.push(format!("  {}", "-".repeat(65)));
                for edge in edges {
                    let layer = edge["layer"].as_u64().unwrap_or(0);
                    let feature = edge["feature"].as_u64().unwrap_or(0);
                    let target = edge["target"].as_str().unwrap_or("?");
                    let score = edge["c_score"].as_f64().unwrap_or(0.0);
                    let relation = edge["relation"].as_str().unwrap_or("");
                    out.push(format!(
                        "  {:<20} {:<15} {:>6.3}  L{:<5} F{}",
                        target, relation, score, layer, feature
                    ));
                }
            }
        }

        if let Some(total) = result["total"].as_u64() {
            out.push(format!("\n{} total", total));
        }

        Ok(out)
    }

    // ── Local patch management (client-side overlay) ──

    pub(crate) fn remote_apply_local_patch(&mut self, path: &str) -> Result<Vec<String>, LqlError> {
        let patch_path = std::path::PathBuf::from(path);
        if !patch_path.exists() {
            return Err(LqlError::Execution(format!("patch not found: {path}")));
        }

        let patch = larql_vindex::VindexPatch::load(&patch_path)
            .map_err(|e| LqlError::Execution(format!("failed to load patch: {e}")))?;

        let (ins, upd, del) = patch.counts();
        let total = patch.len();

        match &mut self.backend {
            Backend::Remote { local_patches, .. } => {
                local_patches.push(patch);
                Ok(vec![format!(
                    "Applied locally: {path} ({total} ops: {ins} ins, {upd} upd, {del} del)\n\
                     Patch stays client-side — server never sees it."
                )])
            }
            _ => Err(LqlError::Execution("not connected to a remote server".into())),
        }
    }

    pub(crate) fn remote_show_patches(&self) -> Result<Vec<String>, LqlError> {
        let local_patches = match &self.backend {
            Backend::Remote { local_patches, .. } => local_patches,
            _ => return Err(LqlError::Execution("not connected to a remote server".into())),
        };

        let mut out = Vec::new();
        if local_patches.is_empty() {
            out.push("  (no local patches)".into());
        } else {
            out.push("Local patches (client-side only):".into());
            for (i, patch) in local_patches.iter().enumerate() {
                let (ins, upd, del) = patch.counts();
                let name = patch.description.as_deref().unwrap_or("(unnamed)");
                out.push(format!(
                    "  {}. {:<40} {} ops ({} ins, {} upd, {} del)",
                    i + 1, name, patch.len(), ins, upd, del,
                ));
            }
        }
        Ok(out)
    }

    pub(crate) fn remote_remove_local_patch(&mut self, name: &str) -> Result<Vec<String>, LqlError> {
        let local_patches = match &mut self.backend {
            Backend::Remote { local_patches, .. } => local_patches,
            _ => return Err(LqlError::Execution("not connected to a remote server".into())),
        };

        let pos = local_patches
            .iter()
            .position(|p| p.description.as_deref().unwrap_or("unnamed") == name);

        match pos {
            Some(i) => {
                local_patches.remove(i);
                Ok(vec![format!("Removed local patch: {name}")])
            }
            None => Err(LqlError::Execution(format!("local patch not found: {name}"))),
        }
    }
}
