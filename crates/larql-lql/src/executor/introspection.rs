/// Introspection executor: SHOW RELATIONS, SHOW LAYERS, SHOW FEATURES, SHOW MODELS.

use std::collections::HashMap;

use crate::ast::*;
use crate::error::LqlError;
use super::Session;
use super::helpers::{format_number, format_bytes, dir_size, is_content_token};

impl Session {
    pub(crate) fn exec_show_relations(
        &self,
        layer_filter: Option<u32>,
        _with_examples: bool,
    ) -> Result<Vec<String>, LqlError> {
        let (_path, _config, index) = self.require_vindex()?;

        let all_layers = index.loaded_layers();
        let scan_layers: Vec<usize> = if let Some(l) = layer_filter {
            vec![l as usize]
        } else {
            all_layers.iter().copied().filter(|l| *l >= 14 && *l <= 27).collect()
        };

        struct TokenInfo {
            count: usize,
            max_score: f32,
            min_layer: usize,
            max_layer: usize,
            original: String,
        }

        let mut tokens: HashMap<String, TokenInfo> = HashMap::new();

        for &layer in &scan_layers {
            if let Some(metas) = index.down_meta_at(layer) {
                for meta_opt in metas.iter() {
                    if let Some(meta) = meta_opt {
                        let tok = meta.top_token.trim();
                        if !is_content_token(tok) {
                            continue;
                        }
                        if meta.c_score < 0.2 {
                            continue;
                        }
                        let key = tok.to_lowercase();
                        let entry = tokens.entry(key).or_insert(TokenInfo {
                            count: 0,
                            max_score: 0.0,
                            min_layer: layer,
                            max_layer: layer,
                            original: tok.to_string(),
                        });
                        entry.count += 1;
                        if meta.c_score > entry.max_score {
                            entry.max_score = meta.c_score;
                        }
                        if layer < entry.min_layer {
                            entry.min_layer = layer;
                        }
                        if layer > entry.max_layer {
                            entry.max_layer = layer;
                        }
                    }
                }
            }
        }

        let mut sorted: Vec<(&str, &TokenInfo)> = tokens.iter()
            .map(|(_, info)| (info.original.as_str(), info))
            .collect();
        sorted.sort_by(|a, b| b.1.count.cmp(&a.1.count));
        sorted.truncate(30);

        let mut out = Vec::new();
        let layer_label = if let Some(l) = layer_filter {
            format!("L{}", l)
        } else {
            "L14-27".into()
        };
        out.push(format!(
            "{:<25} {:>8} {:>8} {:>10}",
            format!("Token ({})", layer_label), "Count", "Score", "Layers"
        ));
        out.push("-".repeat(55));

        for (tok, info) in &sorted {
            out.push(format!(
                "{:<25} {:>8} {:>8.2} {:>5}-{}",
                tok,
                info.count,
                info.max_score,
                info.min_layer,
                info.max_layer,
            ));
        }

        if sorted.is_empty() {
            out.push("  (no content tokens found)".into());
        }

        Ok(out)
    }

    pub(crate) fn exec_show_layers(&self, range: Option<&Range>) -> Result<Vec<String>, LqlError> {
        let (_path, _config, index) = self.require_vindex()?;

        let all_layers = index.loaded_layers();
        let show_layers: Vec<usize> = if let Some(r) = range {
            (r.start as usize..=r.end as usize)
                .filter(|l| all_layers.contains(l))
                .collect()
        } else {
            all_layers
        };

        let mut out = Vec::new();
        out.push(format!(
            "{:<8} {:>10} {:>10} {:>15}",
            "Layer", "Features", "With Meta", "Top Token"
        ));
        out.push("-".repeat(48));

        for layer in &show_layers {
            let gate_count = index
                .gate_vectors_at(*layer)
                .map(|m| m.shape()[0])
                .unwrap_or(0);
            let (meta_count, top_tok) = if let Some(metas) = index.down_meta_at(*layer) {
                let count = metas.iter().filter(|m| m.is_some()).count();
                let mut freq: HashMap<&str, usize> = HashMap::new();
                for m in metas.iter().flatten() {
                    *freq.entry(&m.top_token).or_default() += 1;
                }
                let top = freq
                    .into_iter()
                    .max_by_key(|(_, c)| *c)
                    .map(|(t, _)| t.to_string())
                    .unwrap_or_default();
                (count, top)
            } else {
                (0, String::new())
            };

            out.push(format!(
                "L{:<7} {:>10} {:>10} {:>15}",
                layer,
                format_number(gate_count),
                format_number(meta_count),
                top_tok
            ));
        }

        Ok(out)
    }

    pub(crate) fn exec_show_features(
        &self,
        layer: u32,
        _conditions: &[Condition],
        limit: Option<u32>,
    ) -> Result<Vec<String>, LqlError> {
        let (_path, _config, index) = self.require_vindex()?;
        let limit = limit.unwrap_or(20) as usize;

        let metas = index
            .down_meta_at(layer as usize)
            .ok_or_else(|| LqlError::Execution(format!("no metadata for layer {layer}")))?;

        let mut out = Vec::new();
        out.push(format!(
            "{:<8} {:<20} {:>10} {:>30}",
            "Feature", "Top Token", "Score", "Down outputs"
        ));
        out.push("-".repeat(72));

        let mut count = 0;
        for (feat_idx, meta_opt) in metas.iter().enumerate() {
            if count >= limit {
                break;
            }
            if let Some(meta) = meta_opt {
                let down_tokens: String = meta
                    .top_k
                    .iter()
                    .take(5)
                    .map(|t| format!("{}", t.token))
                    .collect::<Vec<_>>()
                    .join(", ");

                out.push(format!(
                    "F{:<7} {:<20} {:>10.4} {:>30}",
                    feat_idx, meta.top_token, meta.c_score, down_tokens
                ));
                count += 1;
            }
        }

        Ok(out)
    }

    pub(crate) fn exec_show_models(&self) -> Result<Vec<String>, LqlError> {
        let mut out = Vec::new();
        out.push(format!(
            "{:<35} {:>10} {:>8} {:>12}",
            "Model", "Size", "Layers", "Status"
        ));
        out.push("-".repeat(70));

        let cwd = std::env::current_dir().unwrap_or_default();
        if let Ok(entries) = std::fs::read_dir(&cwd) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    let index_json = path.join("index.json");
                    if index_json.exists() {
                        if let Ok(config) = larql_vindex::load_vindex_config(&path) {
                            let size = dir_size(&path);
                            out.push(format!(
                                "{:<35} {:>10} {:>8} {:>12}",
                                path.file_name()
                                    .unwrap_or_default()
                                    .to_string_lossy(),
                                format_bytes(size),
                                config.num_layers,
                                "ready",
                            ));
                        }
                    }
                }
            }
        }

        if out.len() == 2 {
            out.push("  (no vindexes found in current directory)".into());
        }

        Ok(out)
    }
}
