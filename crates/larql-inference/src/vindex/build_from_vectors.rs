//! Build a .vindex from pre-extracted NDJSON vector files.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use crate::error::InferenceError;

use super::build::IndexBuildCallbacks;
use larql_vindex::config::{
    DownMetaRecord, DownMetaTopK, VindexConfig, VindexLayerInfo,
};

    /// Build a .vindex from already-extracted NDJSON vector files.
    ///
    /// Reads ffn_gate.vectors.jsonl, ffn_down.vectors.jsonl, and
    /// embeddings.vectors.jsonl, packs them into the binary .vindex format.
    /// Much faster than build_vindex since no vocab projection needed.
    pub fn build_vindex_from_vectors(
        vectors_dir: &Path,
        output_dir: &Path,
        callbacks: &mut dyn IndexBuildCallbacks,
    ) -> Result<(), InferenceError> {
        std::fs::create_dir_all(output_dir)?;

        let gate_path = vectors_dir.join("ffn_gate.vectors.jsonl");
        let down_path = vectors_dir.join("ffn_down.vectors.jsonl");
        let embed_path = vectors_dir.join("embeddings.vectors.jsonl");

        if !gate_path.exists() {
            return Err(InferenceError::Parse(
                format!("ffn_gate.vectors.jsonl not found in {}", vectors_dir.display()),
            ));
        }

        // ── 1. Read gate header for config ──
        let gate_file = std::fs::File::open(&gate_path)?;
        let reader = BufReader::with_capacity(1 << 20, gate_file);
        let first_line = reader.lines().next()
            .ok_or_else(|| InferenceError::Parse("empty gate file".into()))??;
        let header: serde_json::Value = serde_json::from_str(&first_line)
            .map_err(|e| InferenceError::Parse(e.to_string()))?;

        let model_name = header.get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let hidden_size = header.get("dimension")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        // ── 2. Stream gate vectors → binary + collect layer info ──
        callbacks.on_stage("gate_vectors");
        let start = std::time::Instant::now();

        let gate_file = std::fs::File::open(&gate_path)?;
        let reader = BufReader::with_capacity(1 << 20, gate_file);

        // First pass: collect all records to determine layout
        let mut gate_records: Vec<(usize, usize, Vec<f32>)> = Vec::new();
        let mut max_layer: usize = 0;
        let mut count: usize = 0;

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() { continue; }

            let obj: serde_json::Value = serde_json::from_str(line)
                .map_err(|e| InferenceError::Parse(e.to_string()))?;
            if obj.get("_header").is_some() { continue; }

            let layer = obj["layer"].as_u64().unwrap() as usize;
            let feature = obj["feature"].as_u64().unwrap() as usize;
            let vector: Vec<f32> = obj["vector"].as_array().unwrap()
                .iter().map(|v| v.as_f64().unwrap() as f32).collect();

            if layer > max_layer { max_layer = layer; }
            gate_records.push((layer, feature, vector));

            count += 1;
            if count.is_multiple_of(10000) {
                callbacks.on_feature_progress("gate", 0, count, 0);
            }
        }

        let num_layers = max_layer + 1;

        // Find features per layer
        let mut layer_feature_counts: HashMap<usize, usize> = HashMap::new();
        for &(layer, feature, _) in &gate_records {
            let e = layer_feature_counts.entry(layer).or_insert(0);
            if feature + 1 > *e { *e = feature + 1; }
        }

        // Sort records by (layer, feature) for contiguous binary write
        gate_records.sort_unstable_by_key(|r| (r.0, r.1));

        // Write binary
        let bin_path = output_dir.join("gate_vectors.bin");
        let mut bin_file = BufWriter::new(std::fs::File::create(&bin_path)?);
        let mut layer_infos: Vec<VindexLayerInfo> = Vec::new();
        let mut offset: u64 = 0;

        let mut sorted_layers: Vec<usize> = layer_feature_counts.keys().copied().collect();
        sorted_layers.sort();

        for &layer in &sorted_layers {
            let num_features = layer_feature_counts[&layer];
            // Write zeros for all features, then overwrite with actual data
            let mut layer_data = vec![0.0f32; num_features * hidden_size];

            for &(l, feat, ref vec) in &gate_records {
                if l == layer {
                    let dst = feat * hidden_size;
                    layer_data[dst..dst + hidden_size].copy_from_slice(vec);
                }
            }

            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    layer_data.as_ptr() as *const u8,
                    layer_data.len() * 4,
                )
            };
            bin_file.write_all(bytes)?;

            let length = bytes.len() as u64;
            layer_infos.push(VindexLayerInfo { layer, num_features, offset, length });
            offset += length;
        }
        bin_file.flush()?;

        callbacks.on_stage_done("gate_vectors", start.elapsed().as_secs_f64() * 1000.0);

        // ── 3. Stream embeddings → binary ──
        callbacks.on_stage("embeddings");
        let start = std::time::Instant::now();

        let embed_bin_path = output_dir.join("embeddings.bin");
        let mut embed_out = BufWriter::new(std::fs::File::create(&embed_bin_path)?);

        let embed_file = std::fs::File::open(&embed_path)?;
        let reader = BufReader::with_capacity(1 << 20, embed_file);

        let mut vocab_size: usize = 0;
        let mut embed_count: usize = 0;

        // Collect all embeddings (they may not be in order)
        let mut embed_records: Vec<(usize, Vec<f32>)> = Vec::new();

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() { continue; }

            let obj: serde_json::Value = serde_json::from_str(line)
                .map_err(|e| InferenceError::Parse(e.to_string()))?;
            if obj.get("_header").is_some() { continue; }

            let feature = obj["feature"].as_u64().unwrap() as usize;
            let vector: Vec<f32> = obj["vector"].as_array().unwrap()
                .iter().map(|v| v.as_f64().unwrap() as f32).collect();

            if feature + 1 > vocab_size { vocab_size = feature + 1; }
            embed_records.push((feature, vector));

            embed_count += 1;
            if embed_count.is_multiple_of(10000) {
                callbacks.on_feature_progress("embeddings", 0, embed_count, 0);
            }
        }

        // Sort by feature ID and write contiguously
        embed_records.sort_unstable_by_key(|r| r.0);
        let mut embed_data = vec![0.0f32; vocab_size * hidden_size];
        for (feat, vec) in &embed_records {
            let dst = feat * hidden_size;
            embed_data[dst..dst + hidden_size].copy_from_slice(vec);
        }

        let embed_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                embed_data.as_ptr() as *const u8,
                embed_data.len() * 4,
            )
        };
        embed_out.write_all(embed_bytes)?;
        embed_out.flush()?;

        callbacks.on_stage_done("embeddings", start.elapsed().as_secs_f64() * 1000.0);

        // ── 4. Stream down metadata (copy top_k, skip vectors) ──
        callbacks.on_stage("down_meta");
        let start = std::time::Instant::now();

        let down_meta_path = output_dir.join("down_meta.jsonl");
        let mut down_out = BufWriter::new(std::fs::File::create(&down_meta_path)?);

        let down_file = std::fs::File::open(&down_path)?;
        let reader = BufReader::with_capacity(1 << 20, down_file);
        let mut down_count: usize = 0;
        let mut down_top_k_size: usize = 0;

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() { continue; }

            let obj: serde_json::Value = serde_json::from_str(line)
                .map_err(|e| InferenceError::Parse(e.to_string()))?;
            if obj.get("_header").is_some() { continue; }

            let layer = obj["layer"].as_u64().unwrap() as usize;
            let feature = obj["feature"].as_u64().unwrap() as usize;
            let top_token = obj["top_token"].as_str().unwrap_or("").to_string();
            let top_token_id = obj["top_token_id"].as_u64().unwrap_or(0) as u32;
            let c_score = obj["c_score"].as_f64().unwrap_or(0.0) as f32;

            let top_k: Vec<DownMetaTopK> = match obj.get("top_k").and_then(|v| v.as_array()) {
                Some(arr) => {
                    if down_top_k_size == 0 { down_top_k_size = arr.len(); }
                    arr.iter().filter_map(|entry| {
                        Some(DownMetaTopK {
                            token: entry.get("token")?.as_str()?.to_string(),
                            token_id: entry.get("token_id")?.as_u64()? as u32,
                            logit: entry.get("logit")?.as_f64()? as f32,
                        })
                    }).collect()
                }
                None => vec![],
            };

            let record = DownMetaRecord {
                layer, feature, top_token, top_token_id, c_score, top_k,
            };

            serde_json::to_writer(&mut down_out, &record)
                .map_err(|e| InferenceError::Parse(e.to_string()))?;
            down_out.write_all(b"\n")?;

            down_count += 1;
            if down_count.is_multiple_of(10000) {
                callbacks.on_feature_progress("down", 0, down_count, 0);
            }
        }
        down_out.flush()?;

        callbacks.on_stage_done("down_meta", start.elapsed().as_secs_f64() * 1000.0);

        // ── 5. Copy tokenizer if available ──
        // Look for tokenizer.json near the vectors dir or in common locations
        let tokenizer_src = find_tokenizer(vectors_dir);
        if let Some(ref src) = tokenizer_src {
            callbacks.on_stage("tokenizer");
            std::fs::copy(src, output_dir.join("tokenizer.json"))?;
            callbacks.on_stage_done("tokenizer", 0.0);
        }

        // ── 6. Determine embed_scale from model family ──
        // Gemma models use sqrt(hidden_size), others use 1.0
        let intermediate_size = layer_feature_counts.values().max().copied().unwrap_or(0);
        let embed_scale = if model_name.contains("gemma") {
            (hidden_size as f32).sqrt()
        } else {
            1.0
        };
        let family = if model_name.contains("gemma") {
            "gemma3"
        } else if model_name.contains("llama") || model_name.contains("Llama") {
            "llama"
        } else {
            "unknown"
        };

        // ── 7. Write index.json ──
        let config = VindexConfig {
            version: 1,
            model: model_name,
            family: family.to_string(),
            num_layers,
            hidden_size,
            intermediate_size,
            vocab_size,
            embed_scale,
            layers: layer_infos,
            down_top_k: down_top_k_size,
            has_model_weights: false,
            model_config: None,
        };

        let config_json = serde_json::to_string_pretty(&config)
            .map_err(|e| InferenceError::Parse(e.to_string()))?;
        std::fs::write(output_dir.join("index.json"), config_json)?;

        Ok(())
    }

/// Try to find tokenizer.json near the vectors directory.
fn find_tokenizer(vectors_dir: &Path) -> Option<std::path::PathBuf> {
    // Check parent directory
    if let Some(parent) = vectors_dir.parent() {
        let p = parent.join("tokenizer.json");
        if p.exists() { return Some(p); }
    }
    // Check vectors dir itself
    let p = vectors_dir.join("tokenizer.json");
    if p.exists() { return Some(p); }
    // Check sibling
    if let Some(parent) = vectors_dir.parent() {
        let p = parent.join("vectors").join("tokenizer.json");
        if p.exists() { return Some(p); }
    }
    None
}
