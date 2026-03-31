//! Build a .vindex from model weights — the extraction/clustering pipeline.

use std::io::{BufWriter, Write};
use std::path::Path;

use ndarray::Array2;

use crate::error::InferenceError;
use crate::model::ModelWeights;

use larql_models::TopKEntry;

/// Collected data for relation clustering.
struct ClusterData {
    directions: Vec<f32>,
    features: Vec<(usize, usize)>,
    top_tokens: Vec<String>,
    #[allow(dead_code)]
    input_tokens: Vec<String>,
    output_tokens: Vec<String>,
}

/// Build the whole-word vocabulary: tokens that decode as 3+ char alphabetic words.
/// Returns (token_ids, reduced_embedding_matrix).
fn build_whole_word_vocab(
    tokenizer: &tokenizers::Tokenizer,
    embed: &Array2<f32>,
    vocab_size: usize,
    hidden_size: usize,
) -> (Vec<usize>, Array2<f32>) {
    let mut ww_ids: Vec<usize> = Vec::new();
    for id in 0..vocab_size {
        if let Ok(tok) = tokenizer.decode(&[id as u32], true) {
            let tok = tok.trim();
            if tok.len() >= 3
                && tok.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '\'')
            {
                ww_ids.push(id);
            }
        }
    }

    let ww_count = ww_ids.len();
    let mut ww_embed = Array2::<f32>::zeros((ww_count, hidden_size));
    for (i, &id) in ww_ids.iter().enumerate() {
        ww_embed.row_mut(i).assign(&embed.row(id));
    }

    eprintln!("    Whole-word vocab: {} tokens (of {})", ww_count, vocab_size);
    (ww_ids, ww_embed)
}

/// Compute gate top tokens for features at a layer using whole-word embeddings.
/// Returns a Vec<String> of decoded whole-word tokens, one per feature.
fn compute_gate_top_tokens(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    layer: usize,
    num_features: usize,
    ww_ids: &[usize],
    ww_embed: &Array2<f32>,
) -> Vec<String> {
    let gate_key = weights.arch.ffn_gate_key(layer);
    let w_gate = match weights.tensors.get(&gate_key) {
        Some(w) => w,
        None => return vec![String::new(); num_features],
    };

    let mut tokens = vec![String::new(); num_features];
    let gbatch = 1024;
    for gstart in (0..num_features).step_by(gbatch) {
        let gend = (gstart + gbatch).min(num_features);
        let chunk = w_gate.slice(ndarray::s![gstart..gend, ..]);
        let proj = ww_embed.dot(&chunk.t());
        for f in 0..(gend - gstart) {
            let col = proj.column(f);
            let mut best_idx = 0;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &val) in col.iter().enumerate() {
                if val > best_val {
                    best_val = val;
                    best_idx = i;
                }
            }
            let tok_id = ww_ids[best_idx];
            tokens[gstart + f] = tokenizer
                .decode(&[tok_id as u32], true)
                .unwrap_or_default()
                .trim()
                .to_string();
        }
    }
    tokens
}

/// Compute the offset direction for a gate→down feature pair.
/// Returns normalized(output_embed - input_embed) or None if invalid.
fn compute_offset_direction(
    gate_token: &str,
    output_token_id: usize,
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    hidden_size: usize,
    vocab_size: usize,
) -> Option<Vec<f32>> {
    if gate_token.is_empty() || output_token_id <= 2 || output_token_id >= vocab_size {
        return None;
    }

    // Get gate token embedding (may be multi-subword)
    let enc = tokenizer.encode(gate_token, false).ok()?;
    let ids = enc.get_ids();
    let valid: Vec<usize> = ids
        .iter()
        .filter(|&&id| id > 2)
        .map(|&id| id as usize)
        .filter(|&id| id < vocab_size)
        .collect();
    if valid.is_empty() {
        return None;
    }

    let mut input_avg = vec![0.0f32; hidden_size];
    for &id in &valid {
        for (j, &v) in weights.embed.row(id).iter().enumerate() {
            input_avg[j] += v;
        }
    }
    let n = valid.len() as f32;
    for v in &mut input_avg {
        *v /= n;
    }

    let output_embed = weights.embed.row(output_token_id);
    let offset: Vec<f32> = output_embed
        .iter()
        .zip(input_avg.iter())
        .map(|(o, i)| o - i)
        .collect();
    let norm: f32 = offset.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 1e-8 {
        Some(offset.iter().map(|v| v / norm).collect())
    } else {
        None
    }
}

/// Run the clustering and labeling pipeline on collected cluster data.
/// Writes relation_clusters.json and feature_clusters.jsonl.
fn run_clustering_pipeline(
    data: ClusterData,
    hidden_size: usize,
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    output_dir: &Path,
    callbacks: &mut dyn IndexBuildCallbacks,
) -> Result<(), InferenceError> {
    if data.directions.is_empty() {
        return Ok(());
    }

    callbacks.on_stage("relation_clusters");

    let n_features = data.features.len();
    let matrix = ndarray::Array2::from_shape_vec((n_features, hidden_size), data.directions)
        .map_err(|e| InferenceError::Parse(format!("cluster data shape: {e}")))?;

    let optimal_k = 512.min(n_features);

    let (centres, assignments, _distances) = crate::clustering::kmeans(&matrix, optimal_k, 50);

    // Load reference databases
    let ref_dbs = crate::clustering::load_reference_databases();

    // Tier 1: output-only matching — Wikidata ONLY for L14-27 features.
    // WordNet is for L0-13 (linguistic). Wikidata is for L14-27 (factual).
    // They don't compete — each database matches its own layer range.
    let wikidata_refs: Vec<&crate::clustering::pair_matching::RelationDatabase> =
        ref_dbs.wikidata.iter().collect();
    let output_labels = if !wikidata_refs.is_empty() {
        crate::clustering::pair_matching::label_clusters_from_outputs(
            &assignments,
            &data.output_tokens,
            optimal_k,
            &wikidata_refs,
        )
    } else {
        vec![None; optimal_k]
    };

    let output_labeled = output_labels.iter().filter(|l| l.is_some()).count();
    eprintln!("  Wikidata output matching: {}/{} clusters labeled", output_labeled, optimal_k);

    // Tier 2+3: embedding projection + pattern detection
    let (embed_labels, top_tokens_per_cluster) =
        crate::clustering::auto_label_clusters_from_embeddings(
            &centres,
            &weights.embed,
            tokenizer,
            &assignments,
            &data.top_tokens,
            optimal_k,
        );

    // Merge: Wikidata output labels > embedding/pattern labels
    let labels: Vec<String> = (0..optimal_k)
        .map(|c| {
            output_labels[c]
                .clone()
                .unwrap_or_else(|| embed_labels[c].clone())
        })
        .collect();

    let mut counts = vec![0usize; optimal_k];
    for &a in &assignments {
        if a < optimal_k {
            counts[a] += 1;
        }
    }

    // Write relation_clusters.json
    let cluster_result = crate::clustering::ClusterResult {
        k: optimal_k,
        centres: centres.rows().into_iter().map(|r| r.to_vec()).collect(),
        labels,
        counts,
        top_tokens: top_tokens_per_cluster,
    };

    let clusters_json = serde_json::to_string_pretty(&cluster_result)
        .map_err(|e| InferenceError::Parse(e.to_string()))?;
    std::fs::write(output_dir.join("relation_clusters.json"), clusters_json)?;

    // Write per-feature cluster assignments
    let assign_path = output_dir.join("feature_clusters.jsonl");
    let mut assign_file = BufWriter::new(std::fs::File::create(&assign_path)?);
    for (i, &(layer, feat)) in data.features.iter().enumerate() {
        let record = serde_json::json!({ "l": layer, "f": feat, "c": assignments[i] });
        serde_json::to_writer(&mut assign_file, &record)
            .map_err(|e| InferenceError::Parse(e.to_string()))?;
        assign_file.write_all(b"\n")?;
    }
    assign_file.flush()?;

    callbacks.on_stage_done(
        &format!("relation_clusters (k={}, {} features)", optimal_k, n_features),
        0.0,
    );

    Ok(())
}

use larql_vindex::config::{
    DownMetaRecord, DownMetaTopK, VindexConfig, VindexLayerInfo, VindexModelConfig,
};

/// Callbacks for index build progress.
pub trait IndexBuildCallbacks {
    fn on_stage(&mut self, _stage: &str) {}
    fn on_layer_start(&mut self, _component: &str, _layer: usize, _total: usize) {}
    fn on_feature_progress(&mut self, _component: &str, _layer: usize, _done: usize, _total: usize) {}
    fn on_layer_done(&mut self, _component: &str, _layer: usize, _elapsed_ms: f64) {}
    fn on_stage_done(&mut self, _stage: &str, _elapsed_ms: f64) {}
}

pub struct SilentBuildCallbacks;
impl IndexBuildCallbacks for SilentBuildCallbacks {}

    /// Build a .vindex from model weights and write it to disk.
    ///
    /// Reads gate vectors and down projections directly from safetensors,
    /// projects down vectors to vocabulary for top-k token metadata,
    /// writes everything to a self-contained directory.
    pub fn build_vindex(
        weights: &ModelWeights,
        tokenizer: &tokenizers::Tokenizer,
        model_name: &str,
        output_dir: &Path,
        down_top_k: usize,
        callbacks: &mut dyn IndexBuildCallbacks,
    ) -> Result<(), InferenceError> {
        std::fs::create_dir_all(output_dir)?;

        let num_layers = weights.num_layers;
        let hidden_size = weights.hidden_size;
        let intermediate_size = weights.intermediate_size;
        let vocab_size = weights.vocab_size;
        let embed_scale = weights.arch.embed_scale();

        // ── 1. Write gate vectors (binary f32) ──
        callbacks.on_stage("gate_vectors");
        let gate_path = output_dir.join("gate_vectors.bin");
        let mut gate_file = BufWriter::new(std::fs::File::create(&gate_path)?);
        let mut layer_infos: Vec<VindexLayerInfo> = Vec::new();
        let mut offset: u64 = 0;

        for layer in 0..num_layers {
            callbacks.on_layer_start("gate", layer, num_layers);
            let start = std::time::Instant::now();

            let gate_key = weights.arch.ffn_gate_key(layer);
            let w_gate = match weights.tensors.get(&gate_key) {
                Some(w) => w,
                None => continue,
            };

            let num_features = w_gate.shape()[0];
            let data = w_gate.as_slice().unwrap();
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
            };
            gate_file.write_all(bytes)?;

            let length = bytes.len() as u64;
            layer_infos.push(VindexLayerInfo {
                layer,
                num_features,
                offset,
                length,
            });
            offset += length;

            callbacks.on_layer_done("gate", layer, start.elapsed().as_secs_f64() * 1000.0);
        }
        gate_file.flush()?;
        callbacks.on_stage_done("gate_vectors", 0.0);

        // ── 2. Write embeddings (binary f32) ──
        callbacks.on_stage("embeddings");
        let embed_path = output_dir.join("embeddings.bin");
        let embed_data = weights.embed.as_slice().unwrap();
        let embed_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(embed_data.as_ptr() as *const u8, embed_data.len() * 4)
        };
        std::fs::write(&embed_path, embed_bytes)?;
        callbacks.on_stage_done("embeddings", 0.0);

        // ── 3. Write down metadata + collect directions for relation clustering ──
        callbacks.on_stage("down_meta");
        let down_path = output_dir.join("down_meta.jsonl");
        let mut down_file = BufWriter::new(std::fs::File::create(&down_path)?);

        // Collect offset directions for knowledge layers (L14-28) for relation clustering
        let cluster_layer_min = 14.min(num_layers);
        let cluster_layer_max = 28.min(num_layers);
        let mut cluster_directions: Vec<f32> = Vec::new();
        let mut cluster_features: Vec<(usize, usize)> = Vec::new();
        let mut cluster_top_tokens: Vec<String> = Vec::new();
        let mut cluster_input_tokens: Vec<String> = Vec::new();
        let mut cluster_output_tokens: Vec<String> = Vec::new();
        // Build whole-word vocab once, shared across layers
        let (ww_ids_shared, ww_embed_shared) =
            build_whole_word_vocab(tokenizer, &weights.embed, vocab_size, hidden_size);

        for layer in 0..num_layers {
            callbacks.on_layer_start("down", layer, num_layers);
            let start = std::time::Instant::now();

            let down_key = weights.arch.ffn_down_key(layer);
            let w_down = match weights.tensors.get(&down_key) {
                Some(w) => w,
                None => continue,
            };

            // w_down is (hidden_size, intermediate_size)
            let num_features = w_down.shape()[1];
            let is_knowledge_layer = layer >= cluster_layer_min && layer < cluster_layer_max;

            // For knowledge layers: find each feature's gate input token via
            // For knowledge layers: find what entity each gate responds to.
            let gate_top_tokens: Vec<String> = if is_knowledge_layer {
                compute_gate_top_tokens(
                    weights, tokenizer, layer, num_features,
                    &ww_ids_shared, &ww_embed_shared,
                )
            } else {
                vec![]
            };

            // Batch features: embed @ w_down_chunk → (vocab, chunk_size)
            let batch_size = 1024;

            for batch_start in (0..num_features).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(num_features);
                callbacks.on_feature_progress("down", layer, batch_start, num_features);

                // Extract columns [batch_start..batch_end] from w_down
                let w_chunk = w_down.slice(ndarray::s![.., batch_start..batch_end]).to_owned();
                // BLAS: (vocab, hidden) @ (hidden, chunk) → (vocab, chunk)
                let chunk_logits = weights.embed.dot(&w_chunk);

            for feat in batch_start..batch_end {
                let col = chunk_logits.column(feat - batch_start);
                let mut scores: Vec<(usize, f32)> = col.iter().copied().enumerate().collect();

                let k = down_top_k.min(scores.len());
                if k > 0 && k < scores.len() {
                    scores.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
                }
                scores.truncate(k);
                scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                let top_k_entries: Vec<TopKEntry> = scores
                    .into_iter()
                    .filter_map(|(idx, logit)| {
                        tokenizer
                            .decode(&[idx as u32], true)
                            .ok()
                            .map(|s| s.trim().to_string())
                            .filter(|s| !s.is_empty())
                            .map(|token| TopKEntry {
                                token,
                                token_id: idx as u32,
                                logit,
                            })
                    })
                    .collect();

                let (top_token, top_token_id, c_score) = if let Some(first) = top_k_entries.first() {
                    (first.token.clone(), first.token_id, first.logit)
                } else {
                    (String::new(), 0, 0.0)
                };

                // Collect gate→down offset direction for relation clustering.
                // The offset = normalize(target_embed - input_embed) captures
                // the RELATION between what activates the feature (entity) and
                // what it outputs (target). France→Paris and Germany→Berlin
                // share the same offset direction = "capital-of".
                if is_knowledge_layer && top_token_id > 0 && !gate_top_tokens.is_empty() {
                    let gate_tok = &gate_top_tokens[feat];
                    if let Some(offset) = compute_offset_direction(
                        gate_tok, top_token_id as usize,
                        weights, tokenizer, hidden_size, vocab_size,
                    ) {
                        cluster_directions.extend_from_slice(&offset);
                        cluster_features.push((layer, feat));
                        let all_tokens: Vec<String> = top_k_entries.iter()
                            .map(|e| e.token.clone())
                            .collect();
                        cluster_top_tokens.push(all_tokens.join("|"));
                        cluster_input_tokens.push(gate_tok.clone());
                        cluster_output_tokens.push(top_token.clone());
                    }
                }

                let record = DownMetaRecord {
                    layer,
                    feature: feat,
                    top_token,
                    top_token_id,
                    c_score,
                    top_k: top_k_entries
                        .iter()
                        .map(|e| DownMetaTopK {
                            token: e.token.clone(),
                            token_id: e.token_id,
                            logit: e.logit,
                        })
                        .collect(),
                };

                serde_json::to_writer(&mut down_file, &record)
                    .map_err(|e| InferenceError::Parse(e.to_string()))?;
                down_file.write_all(b"\n")?;
            }
            } // end batch

            callbacks.on_layer_done("down", layer, start.elapsed().as_secs_f64() * 1000.0);
        }
        down_file.flush()?;
        callbacks.on_stage_done("down_meta", 0.0);

        // ── 3b. Cluster down directions to discover relation types ──
        run_clustering_pipeline(
            ClusterData {
                directions: cluster_directions,
                features: cluster_features,
                top_tokens: cluster_top_tokens,
                input_tokens: cluster_input_tokens,
                output_tokens: cluster_output_tokens,
            },
            hidden_size,
            weights,
            tokenizer,
            output_dir,
            callbacks,
        )?;

        // ── 4. Copy tokenizer ──
        callbacks.on_stage("tokenizer");
        let tokenizer_json = tokenizer
            .to_string(true)
            .map_err(|e| InferenceError::Parse(format!("tokenizer serialize: {e}")))?;
        std::fs::write(output_dir.join("tokenizer.json"), tokenizer_json)?;
        callbacks.on_stage_done("tokenizer", 0.0);

        // ── 5. Write index.json ──
        let config = VindexConfig {
            version: 1,
            model: model_name.to_string(),
            family: weights.arch.family().to_string(),
            num_layers,
            hidden_size,
            intermediate_size,
            vocab_size,
            embed_scale,
            layers: layer_infos,
            down_top_k,
            has_model_weights: false,
            model_config: Some(VindexModelConfig {
                model_type: weights.arch.config().model_type.clone(),
                head_dim: weights.head_dim,
                num_q_heads: weights.num_q_heads,
                num_kv_heads: weights.num_kv_heads,
                rope_base: weights.rope_base,
                sliding_window: weights.arch.config().sliding_window,
            }),
        };

        let config_json = serde_json::to_string_pretty(&config)
            .map_err(|e| InferenceError::Parse(e.to_string()))?;
        std::fs::write(output_dir.join("index.json"), config_json)?;

        Ok(())
    }

    /// Resume an interrupted vindex build.
    /// Assumes gate_vectors.bin, embeddings.bin, and down_meta.jsonl exist.
    /// Runs: relation clustering + tokenizer + index.json.
    pub fn build_vindex_resume(
        weights: &ModelWeights,
        tokenizer: &tokenizers::Tokenizer,
        model_name: &str,
        output_dir: &Path,
        callbacks: &mut dyn IndexBuildCallbacks,
    ) -> Result<(), InferenceError> {
        let num_layers = weights.num_layers;
        let hidden_size = weights.hidden_size;
        let intermediate_size = weights.intermediate_size;
        let vocab_size = weights.vocab_size;
        let embed_scale = weights.arch.embed_scale();

        // Reconstruct layer_infos from gate_vectors.bin
        let gate_path = output_dir.join("gate_vectors.bin");
        let gate_size = std::fs::metadata(&gate_path)?.len();
        let bytes_per_layer = (intermediate_size * hidden_size * 4) as u64;
        let mut layer_infos = Vec::new();
        for layer in 0..num_layers {
            layer_infos.push(VindexLayerInfo {
                layer,
                num_features: intermediate_size,
                offset: layer as u64 * bytes_per_layer,
                length: bytes_per_layer,
            });
        }
        eprintln!("  Reconstructed {} layer infos from gate_vectors.bin ({:.1} GB)",
            layer_infos.len(), gate_size as f64 / 1e9);

        // Read down_meta.jsonl to collect cluster directions (L14-28)
        let cluster_layer_min = 14.min(num_layers);
        let cluster_layer_max = 28.min(num_layers);
        let mut cluster_directions: Vec<f32> = Vec::new();
        let mut cluster_features: Vec<(usize, usize)> = Vec::new();
        let mut cluster_top_tokens: Vec<String> = Vec::new();
        let mut cluster_input_tokens: Vec<String> = Vec::new();
        let mut cluster_output_tokens: Vec<String> = Vec::new();

        // Build whole-word vocab and gate top tokens
        eprintln!("  Building whole-word vocabulary...");
        let (ww_ids, ww_embed) =
            build_whole_word_vocab(tokenizer, &weights.embed, vocab_size, hidden_size);

        eprintln!("  Computing gate input tokens for L{}-{}...", cluster_layer_min, cluster_layer_max - 1);
        let mut gate_top_tokens_per_layer: std::collections::HashMap<usize, Vec<String>> =
            std::collections::HashMap::new();
        for layer in cluster_layer_min..cluster_layer_max {
            let layer_start = std::time::Instant::now();
            let tokens = compute_gate_top_tokens(
                weights, tokenizer, layer, intermediate_size,
                &ww_ids, &ww_embed,
            );
            gate_top_tokens_per_layer.insert(layer, tokens);
            eprintln!("    gate L{:2}: {:.1}s", layer, layer_start.elapsed().as_secs_f64());
        }
        eprintln!("  Gate input tokens computed for {} layers", gate_top_tokens_per_layer.len());

        eprintln!("  Reading down_meta.jsonl for offset directions...");
        let down_path = output_dir.join("down_meta.jsonl");
        let down_file = std::fs::File::open(&down_path)?;
        let reader = std::io::BufReader::new(down_file);
        let mut count = 0usize;
        for line in std::io::BufRead::lines(reader) {
            let line = line?;
            let line = line.trim();
            if line.is_empty() { continue; }
            let obj: serde_json::Value = serde_json::from_str(line)
                .map_err(|e| InferenceError::Parse(e.to_string()))?;
            if obj.get("_header").is_some() { continue; }

            let layer = obj.get("l").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let feat = obj.get("f").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let top_token_id = obj.get("i").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

            if layer >= cluster_layer_min && layer < cluster_layer_max
                && top_token_id > 2 && top_token_id < vocab_size
            {
                // Gate→down offset using whole-word gate tokens
                if let Some(gate_tokens) = gate_top_tokens_per_layer.get(&layer) {
                    if feat < gate_tokens.len() {
                        let gate_tok = &gate_tokens[feat];
                        if let Some(offset) = compute_offset_direction(
                            gate_tok, top_token_id,
                            weights, tokenizer, hidden_size, vocab_size,
                        ) {
                            cluster_directions.extend_from_slice(&offset);
                            cluster_features.push((layer, feat));
                            let all_tokens: Vec<String> = obj.get("k")
                                .and_then(|v| v.as_array())
                                .map(|arr| arr.iter()
                                    .filter_map(|e| e.get("t").and_then(|t| t.as_str()).map(|s| s.to_string()))
                                    .collect())
                                .unwrap_or_default();
                            cluster_top_tokens.push(all_tokens.join("|"));
                            let out_str = obj.get("t")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();
                            cluster_input_tokens.push(gate_tok.clone());
                            cluster_output_tokens.push(out_str);
                        }
                    }
                }
            }
            count += 1;
            if count % 50000 == 0 {
                eprint!("\r  Read {} features...", count);
            }
        }
        eprintln!("\r  Read {} features, {} in knowledge layers", count, cluster_features.len());

        // Relation clustering
        run_clustering_pipeline(
            ClusterData {
                directions: cluster_directions,
                features: cluster_features,
                top_tokens: cluster_top_tokens,
                input_tokens: cluster_input_tokens,
                output_tokens: cluster_output_tokens,
            },
            hidden_size,
            weights,
            tokenizer,
            output_dir,
            callbacks,
        )?;

        // Tokenizer
        callbacks.on_stage("tokenizer");
        let tokenizer_json = tokenizer.to_string(true)
            .map_err(|e| InferenceError::Parse(format!("tokenizer serialize: {e}")))?;
        std::fs::write(output_dir.join("tokenizer.json"), tokenizer_json)?;
        callbacks.on_stage_done("tokenizer", 0.0);

        // index.json
        let down_top_k = 10; // default
        let config = VindexConfig {
            version: 1,
            model: model_name.to_string(),
            family: weights.arch.family().to_string(),
            num_layers,
            hidden_size,
            intermediate_size,
            vocab_size,
            embed_scale,
            layers: layer_infos,
            down_top_k,
            has_model_weights: output_dir.join("model_weights.bin").exists(),
            model_config: Some(VindexModelConfig {
                model_type: weights.arch.config().model_type.clone(),
                head_dim: weights.head_dim,
                num_q_heads: weights.num_q_heads,
                num_kv_heads: weights.num_kv_heads,
                rope_base: weights.rope_base,
                sliding_window: weights.arch.config().sliding_window,
            }),
        };
        let config_json = serde_json::to_string_pretty(&config)
            .map_err(|e| InferenceError::Parse(e.to_string()))?;
        std::fs::write(output_dir.join("index.json"), config_json)?;

        Ok(())
    }
