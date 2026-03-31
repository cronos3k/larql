//! Binary loading path for .vindex directories.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::Path;

use ndarray::Array2;

use crate::error::VindexError;

use larql_models::TopKEntry;

use crate::config::{DownMetaRecord, VindexConfig};
use crate::index::{FeatureMeta, IndexLoadCallbacks, VectorIndex};

impl VectorIndex {
    /// Load a VectorIndex from a .vindex directory.
    ///
    /// Reads gate_vectors.bin (mmap'd), down_meta.jsonl, and index.json.
    /// The embeddings and tokenizer are loaded separately via `load_vindex_embeddings`.
    pub fn load_vindex(
        dir: &Path,
        callbacks: &mut dyn IndexLoadCallbacks,
    ) -> Result<Self, VindexError> {
        // Read config
        let config_path = dir.join("index.json");
        let config_text = std::fs::read_to_string(&config_path)?;
        let config: VindexConfig = serde_json::from_str(&config_text)
            .map_err(|e| VindexError::Parse(e.to_string()))?;

        let num_layers = config.num_layers;
        let hidden_size = config.hidden_size;

        // Load gate vectors from binary
        callbacks.on_file_start("gate_vectors", &dir.join("gate_vectors.bin").display().to_string());
        let start = std::time::Instant::now();

        let gate_path = dir.join("gate_vectors.bin");
        let gate_bytes = std::fs::read(&gate_path)?;
        let gate_floats: &[f32] = unsafe {
            std::slice::from_raw_parts(
                gate_bytes.as_ptr() as *const f32,
                gate_bytes.len() / 4,
            )
        };

        let mut gate_vectors: Vec<Option<Array2<f32>>> = vec![None; num_layers];
        let mut total_gate = 0;

        for info in &config.layers {
            let float_offset = info.offset as usize / 4;
            let float_count = info.num_features * hidden_size;
            let layer_data = &gate_floats[float_offset..float_offset + float_count];
            let matrix = Array2::from_shape_vec(
                (info.num_features, hidden_size),
                layer_data.to_vec(),
            )
            .map_err(|e| VindexError::Parse(e.to_string()))?;
            gate_vectors[info.layer] = Some(matrix);
            total_gate += info.num_features;
        }

        callbacks.on_file_done(
            "gate_vectors",
            total_gate,
            start.elapsed().as_secs_f64() * 1000.0,
        );

        // Load down metadata
        callbacks.on_file_start("down_meta", &dir.join("down_meta.jsonl").display().to_string());
        let start = std::time::Instant::now();

        let down_path = dir.join("down_meta.jsonl");
        let down_file = std::fs::File::open(&down_path)?;
        let reader = BufReader::with_capacity(1 << 20, down_file);

        let mut down_meta: Vec<Option<Vec<Option<FeatureMeta>>>> = vec![None; num_layers];
        let mut down_count = 0;

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let record: DownMetaRecord = serde_json::from_str(line)
                .map_err(|e| VindexError::Parse(e.to_string()))?;

            let layer = record.layer;
            let feature = record.feature;

            let meta = FeatureMeta {
                top_token: record.top_token,
                top_token_id: record.top_token_id,
                c_score: record.c_score,
                top_k: record
                    .top_k
                    .into_iter()
                    .map(|e| TopKEntry {
                        token: e.token,
                        token_id: e.token_id,
                        logit: e.logit,
                    })
                    .collect(),
            };

            if layer < num_layers {
                if down_meta[layer].is_none() {
                    down_meta[layer] = Some(Vec::new());
                }
                if let Some(ref mut metas) = down_meta[layer] {
                    while metas.len() <= feature {
                        metas.push(None);
                    }
                    metas[feature] = Some(meta);
                }
            }

            down_count += 1;
            if down_count % 10000 == 0 {
                callbacks.on_progress(down_count);
            }
        }

        callbacks.on_file_done(
            "down_meta",
            down_count,
            start.elapsed().as_secs_f64() * 1000.0,
        );

        Ok(VectorIndex::new(gate_vectors, down_meta, num_layers, hidden_size))
    }
}

/// Load embeddings from a .vindex directory.
pub fn load_vindex_embeddings(dir: &Path) -> Result<(Array2<f32>, f32), VindexError> {
    let config_text = std::fs::read_to_string(dir.join("index.json"))?;
    let config: VindexConfig = serde_json::from_str(&config_text)
        .map_err(|e| VindexError::Parse(e.to_string()))?;

    let embed_bytes = std::fs::read(dir.join("embeddings.bin"))?;
    let embed_floats: Vec<f32> = unsafe {
        std::slice::from_raw_parts(
            embed_bytes.as_ptr() as *const f32,
            embed_bytes.len() / 4,
        )
    }
    .to_vec();

    let embed = Array2::from_shape_vec((config.vocab_size, config.hidden_size), embed_floats)
        .map_err(|e| VindexError::Parse(e.to_string()))?;

    Ok((embed, config.embed_scale))
}

/// Load tokenizer from a .vindex directory.
pub fn load_vindex_tokenizer(dir: &Path) -> Result<tokenizers::Tokenizer, VindexError> {
    let path = dir.join("tokenizer.json");
    tokenizers::Tokenizer::from_file(&path).map_err(|e| VindexError::Parse(e.to_string()))
}

/// Load the vindex config.
pub fn load_vindex_config(dir: &Path) -> Result<VindexConfig, VindexError> {
    let text = std::fs::read_to_string(dir.join("index.json"))?;
    serde_json::from_str(&text).map_err(|e| VindexError::Parse(e.to_string()))
}

/// Load feature labels from down_meta.jsonl — fast hash lookup, no vocab projection.
///
/// Returns a map: (layer, feature) → top_token string.
/// Also works with the gate vectors NDJSON from vector-extract (has same fields).
pub fn load_feature_labels(path: &Path) -> Result<HashMap<(usize, usize), String>, VindexError> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::with_capacity(1 << 20, file);
    let mut labels: HashMap<(usize, usize), String> = HashMap::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let obj: serde_json::Value =
            serde_json::from_str(line).map_err(|e| VindexError::Parse(e.to_string()))?;

        if obj.get("_header").is_some() {
            continue;
        }

        // Support both compact (l/f/t) and full (layer/feature/top_token) formats
        let layer = obj
            .get("l")
            .or_else(|| obj.get("layer"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let feature = obj
            .get("f")
            .or_else(|| obj.get("feature"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let token = obj
            .get("t")
            .or_else(|| obj.get("top_token"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        labels.insert((layer, feature), token);
    }

    Ok(labels)
}
