//! Binary loading path for .vindex directories.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::Path;

use ndarray::Array2;

use crate::error::VindexError;
use crate::config::VindexConfig;
use crate::index::{IndexLoadCallbacks, VectorIndex};

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
        let gate_file = std::fs::File::open(&gate_path)?;
        let gate_mmap = unsafe { memmap2::Mmap::map(&gate_file)? };
        let bpf = crate::config::dtype::bytes_per_float(config.dtype);

        // Build per-layer slice info — offsets in floats (not bytes)
        let mut gate_slices: Vec<crate::index::core::GateLayerSlice> = vec![
            crate::index::core::GateLayerSlice { float_offset: 0, num_features: 0 };
            num_layers
        ];
        let mut total_gate = 0;

        for info in &config.layers {
            gate_slices[info.layer] = crate::index::core::GateLayerSlice {
                float_offset: info.offset as usize / bpf,
                num_features: info.num_features,
            };
            total_gate += info.num_features;
        }

        callbacks.on_file_done(
            "gate_vectors",
            total_gate,
            start.elapsed().as_secs_f64() * 1000.0,
        );

        // Load down metadata — mmap binary (zero heap), fall back to JSONL (legacy)
        let start = std::time::Instant::now();

        let down_meta_mmap = if crate::format::down_meta::has_binary(dir) {
            match load_vindex_tokenizer(dir) {
                Ok(tokenizer) => {
                    callbacks.on_file_start("down_meta", &dir.join("down_meta.bin").display().to_string());
                    let tok = std::sync::Arc::new(tokenizer);
                    match crate::format::down_meta::mmap_binary(dir, tok) {
                        Ok(dm) => {
                            let count = dm.total_features();
                            callbacks.on_file_done("down_meta", count, start.elapsed().as_secs_f64() * 1000.0);
                            Some(dm)
                        }
                        Err(_) => None,
                    }
                }
                Err(_) => None,
            }
        } else {
            None
        };

        Ok(VectorIndex::new_mmap(gate_mmap, gate_slices, config.dtype, down_meta_mmap, num_layers, hidden_size))
    }
}

/// Load embeddings from a .vindex directory.
pub fn load_vindex_embeddings(dir: &Path) -> Result<(Array2<f32>, f32), VindexError> {
    let config_text = std::fs::read_to_string(dir.join("index.json"))?;
    let config: VindexConfig = serde_json::from_str(&config_text)
        .map_err(|e| VindexError::Parse(e.to_string()))?;

    let embed_file = std::fs::File::open(dir.join("embeddings.bin"))?;
    let embed_mmap = unsafe { memmap2::Mmap::map(&embed_file)? };
    // Detect actual dtype from file size (may differ from index.json global dtype
    // if gate vectors were converted to f32 but embeddings remain f16).
    let expected_f32 = config.vocab_size * config.hidden_size * 4;
    let actual_dtype = if embed_mmap.len() == expected_f32 {
        crate::config::dtype::StorageDtype::F32
    } else {
        crate::config::dtype::StorageDtype::F16
    };
    let embed_floats = crate::config::dtype::decode_floats(&embed_mmap, actual_dtype);

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
