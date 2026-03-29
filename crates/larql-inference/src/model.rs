//! Load model weights from safetensors files.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ndarray::Array2;

use larql_models::ModelArchitecture;

use crate::error::InferenceError;

/// A loaded model's weight tensors, configuration, and architecture.
pub struct ModelWeights {
    pub tensors: HashMap<String, Array2<f32>>,
    pub vectors: HashMap<String, Vec<f32>>,
    pub embed: Array2<f32>,
    pub arch: Box<dyn ModelArchitecture>,
    // Cached from arch.config() for convenience — these are hot-path values.
    pub num_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub head_dim: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub rope_base: f64,
}

/// Load all safetensors files from a model directory.
/// Detects architecture from config.json and uses the architecture trait
/// for key prefix stripping and embed key resolution.
pub fn load_model_dir(path: impl AsRef<Path>) -> Result<ModelWeights, InferenceError> {
    let path = path.as_ref();
    if !path.is_dir() {
        return Err(InferenceError::NotADirectory(path.to_path_buf()));
    }

    // Detect architecture from config.json
    let arch = larql_models::detect_architecture(path)
        .map_err(|e| InferenceError::Parse(e.to_string()))?;

    let prefixes = arch.key_prefixes_to_strip();

    // Find safetensors files
    let mut st_files: Vec<PathBuf> = std::fs::read_dir(path)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    st_files.sort();

    if st_files.is_empty() {
        return Err(InferenceError::NoSafetensors(path.to_path_buf()));
    }

    // Load all tensors
    let mut tensors: HashMap<String, Array2<f32>> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();

    for st_path in &st_files {
        let file = std::fs::File::open(st_path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let st = safetensors::SafeTensors::deserialize(&mmap)
            .map_err(|e| InferenceError::Parse(e.to_string()))?;

        for (name, view) in st.tensors() {
            let key = normalize_key(&name, prefixes);
            let shape = view.shape();
            let data = tensor_to_f32(&view)?;

            match shape.len() {
                2 => {
                    let arr = Array2::from_shape_vec((shape[0], shape[1]), data)
                        .map_err(|e| InferenceError::Parse(e.to_string()))?;
                    tensors.insert(key, arr);
                }
                1 => {
                    vectors.insert(key, data);
                }
                _ => {} // skip 3D+ tensors
            }
        }
    }

    // Find embedding matrix using architecture's embed key
    let embed_key = arch.embed_key();
    let embed = tensors
        .get(embed_key)
        .ok_or_else(|| InferenceError::MissingTensor(embed_key.into()))?
        .clone();

    let vocab_size = embed.shape()[0];

    // Cache config values
    let cfg = arch.config();
    let num_layers = cfg.num_layers;
    let hidden_size = cfg.hidden_size;
    let intermediate_size = cfg.intermediate_size;
    let head_dim = cfg.head_dim;
    let num_q_heads = cfg.num_q_heads;
    let num_kv_heads = cfg.num_kv_heads;
    let rope_base = cfg.rope_base;

    Ok(ModelWeights {
        tensors,
        vectors,
        embed,
        arch,
        num_layers,
        hidden_size,
        intermediate_size,
        vocab_size,
        head_dim,
        num_q_heads,
        num_kv_heads,
        rope_base,
    })
}

/// Resolve a HuggingFace model ID or path to a local directory.
pub fn resolve_model_path(model: &str) -> Result<PathBuf, InferenceError> {
    let path = PathBuf::from(model);
    if path.is_dir() {
        return Ok(path);
    }

    // Try HuggingFace cache
    let cache_name = format!("models--{}", model.replace('/', "--"));
    let hf_cache = dirs_or_home().join(format!(".cache/huggingface/hub/{cache_name}/snapshots"));

    if hf_cache.is_dir() {
        if let Some(snapshot) = std::fs::read_dir(&hf_cache)
            .ok()
            .and_then(|mut d| d.next())
            .and_then(|e| e.ok())
        {
            let snapshot_path = snapshot.path();
            if snapshot_path.is_dir() {
                return Ok(snapshot_path);
            }
        }
    }

    Err(InferenceError::NotADirectory(path))
}

fn dirs_or_home() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

/// Strip known prefixes from tensor keys using the architecture's prefix list.
fn normalize_key(key: &str, prefixes: &[&str]) -> String {
    for prefix in prefixes {
        if let Some(stripped) = key.strip_prefix(prefix) {
            return stripped.to_string();
        }
    }
    key.to_string()
}

/// Convert a safetensors tensor view to Vec<f32>.
fn tensor_to_f32(view: &safetensors::tensor::TensorView<'_>) -> Result<Vec<f32>, InferenceError> {
    match view.dtype() {
        safetensors::Dtype::F32 => {
            let bytes = view.data();
            Ok(bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        }
        safetensors::Dtype::F16 => {
            let bytes = view.data();
            Ok(bytes
                .chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    half_to_f32(bits)
                })
                .collect())
        }
        safetensors::Dtype::BF16 => {
            let bytes = view.data();
            Ok(bytes
                .chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    bf16_to_f32(bits)
                })
                .collect())
        }
        other => Err(InferenceError::UnsupportedDtype(format!("{other:?}"))),
    }
}

fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        let mut e = 1u32;
        let mut m = mant;
        while (m & 0x400) == 0 {
            m <<= 1;
            e += 1;
        }
        let exp32 = (127 - 15 + 1 - e) << 23;
        let mant32 = (m & 0x3FF) << 13;
        return f32::from_bits(sign | exp32 | mant32);
    }
    if exp == 31 {
        let exp32 = 0xFF << 23;
        let mant32 = mant << 13;
        return f32::from_bits(sign | exp32 | mant32);
    }

    let exp32 = (exp + 127 - 15) << 23;
    let mant32 = mant << 13;
    f32::from_bits(sign | exp32 | mant32)
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}
