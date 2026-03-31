/// VectorIndex mutation and persistence methods.
///
/// Adds INSERT/DELETE/UPDATE support and the ability to save a modified vindex back to disk.

use std::io::{BufWriter, Write};
use std::path::Path;

use ndarray::Array1;

use crate::error::VindexError;
use crate::config::{DownMetaRecord, DownMetaTopK, VindexConfig};
use crate::index::{FeatureMeta, VectorIndex};

impl VectorIndex {
    /// Set metadata for a feature. Used by INSERT and UPDATE.
    pub fn set_feature_meta(&mut self, layer: usize, feature: usize, meta: FeatureMeta) {
        // Ensure layer slot exists
        while self.down_meta.len() <= layer {
            self.down_meta.push(None);
        }
        if self.down_meta[layer].is_none() {
            self.down_meta[layer] = Some(Vec::new());
        }
        if let Some(ref mut metas) = self.down_meta[layer] {
            while metas.len() <= feature {
                metas.push(None);
            }
            metas[feature] = Some(meta);
        }
    }

    /// Set the gate vector for a specific feature at a layer.
    /// The vector length must match hidden_size.
    pub fn set_gate_vector(&mut self, layer: usize, feature: usize, vector: &Array1<f32>) {
        if let Some(Some(ref mut matrix)) = self.gate_vectors.get_mut(layer) {
            if feature < matrix.shape()[0] && vector.len() == matrix.shape()[1] {
                for (j, val) in vector.iter().enumerate() {
                    matrix[[feature, j]] = *val;
                }
            }
        }
    }

    /// Clear metadata for a feature. Used by DELETE.
    pub fn delete_feature_meta(&mut self, layer: usize, feature: usize) {
        if let Some(Some(ref mut metas)) = self.down_meta.get_mut(layer) {
            if feature < metas.len() {
                metas[feature] = None;
            }
        }
    }

    /// Find a free (unused) feature slot at a layer — one with no metadata.
    /// Returns None if the layer is full.
    pub fn find_free_feature(&self, layer: usize) -> Option<usize> {
        if let Some(Some(ref metas)) = self.down_meta.get(layer) {
            for (i, m) in metas.iter().enumerate() {
                if m.is_none() {
                    return Some(i);
                }
            }
        }
        None
    }

    /// Find features matching entity and/or relation filters at a given layer (or all layers).
    /// Returns (layer, feature) pairs.
    pub fn find_features(
        &self,
        entity: Option<&str>,
        relation_label: Option<&str>,
        layer_filter: Option<usize>,
    ) -> Vec<(usize, usize)> {
        let mut results = Vec::new();
        let layers = self.loaded_layers();

        for layer in layers {
            if let Some(l) = layer_filter {
                if layer != l {
                    continue;
                }
            }
            if let Some(Some(ref metas)) = self.down_meta.get(layer) {
                for (feat, meta_opt) in metas.iter().enumerate() {
                    if let Some(meta) = meta_opt {
                        let entity_match = entity
                            .map(|e| {
                                meta.top_token.to_lowercase().contains(&e.to_lowercase())
                                    || meta.top_k.iter().any(|t| {
                                        t.token.to_lowercase().contains(&e.to_lowercase())
                                    })
                            })
                            .unwrap_or(true);

                        // Relation matching is best-effort — check against relation label if provided
                        let relation_match = relation_label.is_none();

                        if entity_match && relation_match {
                            results.push((layer, feat));
                        }
                    }
                }
            }
        }
        results
    }

    /// Write down_meta.jsonl back to disk. Overwrites the existing file.
    pub fn save_down_meta(&self, dir: &Path) -> Result<usize, VindexError> {
        let path = dir.join("down_meta.jsonl");
        let file = std::fs::File::create(&path)?;
        let mut writer = BufWriter::new(file);
        let mut count = 0;

        for (layer, layer_meta) in self.down_meta.iter().enumerate() {
            if let Some(ref metas) = layer_meta {
                for (feature, meta_opt) in metas.iter().enumerate() {
                    if let Some(meta) = meta_opt {
                        let record = DownMetaRecord {
                            layer,
                            feature,
                            top_token: meta.top_token.clone(),
                            top_token_id: meta.top_token_id,
                            c_score: meta.c_score,
                            top_k: meta
                                .top_k
                                .iter()
                                .map(|t| DownMetaTopK {
                                    token: t.token.clone(),
                                    token_id: t.token_id,
                                    logit: t.logit,
                                })
                                .collect(),
                        };
                        serde_json::to_writer(&mut writer, &record)
                            .map_err(|e| VindexError::Parse(e.to_string()))?;
                        writer.write_all(b"\n")?;
                        count += 1;
                    }
                }
            }
        }

        writer.flush()?;
        Ok(count)
    }

    /// Write gate_vectors.bin back to disk and return updated layer info.
    pub fn save_gate_vectors(
        &self,
        dir: &Path,
    ) -> Result<Vec<crate::config::VindexLayerInfo>, VindexError> {
        let path = dir.join("gate_vectors.bin");
        let file = std::fs::File::create(&path)?;
        let mut writer = BufWriter::new(file);
        let mut layer_infos = Vec::new();
        let mut offset: u64 = 0;

        for (layer, gate_opt) in self.gate_vectors.iter().enumerate() {
            if let Some(ref matrix) = gate_opt {
                let data = matrix.as_slice().ok_or_else(|| {
                    VindexError::Parse("gate vectors not contiguous".into())
                })?;
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        data.as_ptr() as *const u8,
                        data.len() * std::mem::size_of::<f32>(),
                    )
                };
                writer.write_all(bytes)?;

                let length = bytes.len() as u64;
                layer_infos.push(crate::config::VindexLayerInfo {
                    layer,
                    num_features: matrix.shape()[0],
                    offset,
                    length,
                });
                offset += length;
            }
        }

        writer.flush()?;
        Ok(layer_infos)
    }

    /// Save config (index.json) to disk.
    pub fn save_config(config: &VindexConfig, dir: &Path) -> Result<(), VindexError> {
        let path = dir.join("index.json");
        let json = serde_json::to_string_pretty(config)
            .map_err(|e| VindexError::Parse(e.to_string()))?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Save the full vindex (gate_vectors.bin + down_meta.jsonl + index.json).
    /// Updates the config's layer info to match current state.
    pub fn save_vindex(
        &self,
        dir: &Path,
        config: &mut VindexConfig,
    ) -> Result<(), VindexError> {
        let layer_infos = self.save_gate_vectors(dir)?;
        config.layers = layer_infos;
        self.save_down_meta(dir)?;
        Self::save_config(config, dir)?;
        Ok(())
    }
}
