//! Generic/fallback architecture for unknown model types.
//!
//! Uses Llama-style defaults: no embedding scaling, no norm offset,
//! no QK norm, no post-norms, standard RoPE base.

use crate::config::{ModelArchitecture, ModelConfig};

pub struct GenericArch {
    config: ModelConfig,
}

impl GenericArch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for GenericArch {
    fn family(&self) -> &str {
        "generic"
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }
}
