pub mod config;
pub mod detect;
pub mod gemma3;
pub mod generic;
pub mod vectors;

pub use config::{Activation, FfnType, ModelArchitecture, ModelConfig, NormType};
pub use detect::{detect_architecture, detect_from_json, ModelError};
pub use gemma3::Gemma3Arch;
pub use generic::GenericArch;
pub use vectors::{
    TopKEntry, VectorFileHeader, VectorRecord, ALL_COMPONENTS, COMPONENT_ATTN_OV,
    COMPONENT_ATTN_QK, COMPONENT_EMBEDDINGS, COMPONENT_FFN_DOWN, COMPONENT_FFN_GATE,
    COMPONENT_FFN_UP,
};
