//! Shared NDJSON vector record types.
//!
//! These types are the interchange format between extraction (inference)
//! and loading (surreal). Defined here so both crates can use them
//! without depending on each other.

/// Component name constants — strings, not enums.
pub const COMPONENT_FFN_DOWN: &str = "ffn_down";
pub const COMPONENT_FFN_GATE: &str = "ffn_gate";
pub const COMPONENT_FFN_UP: &str = "ffn_up";
pub const COMPONENT_ATTN_OV: &str = "attn_ov";
pub const COMPONENT_ATTN_QK: &str = "attn_qk";
pub const COMPONENT_EMBEDDINGS: &str = "embeddings";

pub const ALL_COMPONENTS: &[&str] = &[
    COMPONENT_FFN_DOWN,
    COMPONENT_FFN_GATE,
    COMPONENT_FFN_UP,
    COMPONENT_ATTN_OV,
    COMPONENT_ATTN_QK,
    COMPONENT_EMBEDDINGS,
];

/// A single extracted vector with metadata.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct VectorRecord {
    pub id: String,
    pub layer: usize,
    pub feature: usize,
    pub vector: Vec<f32>,
    pub dim: usize,
    pub top_token: String,
    pub top_token_id: u32,
    pub c_score: f32,
    pub top_k: Vec<TopKEntry>,
}

/// A top-k token entry with logit score.
#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct TopKEntry {
    pub token: String,
    pub token_id: u32,
    pub logit: f32,
}

/// Header line written as first line of each NDJSON file.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct VectorFileHeader {
    pub _header: bool,
    pub component: String,
    pub model: String,
    pub dimension: usize,
    pub extraction_date: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_top_k_entry() -> TopKEntry {
        TopKEntry {
            token: "hello".to_string(),
            token_id: 42,
            logit: 2.75,
        }
    }

    fn sample_vector_record() -> VectorRecord {
        VectorRecord {
            id: "layer0_feat1".to_string(),
            layer: 0,
            feature: 1,
            vector: vec![0.1, 0.2, 0.3],
            dim: 3,
            top_token: "hello".to_string(),
            top_token_id: 42,
            c_score: 0.95,
            top_k: vec![sample_top_k_entry()],
        }
    }

    fn sample_header() -> VectorFileHeader {
        VectorFileHeader {
            _header: true,
            component: "ffn_down".to_string(),
            model: "test-model".to_string(),
            dimension: 128,
            extraction_date: "2026-03-29".to_string(),
        }
    }

    #[test]
    fn vector_record_json_roundtrip() {
        let record = sample_vector_record();
        let json = serde_json::to_string(&record).expect("serialize VectorRecord");
        let back: VectorRecord = serde_json::from_str(&json).expect("deserialize VectorRecord");

        assert_eq!(back.id, "layer0_feat1");
        assert_eq!(back.layer, 0);
        assert_eq!(back.feature, 1);
        assert_eq!(back.vector, vec![0.1_f32, 0.2, 0.3]);
        assert_eq!(back.dim, 3);
        assert_eq!(back.top_token, "hello");
        assert_eq!(back.top_token_id, 42);
        assert!((back.c_score - 0.95).abs() < f32::EPSILON);
        assert_eq!(back.top_k.len(), 1);
        assert_eq!(back.top_k[0].token, "hello");
    }

    #[test]
    fn top_k_entry_clone_and_serialize() {
        let entry = sample_top_k_entry();
        let cloned = entry.clone();

        assert_eq!(cloned.token, entry.token);
        assert_eq!(cloned.token_id, entry.token_id);
        assert!((cloned.logit - entry.logit).abs() < f32::EPSILON);

        let json = serde_json::to_string(&cloned).expect("serialize TopKEntry");
        assert!(json.contains("\"token\":\"hello\""));
        assert!(json.contains("\"token_id\":42"));
    }

    #[test]
    fn vector_file_header_json_roundtrip() {
        let header = sample_header();
        let json = serde_json::to_string(&header).expect("serialize VectorFileHeader");
        let back: VectorFileHeader =
            serde_json::from_str(&json).expect("deserialize VectorFileHeader");

        assert!(back._header);
        assert_eq!(back.component, "ffn_down");
        assert_eq!(back.model, "test-model");
        assert_eq!(back.dimension, 128);
        assert_eq!(back.extraction_date, "2026-03-29");
    }

    #[test]
    fn all_components_contains_all_six() {
        assert_eq!(ALL_COMPONENTS.len(), 6);
        assert!(ALL_COMPONENTS.contains(&COMPONENT_FFN_DOWN));
        assert!(ALL_COMPONENTS.contains(&COMPONENT_FFN_GATE));
        assert!(ALL_COMPONENTS.contains(&COMPONENT_FFN_UP));
        assert!(ALL_COMPONENTS.contains(&COMPONENT_ATTN_OV));
        assert!(ALL_COMPONENTS.contains(&COMPONENT_ATTN_QK));
        assert!(ALL_COMPONENTS.contains(&COMPONENT_EMBEDDINGS));
    }

    #[test]
    fn component_constants_match_expected_strings() {
        assert_eq!(COMPONENT_FFN_DOWN, "ffn_down");
        assert_eq!(COMPONENT_FFN_GATE, "ffn_gate");
        assert_eq!(COMPONENT_FFN_UP, "ffn_up");
        assert_eq!(COMPONENT_ATTN_OV, "attn_ov");
        assert_eq!(COMPONENT_ATTN_QK, "attn_qk");
        assert_eq!(COMPONENT_EMBEDDINGS, "embeddings");
    }
}
