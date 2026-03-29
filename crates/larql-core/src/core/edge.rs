use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use super::enums::SourceType;

/// A directed, labeled edge in the knowledge graph.
///
/// Every fact is an edge: subject --relation--> object.
/// Immutable by convention — build with the builder pattern,
/// don't mutate after insertion into a Graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub subject: String,
    pub relation: String,
    pub object: String,
    pub confidence: f64,
    #[serde(default)]
    pub source: SourceType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub injection: Option<(usize, f64)>,
}

impl Edge {
    pub fn new(
        subject: impl Into<String>,
        relation: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            relation: relation.into(),
            object: object.into(),
            confidence: 1.0,
            source: SourceType::Unknown,
            metadata: None,
            injection: None,
        }
    }

    pub fn with_confidence(mut self, c: f64) -> Self {
        self.confidence = c.clamp(0.0, 1.0);
        self
    }

    pub fn with_source(mut self, s: SourceType) -> Self {
        self.source = s;
        self
    }

    pub fn with_metadata(mut self, key: &str, value: serde_json::Value) -> Self {
        self.metadata
            .get_or_insert_with(HashMap::new)
            .insert(key.to_string(), value);
        self
    }

    pub fn triple(&self) -> Triple {
        Triple(
            self.subject.clone(),
            self.relation.clone(),
            self.object.clone(),
        )
    }
}

// Equality and hashing based on the (s, r, o) triple only.
impl PartialEq for Edge {
    fn eq(&self, other: &Self) -> bool {
        self.subject == other.subject
            && self.relation == other.relation
            && self.object == other.object
    }
}
impl Eq for Edge {}

impl Hash for Edge {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.subject.hash(state);
        self.relation.hash(state);
        self.object.hash(state);
    }
}

/// Owned (subject, relation, object) triple for set membership.
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct Triple(pub String, pub String, pub String);

// ── Compact JSON serialization ──
// Matches Python's to_compact() / from_compact() exactly.

#[derive(Debug, Serialize, Deserialize)]
pub struct CompactEdge {
    pub s: String,
    pub r: String,
    pub o: String,
    #[serde(default = "default_confidence")]
    pub c: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub src: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<HashMap<String, serde_json::Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inj: Option<(usize, f64)>,
}

fn default_confidence() -> f64 {
    1.0
}

impl From<&Edge> for CompactEdge {
    fn from(e: &Edge) -> Self {
        Self {
            s: e.subject.clone(),
            r: e.relation.clone(),
            o: e.object.clone(),
            c: e.confidence,
            src: match e.source {
                SourceType::Unknown => None,
                ref s => Some(s.as_str().to_string()),
            },
            meta: e.metadata.clone(),
            inj: e.injection,
        }
    }
}

impl From<CompactEdge> for Edge {
    fn from(c: CompactEdge) -> Self {
        Self {
            subject: c.s,
            relation: c.r,
            object: c.o,
            confidence: c.c,
            source: c.src.as_deref().map_or(SourceType::Unknown, |s| {
                serde_json::from_value(serde_json::Value::String(s.to_string()))
                    .unwrap_or(SourceType::Unknown)
            }),
            metadata: c.meta,
            injection: c.inj,
        }
    }
}
