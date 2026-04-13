//! DESCRIBE types — DescribeEdge and LabelSource.
//!
//! These represent the output of a DESCRIBE operation on an entity.
//! The actual DESCRIBE logic lives in the executor (larql-lql), but these
//! types are vindex-level so they can be shared across consumers.

/// Source of a relation label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelSource {
    /// Model inference confirmed this feature encodes this relation.
    Probe,
    /// Cluster-based matching (Wikidata, WordNet, pattern detection).
    Cluster,
    /// Entity pattern detection (country, language, month, number).
    Pattern,
    /// TF-IDF fallback — no confirmed label (no tag shown in output).
    None,
    /// Architecture B: inserted via KNN store.
    KnnStore,
}

impl std::fmt::Display for LabelSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Probe => write!(f, "probe"),
            Self::Cluster => write!(f, "cluster"),
            Self::Pattern => write!(f, "pattern"),
            Self::None => write!(f, ""),
            Self::KnnStore => write!(f, "knn"),
        }
    }
}

/// A single edge from a DESCRIBE result.
#[derive(Debug, Clone)]
pub struct DescribeEdge {
    /// Relation label (e.g., "capital", "language"). None for unlabelled edges.
    pub relation: Option<String>,
    /// Where the label came from.
    pub source: LabelSource,
    /// Target token (what the feature outputs).
    pub target: String,
    /// Gate activation score.
    pub gate_score: f32,
    /// Lowest layer this edge appears in.
    pub layer_min: usize,
    /// Highest layer this edge appears in.
    pub layer_max: usize,
    /// Number of features across layers that contribute to this edge.
    pub count: usize,
    /// Additional output tokens from the strongest feature (for context).
    pub also_tokens: Vec<String>,
}
