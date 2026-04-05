use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};

use super::edge::{CompactEdge, Edge, Triple};
use super::enums::{MergeStrategy, SourceType};
use super::node::Node;
use super::schema::Schema;

/// Graph statistics.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GraphStats {
    pub entities: usize,
    pub edges: usize,
    pub relations: usize,
    pub sources: HashMap<String, usize>,
    pub avg_confidence: f64,
    pub connected_components: usize,
    pub avg_degree: f64,
}

/// A directed labeled multigraph for knowledge storage and querying.
///
/// Indexes: adjacency (subject->out), reverse (object->in),
/// keyword (token->edge indices), edge_set (dedup).
///
/// Nodes are lazily computed from edges and cached until mutation.
pub struct Graph {
    edges: Vec<Edge>,
    edge_set: HashSet<Triple>,
    adjacency: HashMap<String, Vec<(String, String, usize)>>,
    reverse: HashMap<String, Vec<(String, String, usize)>>,
    keyword_index: HashMap<String, Vec<usize>>,
    pub schema: Schema,
    pub metadata: HashMap<String, serde_json::Value>,
    nodes: RefCell<Option<HashMap<String, Node>>>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            edges: Vec::new(),
            edge_set: HashSet::new(),
            adjacency: HashMap::new(),
            reverse: HashMap::new(),
            keyword_index: HashMap::new(),
            schema: Schema::new(),
            metadata: HashMap::new(),
            nodes: RefCell::new(None),
        }
    }

    pub fn with_schema(mut self, schema: Schema) -> Self {
        self.schema = schema;
        self
    }

    // ── Construction ──

    /// Add an edge. Silently skips exact (s,r,o) duplicates.
    pub fn add_edge(&mut self, edge: Edge) {
        let triple = edge.triple();
        if self.edge_set.contains(&triple) {
            return;
        }

        let idx = self.edges.len();
        self.edge_set.insert(triple);

        self.adjacency
            .entry(edge.subject.clone())
            .or_default()
            .push((edge.relation.clone(), edge.object.clone(), idx));

        self.reverse.entry(edge.object.clone()).or_default().push((
            edge.relation.clone(),
            edge.subject.clone(),
            idx,
        ));

        self.index_keywords(&edge, idx);
        self.edges.push(edge);
        *self.nodes.borrow_mut() = None;
    }

    pub fn add_edges(&mut self, edges: impl IntoIterator<Item = Edge>) {
        for edge in edges {
            self.add_edge(edge);
        }
    }

    /// Remove duplicate (s,r,o) triples. Returns count removed.
    pub fn deduplicate(&mut self, strategy: MergeStrategy) -> usize {
        let original = self.edges.len();
        let mut seen: HashMap<Triple, Edge> = HashMap::new();

        for edge in &self.edges {
            let triple = edge.triple();
            match seen.get(&triple) {
                None => {
                    seen.insert(triple, edge.clone());
                }
                Some(existing) => {
                    if matches!(strategy, MergeStrategy::MaxConfidence)
                        && edge.confidence > existing.confidence
                    {
                        seen.insert(triple, edge.clone());
                    }
                }
            }
        }

        let kept: Vec<Edge> = seen.into_values().collect();
        self.rebuild_indexes(kept);
        original - self.edges.len()
    }

    /// Remove an edge by its (subject, relation, object) triple.
    /// Returns true if the edge was found and removed.
    pub fn remove_edge(&mut self, subject: &str, relation: &str, object: &str) -> bool {
        let triple = Triple(subject.into(), relation.into(), object.into());
        if !self.edge_set.contains(&triple) {
            return false;
        }
        let kept: Vec<Edge> = self
            .edges
            .iter()
            .filter(|e| !(e.subject == subject && e.relation == relation && e.object == object))
            .cloned()
            .collect();
        self.rebuild_indexes(kept);
        true
    }

    // ── Queries ──

    /// Select edges from a subject, optionally filtered by relation.
    pub fn select(&self, subject: &str, relation: Option<&str>) -> Vec<&Edge> {
        self.adjacency
            .get(subject)
            .map(|entries| {
                entries
                    .iter()
                    .filter(|(rel, _, _)| relation.is_none_or(|r| rel == r))
                    .map(|(_, _, idx)| &self.edges[*idx])
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Select edges pointing TO an object.
    pub fn select_reverse(&self, object: &str, relation: Option<&str>) -> Vec<&Edge> {
        self.reverse
            .get(object)
            .map(|entries| {
                entries
                    .iter()
                    .filter(|(rel, _, _)| relation.is_none_or(|r| rel == r))
                    .map(|(_, _, idx)| &self.edges[*idx])
                    .collect()
            })
            .unwrap_or_default()
    }

    /// All edges involving an entity (outgoing + incoming).
    pub fn describe(&self, entity: &str) -> DescribeResult {
        DescribeResult {
            entity: entity.to_string(),
            outgoing: self.select(entity, None).into_iter().cloned().collect(),
            incoming: self
                .select_reverse(entity, None)
                .into_iter()
                .cloned()
                .collect(),
        }
    }

    /// Check if an edge exists.
    pub fn exists(&self, subject: &str, relation: &str, object: &str) -> bool {
        self.edge_set
            .contains(&Triple(subject.into(), relation.into(), object.into()))
    }

    /// Multi-hop walk following a chain of relations.
    ///
    /// At each hop, picks the edge with the **highest confidence** when multiple
    /// edges match the same relation. Returns (final_entity, path) or None if
    /// any hop has no matching edges.
    ///
    /// For all paths (not just highest-confidence), use `walk_all_paths()`.
    pub fn walk(&self, subject: &str, relations: &[&str]) -> Option<(String, Vec<Edge>)> {
        let mut current = subject.to_string();
        let mut path = Vec::new();

        for rel in relations {
            let edges = self.select(&current, Some(rel));
            let best = edges
                .iter()
                .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())?;
            path.push((*best).clone());
            current = best.object.clone();
        }

        Some((current, path))
    }

    /// Keyword search across entity names and relations.
    pub fn search(&self, query: &str, max_results: usize) -> Vec<&Edge> {
        let query_lower = query.to_lowercase();
        let tokens: Vec<&str> = query_lower.split_whitespace().collect();
        let mut scores: HashMap<usize, usize> = HashMap::new();

        for token in &tokens {
            if let Some(indices) = self.keyword_index.get(*token) {
                for &idx in indices {
                    *scores.entry(idx).or_insert(0) += 1;
                }
            }
        }

        let mut ranked: Vec<(usize, usize)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1));
        ranked.truncate(max_results);
        ranked.iter().map(|(idx, _)| &self.edges[*idx]).collect()
    }

    /// BFS neighbourhood extraction.
    pub fn subgraph(&self, entity: &str, depth: u32) -> Graph {
        let mut sub = Graph::new();
        sub.schema = self.schema.clone();

        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<(String, u32)> = VecDeque::new();
        queue.push_back((entity.to_string(), 0));

        while let Some((node, d)) = queue.pop_front() {
            if d > depth || visited.contains(&node) {
                continue;
            }
            visited.insert(node.clone());

            for edge in self.select(&node, None) {
                sub.add_edge(edge.clone());
                if d < depth {
                    queue.push_back((edge.object.clone(), d + 1));
                }
            }
        }
        sub
    }

    // ── Accessors ──

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    pub fn node_count(&self) -> usize {
        self.ensure_nodes();
        self.nodes.borrow().as_ref().map_or(0, |n| n.len())
    }

    pub fn nodes(&self) -> Vec<Node> {
        self.ensure_nodes();
        self.nodes
            .borrow()
            .as_ref()
            .map(|n| n.values().cloned().collect())
            .unwrap_or_default()
    }

    pub fn list_relations(&self) -> Vec<String> {
        self.edges
            .iter()
            .map(|e| e.relation.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect()
    }

    pub fn list_entities(&self) -> Vec<String> {
        self.ensure_nodes();
        self.nodes
            .borrow()
            .as_ref()
            .map(|n| n.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Count edges, optionally filtered by relation and/or source.
    pub fn count(&self, relation: Option<&str>, source: Option<&SourceType>) -> usize {
        self.edges
            .iter()
            .filter(|e| relation.is_none_or(|r| e.relation == r))
            .filter(|e| source.is_none_or(|s| &e.source == s))
            .count()
    }

    /// Get a single node by name.
    pub fn node(&self, name: &str) -> Option<Node> {
        self.ensure_nodes();
        self.nodes
            .borrow()
            .as_ref()
            .and_then(|n| n.get(name).cloned())
    }

    // ── Stats ──

    pub fn stats(&self) -> GraphStats {
        self.ensure_nodes();
        let nodes_ref = self.nodes.borrow();
        let nodes = nodes_ref.as_ref();

        let mut source_counts: HashMap<String, usize> = HashMap::new();
        let mut conf_sum = 0.0f64;
        for edge in &self.edges {
            *source_counts
                .entry(edge.source.as_str().to_string())
                .or_insert(0) += 1;
            conf_sum += edge.confidence;
        }

        let node_count = nodes.map_or(0, |n| n.len());
        let avg_degree = if node_count > 0 {
            nodes.map_or(0.0, |n| {
                n.values().map(|n| n.degree as f64).sum::<f64>() / node_count as f64
            })
        } else {
            0.0
        };

        GraphStats {
            entities: node_count,
            edges: self.edges.len(),
            relations: self.list_relations().len(),
            sources: source_counts,
            avg_confidence: if self.edges.is_empty() {
                0.0
            } else {
                conf_sum / self.edges.len() as f64
            },
            connected_components: self.count_components(),
            avg_degree,
        }
    }

    // ── Serialization ──

    /// Serialize to .larql.json format — matches Python exactly.
    pub fn to_json_value(&self) -> serde_json::Value {
        serde_json::json!({
            "larql_version": "0.1.0",
            "metadata": self.metadata,
            "schema": self.schema.to_json_value(),
            "edges": self.edges.iter()
                .map(|e| serde_json::to_value(CompactEdge::from(e)).unwrap())
                .collect::<Vec<_>>(),
        })
    }

    /// Deserialize from .larql.json format.
    pub fn from_json_value(v: &serde_json::Value) -> Result<Self, GraphError> {
        let schema = v
            .get("schema")
            .map(Schema::from_json_value)
            .unwrap_or_default();

        let metadata: HashMap<String, serde_json::Value> = v
            .get("metadata")
            .and_then(|m| serde_json::from_value(m.clone()).ok())
            .unwrap_or_default();

        let mut graph = Graph::new().with_schema(schema);
        graph.metadata = metadata;

        if let Some(edges) = v.get("edges").and_then(|e| e.as_array()) {
            for edge_val in edges {
                let compact: CompactEdge = serde_json::from_value(edge_val.clone())
                    .map_err(|e| GraphError::Deserialize(e.to_string()))?;
                graph.add_edge(Edge::from(compact));
            }
        }

        Ok(graph)
    }

    // ── Private ──

    fn index_keywords(&mut self, edge: &Edge, idx: usize) {
        let mut tokens: HashSet<String> = HashSet::new();
        for text in [&edge.subject, &edge.relation, &edge.object] {
            for token in text.to_lowercase().replace('-', " ").split_whitespace() {
                tokens.insert(token.to_string());
            }
        }
        for token in tokens {
            self.keyword_index.entry(token).or_default().push(idx);
        }
    }

    fn rebuild_indexes(&mut self, edges: Vec<Edge>) {
        self.edges.clear();
        self.edge_set.clear();
        self.adjacency.clear();
        self.reverse.clear();
        self.keyword_index.clear();
        *self.nodes.borrow_mut() = None;
        for edge in edges {
            self.add_edge(edge);
        }
    }

    fn ensure_nodes(&self) {
        if self.nodes.borrow().is_some() {
            return;
        }

        let mut out_rels: HashMap<&str, HashSet<String>> = HashMap::new();
        let mut in_rels: HashMap<&str, HashSet<String>> = HashMap::new();
        let mut out_deg: HashMap<&str, usize> = HashMap::new();
        let mut in_deg: HashMap<&str, usize> = HashMap::new();

        for edge in &self.edges {
            out_rels
                .entry(&edge.subject)
                .or_default()
                .insert(edge.relation.clone());
            in_rels
                .entry(&edge.object)
                .or_default()
                .insert(edge.relation.clone());
            *out_deg.entry(&edge.subject).or_insert(0) += 1;
            *in_deg.entry(&edge.object).or_insert(0) += 1;
        }

        let all_entities: HashSet<&str> = out_deg.keys().chain(in_deg.keys()).copied().collect();

        let mut nodes = HashMap::new();
        for name in all_entities {
            let out = out_deg.get(name).copied().unwrap_or(0);
            let inp = in_deg.get(name).copied().unwrap_or(0);
            let out_set = out_rels.get(name).cloned().unwrap_or_default();
            let in_set = in_rels.get(name).cloned().unwrap_or_default();
            nodes.insert(
                name.to_string(),
                Node {
                    name: name.to_string(),
                    node_type: self.schema.infer_type(&out_set, &in_set),
                    degree: out + inp,
                    out_degree: out,
                    in_degree: inp,
                },
            );
        }

        *self.nodes.borrow_mut() = Some(nodes);
    }

    fn count_components(&self) -> usize {
        let mut visited: HashSet<&str> = HashSet::new();
        let mut components = 0;

        let mut adj: HashMap<&str, HashSet<&str>> = HashMap::new();
        for edge in &self.edges {
            adj.entry(&edge.subject).or_default().insert(&edge.object);
            adj.entry(&edge.object).or_default().insert(&edge.subject);
        }

        for entity in adj.keys() {
            if visited.contains(entity) {
                continue;
            }
            components += 1;
            let mut queue = VecDeque::new();
            queue.push_back(*entity);
            while let Some(node) = queue.pop_front() {
                if !visited.insert(node) {
                    continue;
                }
                if let Some(neighbors) = adj.get(node) {
                    for n in neighbors {
                        if !visited.contains(n) {
                            queue.push_back(n);
                        }
                    }
                }
            }
        }
        components
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Graph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Graph(edges={}, nodes={})",
            self.edge_count(),
            self.node_count()
        )
    }
}

/// Result of describe().
#[derive(Debug)]
pub struct DescribeResult {
    pub entity: String,
    pub outgoing: Vec<Edge>,
    pub incoming: Vec<Edge>,
}

#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error("deserialization failed: {0}")]
    Deserialize(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
