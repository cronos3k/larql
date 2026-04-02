use pyo3::prelude::*;
use pyo3::types::PyDict;

use larql_core as lq;
use larql_inference as li;

mod vindex;
mod session;

use vindex::{PyVindex, PyFeatureMeta, PyWalkHit, PyDescribeEdge, PyRelation};
use session::PySession;

// ── Helpers ──

fn parse_source(s: &str) -> lq::SourceType {
    match s {
        "parametric" => lq::SourceType::Parametric,
        "document" => lq::SourceType::Document,
        "installed" => lq::SourceType::Installed,
        "wikidata" => lq::SourceType::Wikidata,
        "manual" => lq::SourceType::Manual,
        _ => lq::SourceType::Unknown,
    }
}

fn parse_merge_strategy(s: &str) -> lq::MergeStrategy {
    match s {
        "union" => lq::MergeStrategy::Union,
        "source_priority" => lq::MergeStrategy::SourcePriority,
        _ => lq::MergeStrategy::MaxConfidence,
    }
}

// ── PyEdge ──

#[pyclass(name = "Edge")]
#[derive(Clone)]
pub struct PyEdge {
    inner: lq::Edge,
}

#[pymethods]
impl PyEdge {
    #[new]
    #[pyo3(signature = (subject, relation, object, confidence=1.0, source="unknown", metadata=None, injection=None))]
    fn new(
        subject: &str,
        relation: &str,
        object: &str,
        confidence: f64,
        source: &str,
        metadata: Option<&Bound<'_, PyDict>>,
        injection: Option<(usize, f64)>,
    ) -> PyResult<Self> {
        let mut edge = lq::Edge::new(subject, relation, object)
            .with_confidence(confidence)
            .with_source(parse_source(source));

        if let Some(meta) = metadata {
            for (k, v) in meta.iter() {
                let key: String = k.extract()?;
                let val_str: String = v.str()?.to_string();
                edge = edge.with_metadata(&key, serde_json::Value::String(val_str));
            }
        }

        edge.injection = injection;

        Ok(Self { inner: edge })
    }

    #[getter]
    fn subject(&self) -> &str {
        &self.inner.subject
    }

    #[getter]
    fn relation(&self) -> &str {
        &self.inner.relation
    }

    #[getter]
    fn object(&self) -> &str {
        &self.inner.object
    }

    #[getter]
    fn confidence(&self) -> f64 {
        self.inner.confidence
    }

    #[getter]
    fn source(&self) -> &str {
        self.inner.source.as_str()
    }

    #[getter]
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        match &self.inner.metadata {
            None => Ok(None),
            Some(meta) => {
                let json_str = serde_json::to_string(meta)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                let json_mod = py.import("json")?;
                let result = json_mod.call_method1("loads", (json_str,))?;
                Ok(Some(result))
            }
        }
    }

    #[getter]
    fn injection(&self) -> Option<(usize, f64)> {
        self.inner.injection
    }

    fn triple(&self) -> (String, String, String) {
        let t = self.inner.triple();
        (t.0, t.1, t.2)
    }

    fn to_compact<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("s", &self.inner.subject)?;
        dict.set_item("r", &self.inner.relation)?;
        dict.set_item("o", &self.inner.object)?;
        dict.set_item("c", self.inner.confidence)?;
        if self.inner.source != lq::SourceType::Unknown {
            dict.set_item("src", self.inner.source.as_str())?;
        }
        if let Some(meta) = &self.inner.metadata {
            let json_str = serde_json::to_string(meta)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            let json_mod = py.import("json")?;
            let parsed = json_mod.call_method1("loads", (json_str,))?;
            dict.set_item("meta", parsed)?;
        }
        if let Some(inj) = self.inner.injection {
            dict.set_item("inj", vec![inj.0 as f64, inj.1])?;
        }
        Ok(dict)
    }

    #[staticmethod]
    fn from_compact(d: &Bound<'_, PyDict>) -> PyResult<PyEdge> {
        let s: String = d.get_item("s")?.unwrap().extract()?;
        let r: String = d.get_item("r")?.unwrap().extract()?;
        let o: String = d.get_item("o")?.unwrap().extract()?;
        let c: f64 = d
            .get_item("c")?
            .map(|v| v.extract().unwrap_or(1.0))
            .unwrap_or(1.0);
        let src: String = d
            .get_item("src")?
            .map(|v| v.extract().unwrap_or_else(|_| "unknown".to_string()))
            .unwrap_or_else(|| "unknown".to_string());

        let mut edge = lq::Edge::new(s, r, o)
            .with_confidence(c)
            .with_source(parse_source(&src));

        // Parse metadata dict
        if let Some(meta_obj) = d.get_item("meta")? {
            let json_mod = d.py().import("json")?;
            let meta_str: String = json_mod.call_method1("dumps", (meta_obj,))?.extract()?;
            if let Ok(meta_map) = serde_json::from_str::<
                std::collections::HashMap<String, serde_json::Value>,
            >(&meta_str)
            {
                edge.metadata = Some(meta_map);
            }
        }

        // Parse injection
        if let Some(inj) = d.get_item("inj")? {
            let vals: Vec<f64> = inj.extract()?;
            if vals.len() == 2 {
                edge.injection = Some((vals[0] as usize, vals[1]));
            }
        }

        Ok(PyEdge { inner: edge })
    }

    fn __repr__(&self) -> String {
        format!(
            "{} --{}--> {} ({:.2})",
            self.inner.subject, self.inner.relation, self.inner.object, self.inner.confidence
        )
    }

    fn __eq__(&self, other: &PyEdge) -> bool {
        self.inner == other.inner
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }
}

// ── PyNode ──

#[pyclass(name = "Node")]
#[derive(Clone)]
pub struct PyNode {
    inner: lq::core::node::Node,
}

#[pymethods]
impl PyNode {
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    /// Returns the node type string, or "unknown" if not inferred.
    /// Matches Python NodeType enum values.
    #[getter]
    fn node_type(&self) -> &str {
        self.inner.node_type.as_deref().unwrap_or("unknown")
    }

    #[getter]
    fn degree(&self) -> usize {
        self.inner.degree
    }

    #[getter]
    fn out_degree(&self) -> usize {
        self.inner.out_degree
    }

    #[getter]
    fn in_degree(&self) -> usize {
        self.inner.in_degree
    }

    fn __repr__(&self) -> String {
        let ntype = self.inner.node_type.as_deref().unwrap_or("unknown");
        format!(
            "Node({}, type={}, degree={})",
            self.inner.name, ntype, self.inner.degree
        )
    }
}

// ── PyGraph ──

#[pyclass(name = "Graph", unsendable)]
pub struct PyGraph {
    inner: lq::Graph,
}

#[pymethods]
impl PyGraph {
    #[new]
    fn new() -> Self {
        Self {
            inner: lq::Graph::new(),
        }
    }

    // ── Construction ──

    fn add_edge(&mut self, edge: &PyEdge) {
        self.inner.add_edge(edge.inner.clone());
    }

    fn add_edges(&mut self, edges: Vec<PyEdge>) {
        self.inner.add_edges(edges.into_iter().map(|e| e.inner));
    }

    fn remove_edge(&mut self, subject: &str, relation: &str, object_: &str) -> bool {
        self.inner.remove_edge(subject, relation, object_)
    }

    #[pyo3(signature = (strategy="max_confidence"))]
    fn deduplicate(&mut self, strategy: &str) -> usize {
        self.inner.deduplicate(parse_merge_strategy(strategy))
    }

    // ── Queries ──

    #[pyo3(signature = (subject, relation=None))]
    fn select(&self, subject: &str, relation: Option<&str>) -> Vec<PyEdge> {
        self.inner
            .select(subject, relation)
            .into_iter()
            .map(|e| PyEdge { inner: e.clone() })
            .collect()
    }

    #[pyo3(signature = (object_, relation=None))]
    fn select_reverse(&self, object_: &str, relation: Option<&str>) -> Vec<PyEdge> {
        self.inner
            .select_reverse(object_, relation)
            .into_iter()
            .map(|e| PyEdge { inner: e.clone() })
            .collect()
    }

    /// Matches Python: returns {"entity", "type", "outgoing", "incoming"}
    fn describe<'py>(&self, py: Python<'py>, entity: &str) -> PyResult<Bound<'py, PyDict>> {
        let result = self.inner.describe(entity);
        let node_type = self
            .inner
            .node(entity)
            .and_then(|n| n.node_type)
            .unwrap_or_else(|| "unknown".to_string());

        let dict = PyDict::new(py);
        dict.set_item("entity", &result.entity)?;
        dict.set_item("type", node_type)?;
        dict.set_item(
            "outgoing",
            result
                .outgoing
                .into_iter()
                .map(|e| PyEdge { inner: e })
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "incoming",
            result
                .incoming
                .into_iter()
                .map(|e| PyEdge { inner: e })
                .collect::<Vec<_>>(),
        )?;
        Ok(dict)
    }

    /// Matches Python: exists(subject, relation=None, object_=None)
    #[pyo3(signature = (subject, relation=None, object_=None))]
    fn exists(&self, subject: &str, relation: Option<&str>, object_: Option<&str>) -> bool {
        match (relation, object_) {
            (Some(r), Some(o)) => self.inner.exists(subject, r, o),
            (Some(r), None) => !self.inner.select(subject, Some(r)).is_empty(),
            (None, Some(o)) => self
                .inner
                .select(subject, None)
                .iter()
                .any(|e| e.object == o),
            (None, None) => !self.inner.select(subject, None).is_empty(),
        }
    }

    /// Matches Python: returns (None, path) on failure instead of None
    fn walk(&self, subject: &str, relations: Vec<String>) -> (Option<String>, Vec<PyEdge>) {
        let refs: Vec<&str> = relations.iter().map(|s| s.as_str()).collect();
        match self.inner.walk(subject, &refs) {
            Some((dest, path)) => (
                Some(dest),
                path.into_iter().map(|e| PyEdge { inner: e }).collect(),
            ),
            None => (None, Vec::new()),
        }
    }

    #[pyo3(signature = (query, max_results=10))]
    fn search(&self, query: &str, max_results: usize) -> Vec<PyEdge> {
        self.inner
            .search(query, max_results)
            .into_iter()
            .map(|e| PyEdge { inner: e.clone() })
            .collect()
    }

    #[pyo3(signature = (entity, depth=2))]
    fn subgraph(&self, entity: &str, depth: u32) -> PyGraph {
        PyGraph {
            inner: self.inner.subgraph(entity, depth),
        }
    }

    // ── Count / Node ──

    #[pyo3(signature = (relation=None, source=None))]
    fn count(&self, relation: Option<&str>, source: Option<&str>) -> usize {
        let source_type = source.map(parse_source);
        self.inner.count(relation, source_type.as_ref())
    }

    fn node(&self, name: &str) -> Option<PyNode> {
        self.inner.node(name).map(|n| PyNode { inner: n })
    }

    fn nodes(&self) -> Vec<PyNode> {
        self.inner
            .nodes()
            .into_iter()
            .map(|n| PyNode { inner: n })
            .collect()
    }

    // ── Accessors ──

    #[getter]
    fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }

    #[getter]
    fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    #[getter]
    fn edges(&self) -> Vec<PyEdge> {
        self.inner
            .edges()
            .iter()
            .map(|e| PyEdge { inner: e.clone() })
            .collect()
    }

    /// Matches Python: list_entities(entity_type=None)
    #[pyo3(signature = (entity_type=None))]
    fn list_entities(&self, entity_type: Option<&str>) -> Vec<String> {
        match entity_type {
            None => self.inner.list_entities(),
            Some(t) => self
                .inner
                .nodes()
                .into_iter()
                .filter(|n| n.node_type.as_deref().unwrap_or("unknown") == t)
                .map(|n| n.name)
                .collect(),
        }
    }

    fn list_relations(&self) -> Vec<String> {
        self.inner.list_relations()
    }

    // ── Stats ──

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.inner.stats();
        let dict = PyDict::new(py);
        dict.set_item("entities", s.entities)?;
        dict.set_item("edges", s.edges)?;
        dict.set_item("relations", s.relations)?;
        dict.set_item("avg_confidence", s.avg_confidence)?;
        dict.set_item("connected_components", s.connected_components)?;
        dict.set_item("avg_degree", s.avg_degree)?;
        let sources = PyDict::new(py);
        for (k, v) in &s.sources {
            sources.set_item(k, v)?;
        }
        dict.set_item("sources", sources)?;
        Ok(dict)
    }

    // ── Serialization ──

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let json_val = self.inner.to_json_value();
        let json_str = serde_json::to_string(&json_val)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let json_mod = py.import("json")?;
        json_mod.call_method1("loads", (json_str,))
    }

    #[staticmethod]
    fn from_dict(data: &Bound<'_, PyAny>) -> PyResult<PyGraph> {
        let json_mod = data.py().import("json")?;
        let json_str: String = json_mod.call_method1("dumps", (data,))?.extract()?;
        let json_val: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let graph = lq::Graph::from_json_value(&json_val)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyGraph { inner: graph })
    }

    // ── Dunder ──

    fn __len__(&self) -> usize {
        self.inner.edge_count()
    }

    fn __repr__(&self) -> String {
        format!(
            "Graph(edges={}, nodes={})",
            self.inner.edge_count(),
            self.inner.node_count()
        )
    }

    fn __contains__(&self, edge: &PyEdge) -> bool {
        self.inner.exists(
            &edge.inner.subject,
            &edge.inner.relation,
            &edge.inner.object,
        )
    }
}

// ── Free functions ──

#[pyfunction]
fn load(path: &str) -> PyResult<PyGraph> {
    let graph = lq::load(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    Ok(PyGraph { inner: graph })
}

#[pyfunction]
fn save(graph: &PyGraph, path: &str) -> PyResult<()> {
    lq::save(&graph.inner, path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

#[pyfunction]
fn shortest_path(graph: &PyGraph, from: &str, to: &str) -> Option<(f64, Vec<PyEdge>)> {
    lq::shortest_path(&graph.inner, from, to).map(|(cost, path)| {
        (
            cost,
            path.into_iter().map(|e| PyEdge { inner: e }).collect(),
        )
    })
}

#[pyfunction]
fn merge_graphs(target: &mut PyGraph, other: &PyGraph) -> usize {
    lq::merge_graphs(&mut target.inner, &other.inner)
}

#[pyfunction]
#[pyo3(signature = (target, other, strategy="union"))]
fn merge_graphs_with_strategy(target: &mut PyGraph, other: &PyGraph, strategy: &str) -> usize {
    let s = match strategy {
        "max_confidence" => lq::MergeStrategy::MaxConfidence,
        "source_priority" => lq::MergeStrategy::SourcePriority,
        _ => lq::MergeStrategy::Union,
    };
    lq::merge_graphs_with_strategy(&mut target.inner, &other.inner, s)
}

#[pyfunction]
fn diff<'py>(py: Python<'py>, old: &PyGraph, new: &PyGraph) -> PyResult<Bound<'py, PyDict>> {
    let result = lq::diff(&old.inner, &new.inner);
    let dict = PyDict::new(py);
    dict.set_item(
        "added",
        result
            .added
            .into_iter()
            .map(|e| PyEdge { inner: e })
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "removed",
        result
            .removed
            .into_iter()
            .map(|e| PyEdge { inner: e })
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("changed", result.changed.len())?;
    Ok(dict)
}

#[pyfunction]
#[pyo3(signature = (graph, damping=0.85, max_iterations=100, tolerance=1e-6))]
fn pagerank<'py>(
    py: Python<'py>,
    graph: &PyGraph,
    damping: f64,
    max_iterations: usize,
    tolerance: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let result = lq::pagerank(&graph.inner, damping, max_iterations, tolerance);
    let dict = PyDict::new(py);
    let ranks = PyDict::new(py);
    for (k, v) in &result.ranks {
        ranks.set_item(k, v)?;
    }
    dict.set_item("ranks", ranks)?;
    dict.set_item("iterations", result.iterations)?;
    dict.set_item("converged", result.converged)?;
    Ok(dict)
}

#[pyfunction]
#[pyo3(signature = (graph, source, max_depth=10))]
fn bfs_traversal(graph: &PyGraph, source: &str, max_depth: usize) -> (Vec<String>, Vec<PyEdge>) {
    let result = lq::bfs_traversal(&graph.inner, source, max_depth);
    (
        result.nodes,
        result
            .edges
            .into_iter()
            .map(|e| PyEdge { inner: e })
            .collect(),
    )
}

#[pyfunction]
#[pyo3(signature = (graph, source, max_depth=10))]
fn dfs_traversal(graph: &PyGraph, source: &str, max_depth: usize) -> (Vec<String>, Vec<PyEdge>) {
    let result = lq::dfs(&graph.inner, source, max_depth);
    (
        result.nodes,
        result
            .edges
            .into_iter()
            .map(|e| PyEdge { inner: e })
            .collect(),
    )
}

#[pyfunction]
fn load_csv(path: &str) -> PyResult<PyGraph> {
    let graph =
        lq::load_csv(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    Ok(PyGraph { inner: graph })
}

#[pyfunction]
fn save_csv(graph: &PyGraph, path: &str) -> PyResult<()> {
    lq::save_csv(&graph.inner, path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

/// Walk FFN weights from a model directory. Returns a Graph of extracted edges.
///
/// Args:
///     model_path: Path to model directory (safetensors + tokenizer.json)
///     output_path: Where to save the result (.larql.json or .larql.bin)
///     layer: Optional single layer to walk (default: all)
///     top_k: Top-k tokens per feature (default: 5)
///     min_score: Minimum activation score (default: 0.02)
#[pyfunction]
#[pyo3(signature = (model_path, output_path=None, layer=None, top_k=5, min_score=0.02))]
fn weight_walk(
    model_path: &str,
    output_path: Option<&str>,
    layer: Option<usize>,
    top_k: usize,
    min_score: f32,
) -> PyResult<PyGraph> {
    let config = li::WalkConfig { top_k, min_score };
    let layers: Option<Vec<usize>> = layer.map(|l| vec![l]);

    let mut graph = lq::Graph::new();
    graph.metadata.insert(
        "model".to_string(),
        serde_json::Value::String(model_path.to_string()),
    );
    graph.metadata.insert(
        "method".to_string(),
        serde_json::Value::String("weight-walk".to_string()),
    );

    let mut callbacks = li::walker::weight_walker::SilentWalkCallbacks;

    li::walk_model(
        model_path,
        layers.as_deref(),
        &config,
        &mut graph,
        &mut callbacks,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    if let Some(path) = output_path {
        lq::save(&graph, path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    }

    Ok(PyGraph { inner: graph })
}

/// Walk attention OV circuits from a model. Returns a Graph of routing edges.
#[pyfunction]
#[pyo3(signature = (model_path, output_path=None, layer=None, top_k=3, min_score=0.0))]
fn attention_walk(
    model_path: &str,
    output_path: Option<&str>,
    layer: Option<usize>,
    top_k: usize,
    min_score: f32,
) -> PyResult<PyGraph> {
    let walker = li::AttentionWalker::load(model_path)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let config = li::WalkConfig { top_k, min_score };

    let mut graph = lq::Graph::new();
    graph.metadata.insert(
        "model".to_string(),
        serde_json::Value::String(model_path.to_string()),
    );
    graph.metadata.insert(
        "method".to_string(),
        serde_json::Value::String("attention-walk".to_string()),
    );

    let layers: Vec<usize> = match layer {
        Some(l) => vec![l],
        None => (0..walker.num_layers()).collect(),
    };

    let mut callbacks = li::walker::weight_walker::SilentWalkCallbacks;
    for &l in &layers {
        walker
            .walk_layer(l, &config, &mut graph, &mut callbacks)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    }

    if let Some(path) = output_path {
        lq::save(&graph, path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    }

    Ok(PyGraph { inner: graph })
}

// ── Vindex top-level functions ──

/// Load a vindex from a directory path.
///
/// Returns a Vindex object with gate vectors, embeddings, and tokenizer.
/// Supports numpy array access for gate vectors and embeddings.
///
/// Example:
///     vindex = larql.load_vindex("gemma3-4b.vindex")
///     embed = vindex.embed("France")
///     hits = vindex.entity_knn("France", layer=26, top_k=10)
#[pyfunction]
fn load_vindex(path: &str) -> PyResult<PyVindex> {
    PyVindex::open(path)
}

/// Create an LQL session connected to a vindex.
///
/// The session provides both LQL query execution and direct vindex access:
///     session = larql.session("gemma3-4b.vindex")
///     session.query("DESCRIBE 'France'")         # LQL queries
///     session.vindex.embed("France")              # numpy arrays
#[pyfunction]
fn create_session(py: Python<'_>, path: &str) -> PyResult<PySession> {
    PySession::create(py, path)
}

// ── Module ──

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Graph types (existing)
    m.add_class::<PyEdge>()?;
    m.add_class::<PyNode>()?;
    m.add_class::<PyGraph>()?;

    // Vindex types (new)
    m.add_class::<PyVindex>()?;
    m.add_class::<PyFeatureMeta>()?;
    m.add_class::<PyWalkHit>()?;
    m.add_class::<PyDescribeEdge>()?;
    m.add_class::<PyRelation>()?;
    m.add_class::<PySession>()?;

    // Graph functions (existing)
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(save, m)?)?;
    m.add_function(wrap_pyfunction!(shortest_path, m)?)?;
    m.add_function(wrap_pyfunction!(merge_graphs, m)?)?;
    m.add_function(wrap_pyfunction!(merge_graphs_with_strategy, m)?)?;
    m.add_function(wrap_pyfunction!(diff, m)?)?;
    m.add_function(wrap_pyfunction!(pagerank, m)?)?;
    m.add_function(wrap_pyfunction!(bfs_traversal, m)?)?;
    m.add_function(wrap_pyfunction!(dfs_traversal, m)?)?;
    m.add_function(wrap_pyfunction!(load_csv, m)?)?;
    m.add_function(wrap_pyfunction!(save_csv, m)?)?;
    m.add_function(wrap_pyfunction!(weight_walk, m)?)?;
    m.add_function(wrap_pyfunction!(attention_walk, m)?)?;

    // Vindex functions (new)
    m.add_function(wrap_pyfunction!(load_vindex, m)?)?;
    m.add_function(wrap_pyfunction!(create_session, m)?)?;

    Ok(())
}
