//! Python bindings for LQL sessions.
//!
//! Wraps larql_lql::Session to provide LQL query execution from Python.
//! Two interfaces, one session:
//! - session.query("DESCRIBE 'France'") — LQL string queries
//! - session.vindex — direct PyVindex access for numpy arrays

use pyo3::prelude::*;

use larql_lql::{parse, Session};
use crate::vindex::PyVindex;

// ── PySession ──

#[pyclass(name = "Session", unsendable)]
pub struct PySession {
    session: Session,
    vindex_obj: Option<Py<PyVindex>>,
    path: String,
}

impl PySession {
    /// Create a session (Rust-callable).
    pub fn create(py: Python<'_>, path: &str) -> PyResult<Self> {
        let mut session = Session::new();

        // Execute USE to connect the LQL session to the vindex
        let use_stmt = format!("USE \"{}\";", path.replace('"', "\\\""));
        let stmt = parse(&use_stmt)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Parse error: {e}")))?;
        session.execute(&stmt)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("USE failed: {e}")))?;

        // Also load a PyVindex for direct array access
        let vindex = PyVindex::open(path)?;
        let vindex_obj = Py::new(py, vindex)?;

        Ok(Self {
            session,
            vindex_obj: Some(vindex_obj),
            path: path.to_string(),
        })
    }
}

#[pymethods]
impl PySession {
    /// Create a session connected to a vindex.
    #[new]
    fn new(py: Python<'_>, path: &str) -> PyResult<Self> {
        Self::create(py, path)
    }

    /// Execute an LQL query string. Returns list of output lines.
    ///
    /// Examples:
    ///   session.query("DESCRIBE 'France'")
    ///   session.query("WALK 'The capital of France is' TOP 10")
    ///   session.query("STATS")
    ///   session.query("SELECT entity, target FROM EDGES WHERE relation = 'capital' LIMIT 10")
    fn query(&mut self, lql: &str) -> PyResult<Vec<String>> {
        // Add semicolon if missing
        let input = if lql.trim_end().ends_with(';') {
            lql.to_string()
        } else {
            format!("{};", lql)
        };

        let stmt = parse(&input)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Parse error: {e}")))?;

        self.session.execute(&stmt)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Execution error: {e}")))
    }

    /// Execute an LQL query and return results as a single string.
    fn query_text(&mut self, lql: &str) -> PyResult<String> {
        let lines = self.query(lql)?;
        Ok(lines.join("\n"))
    }

    /// Access the underlying Vindex for direct numpy operations.
    #[getter]
    fn vindex(&self, py: Python<'_>) -> PyResult<Py<PyVindex>> {
        self.vindex_obj.as_ref()
            .map(|v| v.clone_ref(py))
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No vindex loaded"))
    }

    /// The vindex path this session is connected to.
    #[getter]
    fn path(&self) -> &str {
        &self.path
    }

    fn __repr__(&self) -> String {
        format!("Session(path='{}')", self.path)
    }
}
