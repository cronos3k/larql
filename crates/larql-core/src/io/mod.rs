pub mod checkpoint;
pub mod csv;
pub mod format;
pub mod json;
pub mod msgpack;

use std::path::Path;

use crate::core::graph::{Graph, GraphError};
pub use format::Format;

/// Load a graph from disk, auto-detecting format from the file extension.
pub fn load(path: impl AsRef<Path>) -> Result<Graph, GraphError> {
    let path = path.as_ref();
    let fmt = Format::from_path(path).ok_or_else(|| {
        GraphError::Deserialize(format!("unrecognised file extension: {}", path.display()))
    })?;
    load_with_format(path, fmt)
}

/// Load a graph using an explicit format.
pub fn load_with_format(path: impl AsRef<Path>, fmt: Format) -> Result<Graph, GraphError> {
    match fmt {
        Format::Json => json::load_json(path),
        #[cfg(feature = "msgpack")]
        Format::MessagePack => msgpack::load_msgpack(path),
    }
}

/// Save a graph to disk, auto-detecting format from the file extension.
pub fn save(graph: &Graph, path: impl AsRef<Path>) -> Result<(), GraphError> {
    let path = path.as_ref();
    let fmt = Format::from_path(path).ok_or_else(|| {
        GraphError::Deserialize(format!("unrecognised file extension: {}", path.display()))
    })?;
    save_with_format(graph, path, fmt)
}

/// Save a graph using an explicit format.
pub fn save_with_format(
    graph: &Graph,
    path: impl AsRef<Path>,
    fmt: Format,
) -> Result<(), GraphError> {
    match fmt {
        Format::Json => json::save_json(graph, path),
        #[cfg(feature = "msgpack")]
        Format::MessagePack => msgpack::save_msgpack(graph, path),
    }
}

/// Serialize a graph to bytes in the given format.
pub fn to_bytes(graph: &Graph, fmt: Format) -> Result<Vec<u8>, GraphError> {
    match fmt {
        Format::Json => json::to_json_bytes(graph),
        #[cfg(feature = "msgpack")]
        Format::MessagePack => msgpack::to_msgpack_bytes(graph),
    }
}

/// Deserialize a graph from bytes in the given format.
pub fn from_bytes(bytes: &[u8], fmt: Format) -> Result<Graph, GraphError> {
    match fmt {
        Format::Json => json::from_json_bytes(bytes),
        #[cfg(feature = "msgpack")]
        Format::MessagePack => msgpack::from_msgpack_bytes(bytes),
    }
}
