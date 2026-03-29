use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use crate::core::edge::{CompactEdge, Edge};
use crate::core::graph::Graph;

/// Append-only checkpoint log.
/// Each line is a compact edge JSON. Survives crashes.
pub struct CheckpointLog {
    file: File,
    path: std::path::PathBuf,
    count: usize,
}

impl CheckpointLog {
    /// Open or create a checkpoint log.
    pub fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new().create(true).append(true).open(&path)?;

        // Count existing lines
        let count = if path.exists() {
            BufReader::new(File::open(&path)?).lines().count()
        } else {
            0
        };

        Ok(Self { file, path, count })
    }

    /// Append an edge to the log.
    pub fn append(&mut self, edge: &Edge) -> std::io::Result<()> {
        let compact = CompactEdge::from(edge);
        let json = serde_json::to_string(&compact).map_err(std::io::Error::other)?;
        writeln!(self.file, "{json}")?;
        self.file.flush()?;
        self.count += 1;
        Ok(())
    }

    /// Replay the log into a graph.
    pub fn replay(&self) -> std::io::Result<Graph> {
        let mut graph = Graph::new();
        let file = File::open(&self.path)?;
        for line in BufReader::new(file).lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(compact) = serde_json::from_str::<CompactEdge>(&line) {
                graph.add_edge(Edge::from(compact));
            }
        }
        Ok(graph)
    }

    pub fn edge_count(&self) -> usize {
        self.count
    }
}
