//! CSV I/O for knowledge graphs.
//!
//! Format: subject,relation,object,confidence,source

use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use crate::core::edge::Edge;
use crate::core::enums::SourceType;
use crate::core::graph::{Graph, GraphError};

/// Load a graph from CSV. Expected columns: subject,relation,object,confidence,source
pub fn load_csv(path: impl AsRef<Path>) -> Result<Graph, GraphError> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut graph = Graph::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || (i == 0 && trimmed.starts_with("subject")) {
            continue; // skip empty lines and header
        }

        let fields: Vec<&str> = trimmed.splitn(5, ',').collect();
        if fields.len() < 3 {
            continue;
        }

        let subject = fields[0].trim();
        let relation = fields[1].trim();
        let object = fields[2].trim();
        let confidence: f64 = fields
            .get(3)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(1.0);
        let source = fields
            .get(4)
            .map(|s| parse_source(s.trim()))
            .unwrap_or(SourceType::Unknown);

        graph.add_edge(
            Edge::new(subject, relation, object)
                .with_confidence(confidence)
                .with_source(source),
        );
    }

    Ok(graph)
}

/// Save a graph to CSV.
pub fn save_csv(graph: &Graph, path: impl AsRef<Path>) -> Result<(), GraphError> {
    let mut file = std::fs::File::create(path)?;
    writeln!(file, "subject,relation,object,confidence,source")?;
    for edge in graph.edges() {
        writeln!(
            file,
            "{},{},{},{},{}",
            edge.subject,
            edge.relation,
            edge.object,
            edge.confidence,
            edge.source.as_str()
        )?;
    }
    Ok(())
}

fn parse_source(s: &str) -> SourceType {
    match s {
        "parametric" => SourceType::Parametric,
        "document" => SourceType::Document,
        "installed" => SourceType::Installed,
        "wikidata" => SourceType::Wikidata,
        "manual" => SourceType::Manual,
        _ => SourceType::Unknown,
    }
}
