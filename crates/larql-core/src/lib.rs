pub mod algo;
pub mod core;
pub mod engine;
pub mod io;

// Re-export the essential types at crate root.
pub use core::edge::Edge;
pub use core::enums::{MergeStrategy, SourceType};
pub use core::graph::Graph;
pub use core::schema::Schema;

pub use engine::bfs::{extract_bfs, BfsCallbacks, BfsConfig, BfsResult};
pub use engine::chain::{chain_tokens, ChainResult};
pub use engine::provider::{ModelProvider, PredictionResult, TokenPrediction};
pub use engine::templates::TemplateRegistry;

pub use io::checkpoint::CheckpointLog;
pub use io::format::Format;
pub use io::json::{load_json, save_json};
pub use io::{from_bytes, load, load_with_format, save, save_with_format, to_bytes};

pub use algo::diff::{diff, ChangedEdge, GraphDiff};
pub use algo::filter::{filter_graph, FilterConfig};
pub use algo::merge::{merge_graphs, merge_graphs_with_strategy};
pub use algo::pagerank::{pagerank, PageRankResult};
pub use algo::shortest_path::{astar, shortest_path, shortest_path_with_weight, PathResult};
pub use algo::traversal::{bfs as bfs_traversal, dfs, TraversalResult};
pub use algo::components::{connected_components, are_connected};
pub use algo::walk::{walk_all_paths, WalkResult};
pub use io::csv::{load_csv, save_csv};
pub use io::packed::{from_packed_bytes, load_packed, save_packed, to_packed_bytes};
