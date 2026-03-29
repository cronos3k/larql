use clap::{Parser, Subcommand};

mod commands;
mod formatting;
mod utils;

use commands::extraction::*;
use commands::query::*;
use commands::surreal::*;

#[derive(Parser)]
#[command(
    name = "larql",
    version,
    about = "LARQL knowledge graph extraction and querying"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    // ── Extraction ──
    /// Extract edges from FFN weights. Zero forward passes.
    WeightExtract(weight_walk_cmd::WeightWalkArgs),

    /// Extract routing edges from attention OV circuits. Zero forward passes.
    AttentionExtract(attention_walk_cmd::AttentionWalkArgs),

    /// Extract full vectors from model weights to NDJSON files.
    VectorExtract(vector_extract_cmd::VectorExtractArgs),

    /// Capture residual stream vectors for entities via forward passes.
    Residuals(residuals_cmd::ResidualsArgs),

    /// Run full forward pass and predict next token.
    Predict(predict_cmd::PredictArgs),

    /// BFS extraction from a model endpoint.
    Bfs(bfs_cmd::BfsArgs),

    // ── SurrealDB ──
    /// Load vectors into SurrealDB with HNSW indexes (small tables, HTTP).
    VectorLoad(vector_load_cmd::VectorLoadArgs),

    /// Import vectors into SurrealDB via batched `surreal import` (large tables).
    VectorImport(vector_import_cmd::VectorImportArgs),

    /// Export vectors to .surql files for manual import.
    VectorExportSurql(vector_export_surql_cmd::VectorExportSurqlArgs),

    // ── Query ──
    /// Query a graph for facts.
    Query(query_cmd::QueryArgs),

    /// Describe an entity (all edges).
    Describe(describe_cmd::DescribeArgs),

    /// Show graph statistics.
    Stats(stats_cmd::StatsArgs),

    /// Validate a graph file.
    Validate(validate_cmd::ValidateArgs),

    /// Merge multiple graph files.
    Merge(merge_cmd::MergeArgs),
}

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        // Extraction
        Commands::WeightExtract(args) => weight_walk_cmd::run(args),
        Commands::AttentionExtract(args) => attention_walk_cmd::run(args),
        Commands::VectorExtract(args) => vector_extract_cmd::run(args),
        Commands::Residuals(args) => residuals_cmd::run(args),
        Commands::Predict(args) => predict_cmd::run(args),
        Commands::Bfs(args) => bfs_cmd::run(args),
        // SurrealDB
        Commands::VectorLoad(args) => vector_load_cmd::run(args),
        Commands::VectorImport(args) => vector_import_cmd::run(args),
        Commands::VectorExportSurql(args) => vector_export_surql_cmd::run(args),
        // Query
        Commands::Query(args) => query_cmd::run(args),
        Commands::Describe(args) => describe_cmd::run(args),
        Commands::Stats(args) => stats_cmd::run(args),
        Commands::Validate(args) => validate_cmd::run(args),
        Commands::Merge(args) => merge_cmd::run(args),
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
