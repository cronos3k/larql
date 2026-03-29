pub mod client;
pub mod error;
pub mod loader;

pub use client::{base64_encode, SurrealClient};
pub use error::SurrealError;
pub use loader::{
    // SQL generators
    batch_insert_sql,
    completed_layers_sql,
    count_sql,
    discover_vector_files,
    mark_layer_done_sql,
    progress_table_sql,
    schema_sql,
    setup_sql,
    single_insert_sql,
    LoadCallbacks,
    LoadConfig,
    LoadSummary,
    SilentLoadCallbacks,
    TableSummary,
    VectorReader,
};
