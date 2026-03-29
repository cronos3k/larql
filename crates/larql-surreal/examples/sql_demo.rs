//! Demonstrate SurrealDB SQL generation — no server required.
//!
//! Shows the SQL that larql-surreal generates for schema setup,
//! vector loading, and progress tracking.
//!
//! Run: cargo run -p larql-surreal --example sql_demo

use larql_models::{TopKEntry, VectorRecord};
use larql_surreal::*;

fn main() {
    println!("=== larql-surreal: SQL Generation Demo ===\n");

    // ── Namespace/database setup ──
    println!("--- Setup SQL ---");
    let sql = setup_sql("larql", "gemma3_4b");
    println!("{sql}\n");

    // ── Component schema ──
    println!("--- Schema SQL (ffn_down, dim=2560) ---");
    let sql = schema_sql("ffn_down", 2560).unwrap();
    println!("{sql}\n");

    // ── Progress table ──
    println!("--- Progress Table SQL ---");
    let sql = progress_table_sql();
    println!("{sql}\n");

    // ── Single record insert ──
    let record = VectorRecord {
        id: "L26_F1234".to_string(),
        layer: 26,
        feature: 1234,
        vector: vec![0.1, -0.2, 0.3, 0.4], // truncated for demo
        dim: 4,
        top_token: "France".to_string(),
        top_token_id: 7001,
        c_score: 0.85,
        top_k: vec![
            TopKEntry {
                token: "France".to_string(),
                token_id: 7001,
                logit: 12.5,
            },
            TopKEntry {
                token: "Germany".to_string(),
                token_id: 9405,
                logit: 8.3,
            },
            TopKEntry {
                token: "Spain".to_string(),
                token_id: 8700,
                logit: 7.1,
            },
        ],
    };

    println!("--- Single Insert SQL ---");
    let sql = single_insert_sql("ffn_down", &record);
    println!("{sql}\n");

    // ── Batch insert ──
    let records = vec![
        VectorRecord {
            id: "L26_F0".to_string(),
            layer: 26,
            feature: 0,
            vector: vec![0.5, 0.6],
            dim: 2,
            top_token: "the".to_string(),
            top_token_id: 1,
            c_score: 0.12,
            top_k: vec![],
        },
        record,
    ];

    println!("--- Batch Insert SQL (2 records) ---");
    let sql = batch_insert_sql("ffn_down", &records);
    // Just show first/last lines to keep it readable
    let lines: Vec<&str> = sql.lines().collect();
    println!("{}", lines[0]); // BEGIN TRANSACTION
    println!("  ... {} CREATE statements ...", lines.len() - 2);
    println!("{}", lines[lines.len() - 1]); // COMMIT TRANSACTION
    println!();

    // ── Progress tracking ──
    println!("--- Mark Layer Done SQL ---");
    let sql = mark_layer_done_sql("ffn_down", 26, 10240);
    println!("{sql}\n");

    println!("--- Completed Layers Query ---");
    let sql = completed_layers_sql("ffn_down");
    println!("{sql}\n");

    println!("--- Count Query ---");
    let sql = count_sql("ffn_down");
    println!("{sql}\n");

    // ── Error handling ──
    println!("--- Invalid Component ---");
    match schema_sql("invalid_component", 100) {
        Ok(_) => println!("  (unexpected success)"),
        Err(e) => println!("  Error: {e}"),
    }

    // ── Base64 encoding ──
    println!("\n--- Base64 Auth ---");
    let encoded = base64_encode("root:root");
    println!("  root:root -> {encoded}");
}
