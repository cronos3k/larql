//! Benchmark SurrealDB SQL generation — no server required.
//!
//! Measures throughput of SQL generation for schema setup, single inserts,
//! and batch inserts at various sizes.
//!
//! Run: cargo run --release -p larql-surreal --example bench_sql

use std::time::Instant;

use larql_models::{TopKEntry, VectorRecord};
use larql_surreal::*;

fn bench<F: FnMut()>(name: &str, iters: usize, mut f: F) {
    f();
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = start.elapsed();
    let per_iter_us = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;
    if per_iter_us > 1000.0 {
        println!(
            "  {:<40} {:>8.2} ms  ({} iters)",
            name,
            per_iter_us / 1000.0,
            iters
        );
    } else {
        println!("  {:<40} {:>8.1} us  ({} iters)", name, per_iter_us, iters);
    }
}

fn make_record(id: &str, dim: usize) -> VectorRecord {
    VectorRecord {
        id: id.to_string(),
        layer: 26,
        feature: 1234,
        vector: vec![0.01; dim],
        dim,
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
        ],
    }
}

fn main() {
    println!("=== larql-surreal: SQL Generation Benchmark ===\n");

    // ── Schema generation ──
    println!("--- Schema ---\n");

    bench("setup_sql", 10_000, || {
        let _ = setup_sql("larql", "gemma3_4b");
    });

    bench("schema_sql (dim=2560)", 10_000, || {
        let _ = schema_sql("ffn_down", 2560);
    });

    bench("progress_table_sql", 10_000, || {
        let _ = progress_table_sql();
    });

    // ── Insert generation ──
    println!("\n--- Single Insert (varying dim) ---\n");

    for dim in [128, 512, 2560] {
        let record = make_record("L26_F1234", dim);
        let label = format!("single_insert_sql (dim={})", dim);
        bench(&label, 10_000, || {
            let _ = single_insert_sql("ffn_down", &record);
        });
    }

    // ── Batch insert generation ──
    println!("\n--- Batch Insert (dim=2560, varying batch) ---\n");

    for batch_size in [1, 10, 100, 500] {
        let records: Vec<VectorRecord> = (0..batch_size)
            .map(|i| make_record(&format!("L26_F{i}"), 2560))
            .collect();
        let label = format!("batch_insert_sql (n={})", batch_size);
        let iters = if batch_size > 100 { 10 } else { 100 };
        bench(&label, iters, || {
            let _ = batch_insert_sql("ffn_down", &records);
        });
    }

    // ── Progress tracking ──
    println!("\n--- Progress ---\n");

    bench("mark_layer_done_sql", 10_000, || {
        let _ = mark_layer_done_sql("ffn_down", 26, 10240);
    });

    bench("completed_layers_sql", 10_000, || {
        let _ = completed_layers_sql("ffn_down");
    });

    // ── Throughput ──
    println!("\n--- Throughput (dim=2560) ---\n");
    let record = make_record("L26_F0", 2560);
    let start = Instant::now();
    let n = 10_000;
    for _ in 0..n {
        let _ = single_insert_sql("ffn_down", &record);
    }
    let elapsed = start.elapsed();
    let sql = single_insert_sql("ffn_down", &record);
    let bytes_per_sec = (sql.len() * n) as f64 / elapsed.as_secs_f64();
    println!(
        "  {:.0} inserts/sec, {:.1} MB/sec SQL output",
        n as f64 / elapsed.as_secs_f64(),
        bytes_per_sec / 1024.0 / 1024.0
    );
    println!("  Single insert SQL size: {} bytes", sql.len());
}
