//! Serialization demo — JSON vs MessagePack, format detection, bytes API.
//!
//! Run: cargo run --release -p larql-core --example serialization_demo

use larql_core::*;

fn main() {
    println!("=== LARQL Serialization Demo ===\n");

    // Build a sample graph
    let mut graph = Graph::new();
    for i in 0..1000 {
        graph.add_edge(
            Edge::new(
                format!("Entity_{}", i / 10),
                format!("rel_{}", i % 5),
                format!("Target_{i}"),
            )
            .with_confidence(0.5 + (i as f64 % 50.0) / 100.0)
            .with_source(SourceType::Parametric),
        );
    }
    println!(
        "Graph: {} edges, {} nodes\n",
        graph.edge_count(),
        graph.node_count()
    );

    // ── JSON ──
    let json_bytes = to_bytes(&graph, Format::Json).unwrap();
    println!(
        "JSON:     {} bytes ({:.1} KB)",
        json_bytes.len(),
        json_bytes.len() as f64 / 1024.0
    );

    // ── MessagePack ──
    let msgpack_bytes = to_bytes(&graph, Format::MessagePack).unwrap();
    println!(
        "MsgPack:  {} bytes ({:.1} KB)",
        msgpack_bytes.len(),
        msgpack_bytes.len() as f64 / 1024.0
    );
    println!(
        "Savings:  {:.0}% smaller\n",
        (1.0 - msgpack_bytes.len() as f64 / json_bytes.len() as f64) * 100.0
    );

    // ── Roundtrip bytes ──
    let from_json = from_bytes(&json_bytes, Format::Json).unwrap();
    let from_msgpack = from_bytes(&msgpack_bytes, Format::MessagePack).unwrap();
    println!("Roundtrip JSON:    {} edges", from_json.edge_count());
    println!("Roundtrip MsgPack: {} edges", from_msgpack.edge_count());

    // ── File format detection ──
    println!("\nFormat detection:");
    for path in &[
        "graph.larql.json",
        "graph.json",
        "graph.larql.bin",
        "graph.bin",
        "graph.msgpack",
    ] {
        let fmt = Format::from_path(path);
        println!("  {path:25} → {fmt:?}");
    }

    // ── File save/load ──
    let tmp_json = std::env::temp_dir().join("demo.larql.json");
    let tmp_bin = std::env::temp_dir().join("demo.larql.bin");

    save(&graph, &tmp_json).unwrap();
    save(&graph, &tmp_bin).unwrap();

    let json_size = std::fs::metadata(&tmp_json).unwrap().len();
    let bin_size = std::fs::metadata(&tmp_bin).unwrap().len();
    println!("\nFile sizes:");
    println!("  JSON:    {json_size} bytes");
    println!("  MsgPack: {bin_size} bytes");

    // Auto-detect on load
    let g1 = load(&tmp_json).unwrap();
    let g2 = load(&tmp_bin).unwrap();
    println!("\nAuto-detect load:");
    println!("  .larql.json → {} edges", g1.edge_count());
    println!("  .larql.bin  → {} edges", g2.edge_count());

    std::fs::remove_file(&tmp_json).ok();
    std::fs::remove_file(&tmp_bin).ok();

    // ── Cross-format ──
    let cross = from_bytes(
        &to_bytes(
            &from_bytes(&json_bytes, Format::Json).unwrap(),
            Format::MessagePack,
        )
        .unwrap(),
        Format::MessagePack,
    )
    .unwrap();
    println!(
        "\nCross-format (JSON → MsgPack → Graph): {} edges",
        cross.edge_count()
    );

    println!("\n=== Done ===");
}
