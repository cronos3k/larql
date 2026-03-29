//! Graph engine demo — build, query, traverse, and serialize a knowledge graph.
//!
//! Run: cargo run --release -p larql-core --example graph_demo

use larql_core::*;

fn main() {
    println!("=== LARQL Graph Engine Demo ===\n");

    // ── Build a graph ──
    let mut graph = Graph::new();

    let edges = vec![
        ("France", "capital-of", "Paris", 0.89),
        ("France", "language-of", "French", 0.84),
        ("France", "continent", "Europe", 0.95),
        ("France", "currency", "Euro", 0.91),
        ("Germany", "capital-of", "Berlin", 0.81),
        ("Germany", "language-of", "German", 0.92),
        ("Germany", "continent", "Europe", 0.95),
        ("Germany", "currency", "Euro", 0.88),
        ("Japan", "capital-of", "Tokyo", 0.93),
        ("Japan", "continent", "Asia", 0.97),
        ("Mozart", "birthplace", "Salzburg", 0.98),
        ("Mozart", "nationality", "Austrian", 0.73),
        ("Mozart", "known-for", "composer", 0.91),
        ("Paris", "located-in", "France", 0.98),
        ("Berlin", "located-in", "Germany", 0.97),
        ("Tokyo", "located-in", "Japan", 0.96),
        ("Salzburg", "located-in", "Austria", 0.95),
    ];

    for (s, r, o, c) in &edges {
        graph.add_edge(
            Edge::new(*s, *r, *o)
                .with_confidence(*c)
                .with_source(SourceType::Parametric),
        );
    }

    println!("Built: {graph:?}");
    println!("  Entities: {}", graph.node_count());
    println!("  Relations: {:?}\n", graph.list_relations());

    // ── Select ──
    println!("--- Select ---");
    let capitals = graph.select("France", Some("capital-of"));
    println!("  France capital: {}", capitals[0].object);

    let all_france = graph.select("France", None);
    println!("  France has {} outgoing edges", all_france.len());

    // ── Describe ──
    println!("\n--- Describe ---");
    let desc = graph.describe("Mozart");
    println!("  Mozart:");
    println!("    Outgoing: {}", desc.outgoing.len());
    for e in &desc.outgoing {
        println!(
            "      --{}--> {} ({:.2})",
            e.relation, e.object, e.confidence
        );
    }
    println!("    Incoming: {}", desc.incoming.len());

    // ── Walk ──
    println!("\n--- Multi-hop Walk ---");
    if let Some((dest, path)) = graph.walk("Mozart", &["birthplace", "located-in", "continent"]) {
        print!("  Mozart");
        for e in &path {
            print!(" --[{}]--> {}", e.relation, e.object);
        }
        println!(" = {dest}");
    }

    // ── Search ──
    println!("\n--- Keyword Search ---");
    let results = graph.search("Europe", 5);
    println!("  'Europe': {} hits", results.len());
    for e in &results {
        println!("    {} --{}--> {}", e.subject, e.relation, e.object);
    }

    // ── Subgraph ──
    println!("\n--- Subgraph ---");
    let sub = graph.subgraph("France", 1);
    println!(
        "  France depth=1: {} edges, {} nodes",
        sub.edge_count(),
        sub.node_count()
    );

    // ── Shortest Path ──
    println!("\n--- Shortest Path ---");
    if let Some((cost, path)) = shortest_path(&graph, "Mozart", "Europe") {
        print!("  Mozart → Europe (cost={cost:.2}):");
        for e in &path {
            print!(" {} --[{}]-->", e.subject, e.relation);
        }
        println!(" Europe");
    }

    // ── Merge ──
    println!("\n--- Merge ---");
    let mut other = Graph::new();
    other.add_edge(Edge::new("Einstein", "birthplace", "Ulm").with_confidence(0.92));
    other.add_edge(Edge::new("Einstein", "known-for", "physics").with_confidence(0.96));
    let added = merge_graphs(&mut graph, &other);
    println!("  Merged: {added} new edges, total: {}", graph.edge_count());

    // ── Stats ──
    println!("\n--- Stats ---");
    let stats = graph.stats();
    println!("  Entities: {}", stats.entities);
    println!("  Edges: {}", stats.edges);
    println!("  Components: {}", stats.connected_components);
    println!("  Avg confidence: {:.3}", stats.avg_confidence);

    // ── Save / Load ──
    println!("\n--- Serialization ---");
    let tmp = std::env::temp_dir().join("graph_demo.larql.json");
    save(&graph, &tmp).unwrap();
    let size = std::fs::metadata(&tmp).unwrap().len();
    println!("  Saved: {} ({} bytes)", tmp.display(), size);

    let loaded = load(&tmp).unwrap();
    println!(
        "  Loaded: {} edges, {} nodes",
        loaded.edge_count(),
        loaded.node_count()
    );
    assert_eq!(graph.edge_count(), loaded.edge_count());
    println!("  Roundtrip: OK");

    // MsgPack
    let tmp_bin = std::env::temp_dir().join("graph_demo.larql.bin");
    save(&graph, &tmp_bin).unwrap();
    let size_bin = std::fs::metadata(&tmp_bin).unwrap().len();
    println!(
        "  MsgPack: {} bytes ({:.0}% smaller)",
        size_bin,
        (1.0 - size_bin as f64 / size as f64) * 100.0
    );

    std::fs::remove_file(&tmp).ok();
    std::fs::remove_file(&tmp_bin).ok();

    println!("\n=== Done ===");
}
