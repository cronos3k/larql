//! Algorithm demo — shortest path, merge, subgraph, connected components.
//!
//! Run: cargo run --release -p larql-core --example algorithm_demo

use larql_core::*;

fn main() {
    println!("=== LARQL Algorithm Demo ===\n");

    let mut graph = Graph::new();

    // Build a geography graph
    let facts = vec![
        ("France", "capital-of", "Paris", 0.89),
        ("Germany", "capital-of", "Berlin", 0.81),
        ("Japan", "capital-of", "Tokyo", 0.93),
        ("Paris", "located-in", "France", 0.98),
        ("Berlin", "located-in", "Germany", 0.97),
        ("Tokyo", "located-in", "Japan", 0.96),
        ("France", "continent", "Europe", 0.95),
        ("Germany", "continent", "Europe", 0.95),
        ("Japan", "continent", "Asia", 0.97),
        // Person graph (separate component)
        ("Mozart", "birthplace", "Salzburg", 0.98),
        ("Salzburg", "located-in", "Austria", 0.95),
        ("Austria", "continent", "Europe", 0.94),
    ];

    for (s, r, o, c) in &facts {
        graph.add_edge(Edge::new(*s, *r, *o).with_confidence(*c));
    }
    println!(
        "Graph: {} edges, {} nodes\n",
        graph.edge_count(),
        graph.node_count()
    );

    // ── Shortest Path ──
    println!("--- Shortest Path (weight = 1 - confidence) ---");

    let paths = vec![
        ("Mozart", "Europe"),
        ("France", "Asia"),
        ("Paris", "Berlin"),
        ("Tokyo", "Europe"),
    ];

    for (from, to) in &paths {
        match shortest_path(&graph, from, to) {
            Some((cost, edges)) => {
                print!("  {from}");
                for e in &edges {
                    print!(" → [{}] {}", e.relation, e.object);
                }
                println!("  (cost={cost:.3}, hops={})", edges.len());
            }
            None => println!("  {from} → {to}: no path"),
        }
    }

    // ── Subgraph ──
    println!("\n--- Subgraph Extraction ---");
    for depth in 0..=3 {
        let sub = graph.subgraph("France", depth);
        println!(
            "  France depth={depth}: {} edges, {} nodes",
            sub.edge_count(),
            sub.node_count()
        );
    }

    // ── Connected Components ──
    println!("\n--- Connected Components ---");
    let stats = graph.stats();
    println!("  Components: {}", stats.connected_components);
    println!("  (France/Germany/Japan/Mozart all connected via Europe)");

    // ── Merge ──
    println!("\n--- Graph Merge ---");
    let mut science = Graph::new();
    science.add_edge(Edge::new("Einstein", "birthplace", "Ulm").with_confidence(0.92));
    science.add_edge(Edge::new("Einstein", "known-for", "relativity").with_confidence(0.96));
    science.add_edge(Edge::new("Ulm", "located-in", "Germany").with_confidence(0.94));

    println!("  Before: {} edges", graph.edge_count());
    let added = merge_graphs(&mut graph, &science);
    println!("  Merged: {added} new edges");
    println!(
        "  After:  {} edges, {} nodes",
        graph.edge_count(),
        graph.node_count()
    );

    // Now Einstein is connected to the main graph via Germany
    let new_stats = graph.stats();
    println!(
        "  Components: {} (was {})",
        new_stats.connected_components, stats.connected_components
    );

    // Einstein → Europe path should work now
    if let Some((cost, path)) = shortest_path(&graph, "Einstein", "Europe") {
        print!("  Einstein → Europe:");
        for e in &path {
            print!(" [{}] {}", e.relation, e.object);
        }
        println!(" (cost={cost:.3})");
    }

    // ── Walk ──
    println!("\n--- Multi-hop Walk ---");
    let walks = vec![
        ("France", vec!["capital-of", "located-in"]),
        ("Mozart", vec!["birthplace", "located-in", "continent"]),
        ("Einstein", vec!["birthplace", "located-in", "continent"]),
    ];

    for (start, rels) in &walks {
        match graph.walk(start, &rels.iter().map(|s| *s).collect::<Vec<_>>()) {
            Some((dest, _)) => println!("  {start} → [{}] → {dest}", rels.join(" → ")),
            None => println!("  {start} → [{}] → DEAD END", rels.join(" → ")),
        }
    }

    println!("\n=== Done ===");
}
