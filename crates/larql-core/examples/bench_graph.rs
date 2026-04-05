//! Benchmark the graph engine — insert, query, search, algorithms, serialization.
//!
//! Run: cargo run --release -p larql-core --example bench_graph

use larql_core::*;
use std::time::Instant;

fn bench<F: FnMut()>(name: &str, iters: usize, mut f: F) {
    f(); // warmup
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = start.elapsed();
    let per_iter_us = elapsed.as_secs_f64() * 1_000_000.0 / iters as f64;
    let per_iter_ms = per_iter_us / 1000.0;
    if per_iter_ms > 1.0 {
        println!("  {:<40} {:>8.2} ms  ({} iters)", name, per_iter_ms, iters);
    } else {
        println!("  {:<40} {:>8.1} us  ({} iters)", name, per_iter_us, iters);
    }
}

fn build_graph(n: usize) -> Graph {
    let mut graph = Graph::new();
    for i in 0..n {
        let edge = Edge::new(
            format!("Entity_{}", i / 10),
            format!("rel_{}", i % 10),
            format!("Target_{}", i),
        )
        .with_confidence(0.5 + (i as f64 % 50.0) / 100.0);
        graph.add_edge(edge);
    }
    graph
}

fn main() {
    let n = 100_000usize;
    println!("=== larql-core: Graph Engine Benchmark ===\n");

    // ── Insert ──
    println!("--- Insert ---\n");
    let start = Instant::now();
    let graph = build_graph(n);
    let insert_time = start.elapsed();
    println!(
        "  {} edges in {:.1}ms ({:.0} ns/edge)",
        graph.edge_count(),
        insert_time.as_secs_f64() * 1000.0,
        insert_time.as_nanos() as f64 / n as f64
    );

    // ── Query ──
    println!("\n--- Query ---\n");
    bench("select(entity, None)", 100_000, || {
        let _ = graph.select("Entity_42", None);
    });

    bench("select(entity, Some(rel))", 100_000, || {
        let _ = graph.select("Entity_42", Some("rel_0"));
    });

    bench("exists(s, r, o)", 100_000, || {
        let _ = graph.exists("Entity_42", "rel_0", "Target_420");
    });

    bench("node(entity)", 100_000, || {
        let _ = graph.node("Entity_42");
    });

    bench("describe(entity)", 10_000, || {
        let _ = graph.describe("Entity_42");
    });

    bench("count(relation, None)", 100_000, || {
        let _ = graph.count(Some("rel_0"), None);
    });

    // ── Search ──
    println!("\n--- Search ---\n");
    bench("search(keyword, 10)", 1_000, || {
        let _ = graph.search("Entity_42", 10);
    });

    bench("search(keyword, 100)", 1_000, || {
        let _ = graph.search("Entity", 100);
    });

    // ── Subgraph ──
    println!("\n--- Subgraph ---\n");
    bench("subgraph(depth=1)", 1_000, || {
        let _ = graph.subgraph("Entity_0", 1);
    });

    bench("subgraph(depth=2)", 100, || {
        let _ = graph.subgraph("Entity_0", 2);
    });

    // ── Algorithms ──
    println!("\n--- Algorithms ---\n");

    // Build a smaller connected graph for algorithm benchmarks
    let mut algo_graph = Graph::new();
    for i in 0..1_000 {
        algo_graph.add_edge(
            Edge::new(
                format!("N{}", i),
                "connects",
                format!("N{}", (i + 1) % 1_000),
            )
            .with_confidence(0.9),
        );
        // Add some cross-links
        if i % 10 == 0 {
            algo_graph.add_edge(
                Edge::new(
                    format!("N{}", i),
                    "shortcut",
                    format!("N{}", (i + 100) % 1_000),
                )
                .with_confidence(0.8),
            );
        }
    }

    bench("shortest_path (1000-node ring)", 100, || {
        let _ = shortest_path(&algo_graph, "N0", "N500");
    });

    bench("pagerank (1000 nodes, 1100 edges)", 10, || {
        let _ = pagerank(&algo_graph, 0.85, 100, 1e-6);
    });

    bench("bfs_traversal (depth=5)", 100, || {
        let _ = bfs_traversal(&algo_graph, "N0", 5);
    });

    bench("dfs (depth=5)", 100, || {
        let _ = dfs(&algo_graph, "N0", 5);
    });

    bench("connected_components (1000 nodes)", 100, || {
        let _ = connected_components(&algo_graph);
    });

    bench("are_connected (1000 nodes)", 1_000, || {
        let _ = are_connected(&algo_graph, "N0", "N500");
    });

    bench("walk_all_paths (3 hops, max 10)", 100, || {
        let _ = walk_all_paths(&algo_graph, "N0", &["connects", "connects", "connects"], 10);
    });

    // ── Filter ──
    println!("\n--- Filter ---\n");
    {
        let config = FilterConfig {
            min_confidence: Some(0.85),
            ..Default::default()
        };
        bench("filter_graph (100K, confidence>0.85)", 3, || {
            let _ = filter_graph(&graph, &config);
        });
    }

    // ── Merge ──
    println!("\n--- Merge ---\n");
    // Merge: build fresh each time since Graph doesn't implement Clone
    bench("merge_graphs (10K into 10K)", 3, || {
        let mut target = build_graph(10_000);
        let other = build_graph(10_000);
        merge_graphs(&mut target, &other);
    });

    // ── Diff ──
    println!("\n--- Diff ---\n");
    let small_a = build_graph(1_000);
    let small_b = build_graph(1_200); // 200 extra edges
    bench("diff (1000 vs 1200 edges)", 100, || {
        let _ = diff(&small_a, &small_b);
    });

    // ── Serialization ──
    println!("\n--- Serialization (100K edges) ---\n");

    bench("JSON serialize", 3, || {
        let _ = to_bytes(&graph, Format::Json).unwrap();
    });

    let json_bytes = to_bytes(&graph, Format::Json).unwrap();
    bench("JSON deserialize", 3, || {
        let _ = from_bytes(&json_bytes, Format::Json).unwrap();
    });

    bench("MsgPack serialize", 3, || {
        let _ = to_bytes(&graph, Format::MessagePack).unwrap();
    });

    let msgpack_bytes = to_bytes(&graph, Format::MessagePack).unwrap();
    bench("MsgPack deserialize", 3, || {
        let _ = from_bytes(&msgpack_bytes, Format::MessagePack).unwrap();
    });

    bench("Packed binary serialize", 3, || {
        let _ = to_packed_bytes(&graph).unwrap();
    });

    let packed_bytes = to_packed_bytes(&graph).unwrap();
    bench("Packed binary deserialize", 3, || {
        let _ = from_packed_bytes(&packed_bytes).unwrap();
    });

    println!(
        "\n  Size: JSON {:.1} MB, MsgPack {:.1} MB ({:.0}%), Packed {:.1} MB ({:.0}%)",
        json_bytes.len() as f64 / 1024.0 / 1024.0,
        msgpack_bytes.len() as f64 / 1024.0 / 1024.0,
        (1.0 - msgpack_bytes.len() as f64 / json_bytes.len() as f64) * 100.0,
        packed_bytes.len() as f64 / 1024.0 / 1024.0,
        (1.0 - packed_bytes.len() as f64 / json_bytes.len() as f64) * 100.0,
    );

    // ── Stats ──
    println!("\n--- Stats ---\n");
    bench("stats() (100K edges)", 10, || {
        let _ = graph.stats();
    });

    println!(
        "\n  Graph: {} edges, {} entities",
        graph.edge_count(),
        graph.node_count()
    );
}
