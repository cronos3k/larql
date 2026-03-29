use larql_core::algo::shortest_path::default_weight;
use larql_core::*;

fn geo_graph() -> Graph {
    let mut g = Graph::new();
    let edges = vec![
        ("France", "capital-of", "Paris", 0.89),
        ("Germany", "capital-of", "Berlin", 0.81),
        ("Japan", "capital-of", "Tokyo", 0.93),
        ("Paris", "located-in", "France", 0.98),
        ("Berlin", "located-in", "Germany", 0.97),
        ("Tokyo", "located-in", "Japan", 0.96),
        ("France", "continent", "Europe", 0.95),
        ("Germany", "continent", "Europe", 0.95),
        ("Japan", "continent", "Asia", 0.97),
        ("Mozart", "birthplace", "Salzburg", 0.98),
        ("Salzburg", "located-in", "Austria", 0.95),
        ("Austria", "continent", "Europe", 0.94),
    ];
    for (s, r, o, c) in edges {
        g.add_edge(
            Edge::new(s, r, o)
                .with_confidence(c)
                .with_source(SourceType::Parametric),
        );
    }
    g
}

// ── Diff ──

#[test]
fn test_diff_identical() {
    let g = geo_graph();
    let d = diff(&g, &g);
    assert!(d.added.is_empty());
    assert!(d.removed.is_empty());
    assert!(d.changed.is_empty());
}

#[test]
fn test_diff_added() {
    let old = Graph::new();
    let new = geo_graph();
    let d = diff(&old, &new);
    assert_eq!(d.added.len(), new.edge_count());
    assert!(d.removed.is_empty());
}

#[test]
fn test_diff_removed() {
    let old = geo_graph();
    let new = Graph::new();
    let d = diff(&old, &new);
    assert!(d.added.is_empty());
    assert_eq!(d.removed.len(), old.edge_count());
}

#[test]
fn test_diff_changed_confidence() {
    let mut old = Graph::new();
    old.add_edge(Edge::new("France", "capital-of", "Paris").with_confidence(0.5));

    let mut new = Graph::new();
    new.add_edge(Edge::new("France", "capital-of", "Paris").with_confidence(0.9));

    let d = diff(&old, &new);
    assert!(d.added.is_empty());
    assert!(d.removed.is_empty());
    assert_eq!(d.changed.len(), 1);
    assert!((d.changed[0].old.confidence - 0.5).abs() < 0.01);
    assert!((d.changed[0].new.confidence - 0.9).abs() < 0.01);
}

// ── Merge strategies ──

#[test]
fn test_merge_union() {
    let mut target = Graph::new();
    target.add_edge(Edge::new("A", "r", "B").with_confidence(0.5));

    let mut other = Graph::new();
    other.add_edge(Edge::new("A", "r", "B").with_confidence(0.9)); // dup
    other.add_edge(Edge::new("C", "r", "D").with_confidence(0.8)); // new

    let added = merge_graphs_with_strategy(&mut target, &other, MergeStrategy::Union);
    assert_eq!(added, 1); // only C→D added
                          // Original confidence kept
    assert!((target.select("A", Some("r"))[0].confidence - 0.5).abs() < 0.01);
}

#[test]
fn test_merge_max_confidence() {
    let mut target = Graph::new();
    target.add_edge(Edge::new("A", "r", "B").with_confidence(0.5));

    let mut other = Graph::new();
    other.add_edge(Edge::new("A", "r", "B").with_confidence(0.9));

    let updated = merge_graphs_with_strategy(&mut target, &other, MergeStrategy::MaxConfidence);
    assert_eq!(updated, 1);
    assert!((target.select("A", Some("r"))[0].confidence - 0.9).abs() < 0.01);
}

#[test]
fn test_merge_max_confidence_keeps_higher() {
    let mut target = Graph::new();
    target.add_edge(Edge::new("A", "r", "B").with_confidence(0.9));

    let mut other = Graph::new();
    other.add_edge(Edge::new("A", "r", "B").with_confidence(0.5));

    let updated = merge_graphs_with_strategy(&mut target, &other, MergeStrategy::MaxConfidence);
    assert_eq!(updated, 0); // target was already higher
}

#[test]
fn test_merge_source_priority() {
    let mut target = Graph::new();
    target.add_edge(Edge::new("A", "r", "B").with_source(SourceType::Parametric));

    let mut other = Graph::new();
    other.add_edge(Edge::new("A", "r", "B").with_source(SourceType::Wikidata));

    let updated = merge_graphs_with_strategy(&mut target, &other, MergeStrategy::SourcePriority);
    assert_eq!(updated, 1); // Wikidata > Parametric
    assert_eq!(
        target.select("A", Some("r"))[0].source,
        SourceType::Wikidata
    );
}

// ── PageRank ──

#[test]
fn test_pagerank_basic() {
    let g = geo_graph();
    let result = pagerank(&g, 0.85, 100, 1e-6);

    assert!(result.converged);
    assert!(result.iterations < 100);
    assert_eq!(result.ranks.len(), g.node_count());

    // Europe should have high rank (many incoming edges)
    let top5 = result.top_k(5);
    assert!(top5.iter().any(|(name, _)| *name == "Europe"));
}

#[test]
fn test_pagerank_empty() {
    let g = Graph::new();
    let result = pagerank(&g, 0.85, 100, 1e-6);
    assert!(result.converged);
    assert!(result.ranks.is_empty());
}

#[test]
fn test_pagerank_single_edge() {
    let mut g = Graph::new();
    g.add_edge(Edge::new("A", "r", "B"));
    let result = pagerank(&g, 0.85, 100, 1e-6);
    assert!(result.ranks["B"] > result.ranks["A"]);
}

// ── BFS/DFS Traversal ──

#[test]
fn test_bfs_traversal() {
    let g = geo_graph();
    let result = bfs_traversal(&g, "France", 2);

    assert!(!result.nodes.is_empty());
    assert_eq!(result.nodes[0], "France");
    assert_eq!(*result.depths.get("France").unwrap(), 0);
    assert!(result.depths.contains_key("Paris"));
    assert!(result.depths.contains_key("Europe"));
}

#[test]
fn test_bfs_depth_limit() {
    let g = geo_graph();
    let depth0 = bfs_traversal(&g, "France", 0);
    let depth1 = bfs_traversal(&g, "France", 1);

    assert_eq!(depth0.nodes.len(), 1); // just France
    assert!(depth1.nodes.len() > 1);
    assert!(depth1.max_depth <= 1);
}

#[test]
fn test_dfs_traversal() {
    let g = geo_graph();
    let result = dfs(&g, "France", 3);

    assert!(!result.nodes.is_empty());
    assert_eq!(result.nodes[0], "France");
    assert!(result.nodes.len() > 1);
}

#[test]
fn test_bfs_unknown_entity() {
    let g = geo_graph();
    let result = bfs_traversal(&g, "Narnia", 5);
    assert_eq!(result.nodes.len(), 1); // just Narnia itself
}

// ── A* Search ──

#[test]
fn test_astar_finds_path() {
    let g = geo_graph();
    let result = astar(&g, "Mozart", "Europe", default_weight, |_, _| 0.0);

    assert!(result.found);
    assert!(result.cost < f64::INFINITY);
    assert!(!result.path.is_empty());
}

#[test]
fn test_astar_no_path() {
    let g = geo_graph();
    let result = astar(&g, "France", "Asia", default_weight, |_, _| 0.0);

    // France → Europe but no Europe → Asia edge
    assert!(!result.found || result.path.is_empty());
}

#[test]
fn test_astar_with_heuristic() {
    let g = geo_graph();
    // Heuristic that slightly guides toward the target (always admissible at 0)
    let result = astar(&g, "Mozart", "Europe", default_weight, |_, _| 0.0);
    let dijkstra = shortest_path(&g, "Mozart", "Europe").unwrap();

    // A* with zero heuristic should find same cost as Dijkstra
    assert!((result.cost - dijkstra.0).abs() < 0.001);
}

#[test]
fn test_path_result_nodes_explored() {
    let g = geo_graph();
    let result = astar(&g, "Mozart", "Europe", default_weight, |_, _| 0.0);
    assert!(result.nodes_explored > 0);
}

// ── CSV I/O ──

#[test]
fn test_csv_roundtrip() {
    let g = geo_graph();
    let path = std::env::temp_dir().join("test_csv_roundtrip.csv");

    save_csv(&g, &path).unwrap();
    let loaded = load_csv(&path).unwrap();

    assert_eq!(loaded.edge_count(), g.edge_count());
    for edge in g.edges() {
        assert!(loaded.exists(&edge.subject, &edge.relation, &edge.object));
    }

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_csv_preserves_confidence() {
    let mut g = Graph::new();
    g.add_edge(
        Edge::new("A", "r", "B")
            .with_confidence(0.75)
            .with_source(SourceType::Parametric),
    );

    let path = std::env::temp_dir().join("test_csv_conf.csv");
    save_csv(&g, &path).unwrap();
    let loaded = load_csv(&path).unwrap();

    let edge = &loaded.edges()[0];
    assert!((edge.confidence - 0.75).abs() < 0.001);
    assert_eq!(edge.source, SourceType::Parametric);

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_csv_format() {
    let mut g = Graph::new();
    g.add_edge(Edge::new("France", "capital-of", "Paris").with_confidence(0.89));

    let path = std::env::temp_dir().join("test_csv_format.csv");
    save_csv(&g, &path).unwrap();

    let contents = std::fs::read_to_string(&path).unwrap();
    assert!(contents.starts_with("subject,relation,object,confidence,source"));
    assert!(contents.contains("France,capital-of,Paris,0.89"));

    std::fs::remove_file(&path).ok();
}
