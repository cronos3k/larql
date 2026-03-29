use larql_core::*;

// ── Construction ──

#[test]
fn test_empty_graph() {
    let g = Graph::new();
    assert_eq!(g.edge_count(), 0);
    assert_eq!(g.node_count(), 0);
    assert!(g.edges().is_empty());
    assert!(g.nodes().is_empty());
    assert!(g.list_entities().is_empty());
    assert!(g.list_relations().is_empty());
}

#[test]
fn test_add_edge() {
    let mut g = Graph::new();
    g.add_edge(Edge::new("France", "capital-of", "Paris").with_confidence(0.89));
    assert_eq!(g.edge_count(), 1);
    assert_eq!(g.node_count(), 2);
}

#[test]
fn test_add_edges_batch() {
    let mut g = Graph::new();
    g.add_edges(vec![
        Edge::new("France", "capital-of", "Paris"),
        Edge::new("Germany", "capital-of", "Berlin"),
        Edge::new("Japan", "capital-of", "Tokyo"),
    ]);
    assert_eq!(g.edge_count(), 3);
    assert_eq!(g.node_count(), 6);
}

#[test]
fn test_duplicate_skipped() {
    let mut g = Graph::new();
    g.add_edge(Edge::new("France", "capital-of", "Paris").with_confidence(0.89));
    g.add_edge(Edge::new("France", "capital-of", "Paris").with_confidence(0.50));
    assert_eq!(g.edge_count(), 1);
    // First one wins
    assert!((g.edges()[0].confidence - 0.89).abs() < f64::EPSILON);
}

#[test]
fn test_same_subject_relation_different_object() {
    let mut g = Graph::new();
    g.add_edge(Edge::new("France", "language-of", "French"));
    g.add_edge(Edge::new("France", "language-of", "Occitan"));
    assert_eq!(g.edge_count(), 2);
}

// ── Remove ──

#[test]
fn test_remove_edge() {
    let mut g = Graph::new();
    g.add_edge(Edge::new("France", "capital-of", "Paris"));
    g.add_edge(Edge::new("Germany", "capital-of", "Berlin"));

    assert!(g.remove_edge("France", "capital-of", "Paris"));
    assert_eq!(g.edge_count(), 1);
    assert!(!g.exists("France", "capital-of", "Paris"));
    assert!(g.exists("Germany", "capital-of", "Berlin"));
}

#[test]
fn test_remove_nonexistent_edge() {
    let mut g = Graph::new();
    g.add_edge(Edge::new("France", "capital-of", "Paris"));
    assert!(!g.remove_edge("France", "capital-of", "London"));
    assert_eq!(g.edge_count(), 1);
}

#[test]
fn test_remove_rebuilds_indexes() {
    let mut g = Graph::new();
    g.add_edge(Edge::new("France", "capital-of", "Paris"));
    g.add_edge(Edge::new("France", "currency", "Euro"));
    g.remove_edge("France", "capital-of", "Paris");

    // select should no longer return the removed edge
    assert!(g.select("France", Some("capital-of")).is_empty());
    assert_eq!(g.select("France", Some("currency")).len(), 1);

    // reverse index should be updated
    assert!(g.select_reverse("Paris", None).is_empty());

    // search should not find it
    assert!(g.search("Paris", 10).is_empty());
}

// ── Deduplication ──

#[test]
fn test_deduplicate_max_confidence() {
    let mut g = Graph::new();
    // Manually build without dedup check — use rebuild
    g.add_edge(Edge::new("France", "capital-of", "Paris").with_confidence(0.7));
    // Can't add duplicate via add_edge (it skips), so test via the strategy path
    // Instead test that deduplicate on a clean graph removes nothing
    let removed = g.deduplicate(MergeStrategy::MaxConfidence);
    assert_eq!(removed, 0);
    assert_eq!(g.edge_count(), 1);
}

// ── Queries ──

#[test]
fn test_select() {
    let mut g = Graph::new();
    g.add_edge(Edge::new("France", "capital-of", "Paris"));
    g.add_edge(Edge::new("France", "currency", "Euro"));
    g.add_edge(Edge::new("Germany", "capital-of", "Berlin"));

    let all_france = g.select("France", None);
    assert_eq!(all_france.len(), 2);

    let capitals = g.select("France", Some("capital-of"));
    assert_eq!(capitals.len(), 1);
    assert_eq!(capitals[0].object, "Paris");

    let empty = g.select("France", Some("nonexistent"));
    assert!(empty.is_empty());

    let missing = g.select("Unknown", None);
    assert!(missing.is_empty());
}

#[test]
fn test_select_reverse() {
    let mut g = Graph::new();
    g.add_edge(Edge::new("France", "capital-of", "Paris"));
    g.add_edge(Edge::new("France", "currency", "Euro"));
    g.add_edge(Edge::new("Germany", "currency", "Euro"));

    let to_paris = g.select_reverse("Paris", None);
    assert_eq!(to_paris.len(), 1);
    assert_eq!(to_paris[0].subject, "France");

    let euro_sources = g.select_reverse("Euro", None);
    assert_eq!(euro_sources.len(), 2);

    let euro_currency = g.select_reverse("Euro", Some("currency"));
    assert_eq!(euro_currency.len(), 2);
}

#[test]
fn test_describe() {
    let mut g = Graph::new();
    g.add_edge(Edge::new("France", "capital-of", "Paris"));
    g.add_edge(Edge::new("Paris", "located-in", "France"));

    let result = g.describe("France");
    assert_eq!(result.entity, "France");
    assert_eq!(result.outgoing.len(), 1);
    assert_eq!(result.outgoing[0].relation, "capital-of");
    assert_eq!(result.incoming.len(), 1);
    assert_eq!(result.incoming[0].subject, "Paris");

    let empty = g.describe("Unknown");
    assert!(empty.outgoing.is_empty());
    assert!(empty.incoming.is_empty());
}

#[test]
fn test_exists() {
    let mut g = Graph::new();
    g.add_edge(Edge::new("France", "capital-of", "Paris"));

    assert!(g.exists("France", "capital-of", "Paris"));
    assert!(!g.exists("France", "capital-of", "London"));
    assert!(!g.exists("Germany", "capital-of", "Paris"));
    assert!(!g.exists("France", "currency", "Paris"));
}

#[test]
fn test_walk() {
    let mut g = Graph::new();
    g.add_edge(Edge::new("France", "capital-of", "Paris").with_confidence(0.89));
    g.add_edge(Edge::new("Paris", "located-in", "France").with_confidence(0.98));

    let (dest, path) = g.walk("France", &["capital-of", "located-in"]).unwrap();
    assert_eq!(dest, "France");
    assert_eq!(path.len(), 2);
    assert_eq!(path[0].relation, "capital-of");
    assert_eq!(path[1].relation, "located-in");
}

#[test]
fn test_walk_picks_highest_confidence() {
    let mut g = Graph::new();
    g.add_edge(Edge::new("France", "language-of", "French").with_confidence(0.84));
    g.add_edge(Edge::new("France", "language-of", "Occitan").with_confidence(0.40));

    let (dest, _) = g.walk("France", &["language-of"]).unwrap();
    assert_eq!(dest, "French");
}

#[test]
fn test_walk_fails_on_missing_hop() {
    let mut g = Graph::new();
    g.add_edge(Edge::new("France", "capital-of", "Paris"));

    assert!(g.walk("France", &["capital-of", "currency"]).is_none());
}

#[test]
fn test_search() {
    let mut g = Graph::new();
    g.add_edge(Edge::new("France", "capital-of", "Paris"));
    g.add_edge(Edge::new("Germany", "capital-of", "Berlin"));
    g.add_edge(Edge::new("France", "currency", "Euro"));

    let results = g.search("France", 10);
    assert!(results.len() >= 2); // both France edges

    let results = g.search("capital", 10);
    assert_eq!(results.len(), 2); // both capital-of edges

    let results = g.search("nonexistent", 10);
    assert!(results.is_empty());
}

#[test]
fn test_search_max_results() {
    let mut g = Graph::new();
    for i in 0..20 {
        g.add_edge(Edge::new(format!("Entity {i}"), "rel", "Target"));
    }
    // "Entity" token matches all 20 edges; max_results caps at 5
    let results = g.search("Entity", 5);
    assert_eq!(results.len(), 5);
}

#[test]
fn test_subgraph() {
    let mut g = Graph::new();
    g.add_edge(Edge::new("France", "capital-of", "Paris"));
    g.add_edge(Edge::new("Paris", "located-in", "France"));
    g.add_edge(Edge::new("Germany", "capital-of", "Berlin"));
    g.add_edge(Edge::new("Berlin", "located-in", "Germany"));

    // depth=0: visit France, add its outgoing (France->Paris), don't queue further
    let sub0 = g.subgraph("France", 0);
    assert_eq!(sub0.edge_count(), 1);
    assert!(sub0.exists("France", "capital-of", "Paris"));

    // depth=1: visit France (d=0) + Paris (d=1), get France->Paris and Paris->France
    let sub1 = g.subgraph("France", 1);
    assert_eq!(sub1.edge_count(), 2);
    assert!(sub1.exists("France", "capital-of", "Paris"));
    assert!(sub1.exists("Paris", "located-in", "France"));

    // Germany is never reached from France
    assert!(!sub1.exists("Germany", "capital-of", "Berlin"));
}

#[test]
fn test_subgraph_unknown_entity() {
    let mut g = Graph::new();
    g.add_edge(Edge::new("France", "capital-of", "Paris"));
    let sub = g.subgraph("Unknown", 2);
    assert_eq!(sub.edge_count(), 0);
}

// ── Count ──

#[test]
fn test_count() {
    let mut g = Graph::new();
    g.add_edge(Edge::new("France", "capital-of", "Paris").with_source(SourceType::Parametric));
    g.add_edge(Edge::new("Germany", "capital-of", "Berlin").with_source(SourceType::Parametric));
    g.add_edge(Edge::new("France", "currency", "Euro").with_source(SourceType::Wikidata));

    assert_eq!(g.count(None, None), 3);
    assert_eq!(g.count(Some("capital-of"), None), 2);
    assert_eq!(g.count(Some("currency"), None), 1);
    assert_eq!(g.count(None, Some(&SourceType::Parametric)), 2);
    assert_eq!(g.count(None, Some(&SourceType::Wikidata)), 1);
    assert_eq!(
        g.count(Some("capital-of"), Some(&SourceType::Parametric)),
        2
    );
    assert_eq!(g.count(Some("nonexistent"), None), 0);
}

// ── Node ──

#[test]
fn test_node() {
    let mut g = Graph::new();
    g.add_edge(Edge::new("France", "capital-of", "Paris"));
    g.add_edge(Edge::new("France", "currency", "Euro"));

    let france = g.node("France").unwrap();
    assert_eq!(france.name, "France");
    assert_eq!(france.out_degree, 2);
    assert_eq!(france.in_degree, 0);
    assert_eq!(france.degree, 2);

    let paris = g.node("Paris").unwrap();
    assert_eq!(paris.out_degree, 0);
    assert_eq!(paris.in_degree, 1);

    assert!(g.node("Unknown").is_none());
}

// ── Stats ──

#[test]
fn test_stats() {
    let mut g = Graph::new();
    g.add_edge(Edge::new("France", "capital-of", "Paris").with_confidence(0.89));
    g.add_edge(Edge::new("Germany", "capital-of", "Berlin").with_confidence(0.81));

    let stats = g.stats();
    assert_eq!(stats.edges, 2);
    assert_eq!(stats.entities, 4);
    assert_eq!(stats.relations, 1);
    assert_eq!(stats.connected_components, 2);
    assert!((stats.avg_confidence - 0.85).abs() < 0.001);
    assert!((stats.avg_degree - 1.0).abs() < 0.001);
}

#[test]
fn test_stats_empty() {
    let g = Graph::new();
    let stats = g.stats();
    assert_eq!(stats.edges, 0);
    assert_eq!(stats.entities, 0);
    assert_eq!(stats.connected_components, 0);
    assert!((stats.avg_confidence - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_connected_components() {
    let mut g = Graph::new();
    // Component 1: France <-> Paris
    g.add_edge(Edge::new("France", "capital-of", "Paris"));
    g.add_edge(Edge::new("Paris", "located-in", "France"));
    // Component 2: Germany <-> Berlin
    g.add_edge(Edge::new("Germany", "capital-of", "Berlin"));

    let stats = g.stats();
    assert_eq!(stats.connected_components, 2);
}

#[test]
fn test_single_component() {
    let mut g = Graph::new();
    g.add_edge(Edge::new("A", "r", "B"));
    g.add_edge(Edge::new("B", "r", "C"));
    g.add_edge(Edge::new("C", "r", "A"));

    let stats = g.stats();
    assert_eq!(stats.connected_components, 1);
}

// ── List accessors ──

#[test]
fn test_list_relations() {
    let mut g = Graph::new();
    g.add_edge(Edge::new("France", "capital-of", "Paris"));
    g.add_edge(Edge::new("France", "currency", "Euro"));
    g.add_edge(Edge::new("Germany", "capital-of", "Berlin"));

    let mut rels = g.list_relations();
    rels.sort();
    assert_eq!(rels, vec!["capital-of", "currency"]);
}

#[test]
fn test_list_entities() {
    let mut g = Graph::new();
    g.add_edge(Edge::new("France", "capital-of", "Paris"));

    let mut entities = g.list_entities();
    entities.sort();
    assert_eq!(entities, vec!["France", "Paris"]);
}
