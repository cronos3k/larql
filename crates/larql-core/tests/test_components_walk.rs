//! Tests for connected components and advanced walk strategies.

use larql_core::*;

fn geo_graph() -> Graph {
    let mut g = Graph::new();
    // Component 1: Europe
    g.add_edge(Edge::new("France", "capital", "Paris").with_confidence(0.95));
    g.add_edge(Edge::new("France", "continent", "Europe").with_confidence(0.90));
    g.add_edge(Edge::new("Germany", "capital", "Berlin").with_confidence(0.92));
    g.add_edge(Edge::new("Germany", "continent", "Europe").with_confidence(0.88));

    // Component 2: Asia (disconnected)
    g.add_edge(Edge::new("Japan", "capital", "Tokyo").with_confidence(0.93));
    g.add_edge(Edge::new("Japan", "continent", "Asia").with_confidence(0.91));
    g
}

fn chain_graph() -> Graph {
    let mut g = Graph::new();
    // A -> B -> C with multiple paths
    g.add_edge(Edge::new("A", "leads_to", "B").with_confidence(0.9));
    g.add_edge(Edge::new("A", "leads_to", "B2").with_confidence(0.7));
    g.add_edge(Edge::new("B", "leads_to", "C").with_confidence(0.8));
    g.add_edge(Edge::new("B2", "leads_to", "C").with_confidence(0.6));
    g.add_edge(Edge::new("B2", "leads_to", "C2").with_confidence(0.5));
    g
}

// ══════════════════════════════════════════════════════════════
// CONNECTED COMPONENTS
// ══════════════════════════════════════════════════════════════

#[test]
fn components_finds_two_components() {
    let g = geo_graph();
    let comps = connected_components(&g);
    assert_eq!(comps.len(), 2, "should find 2 disconnected components");
    // Largest first
    assert!(comps[0].len() >= comps[1].len());
}

#[test]
fn components_europe_and_asia_separate() {
    let g = geo_graph();
    assert!(are_connected(&g, "France", "Germany"), "France-Germany via Europe");
    assert!(are_connected(&g, "France", "Paris"), "France-Paris direct");
    assert!(!are_connected(&g, "France", "Japan"), "France-Japan disconnected");
    assert!(!are_connected(&g, "Paris", "Tokyo"), "Paris-Tokyo disconnected");
}

#[test]
fn components_single_component() {
    let mut g = Graph::new();
    g.add_edge(Edge::new("A", "to", "B").with_confidence(1.0));
    g.add_edge(Edge::new("B", "to", "C").with_confidence(1.0));
    let comps = connected_components(&g);
    assert_eq!(comps.len(), 1);
    assert_eq!(comps[0].len(), 3);
}

#[test]
fn components_empty_graph() {
    let g = Graph::new();
    let comps = connected_components(&g);
    assert!(comps.is_empty());
}

#[test]
fn are_connected_same_node() {
    let g = geo_graph();
    assert!(are_connected(&g, "France", "France"));
}

#[test]
fn are_connected_nonexistent() {
    let g = geo_graph();
    assert!(!are_connected(&g, "France", "Narnia"));
}

// ══════════════════════════════════════════════════════════════
// WALK: HIGHEST CONFIDENCE SELECTION
// ══════════════════════════════════════════════════════════════

#[test]
fn walk_picks_highest_confidence() {
    let mut g = Graph::new();
    g.add_edge(Edge::new("France", "capital", "Paris").with_confidence(0.95));
    g.add_edge(Edge::new("France", "capital", "Lyon").with_confidence(0.3));
    g.add_edge(Edge::new("France", "capital", "Marseille").with_confidence(0.1));

    let (dest, path) = g.walk("France", &["capital"]).unwrap();
    assert_eq!(dest, "Paris", "should pick Paris (highest confidence)");
    assert_eq!(path.len(), 1);
    assert!((path[0].confidence - 0.95).abs() < 1e-10);
}

#[test]
fn walk_returns_none_on_missing_relation() {
    let g = geo_graph();
    assert!(g.walk("France", &["currency"]).is_none());
}

#[test]
fn walk_multi_hop() {
    let mut g = Graph::new();
    g.add_edge(Edge::new("France", "capital", "Paris").with_confidence(0.95));
    g.add_edge(Edge::new("Paris", "river", "Seine").with_confidence(0.88));

    let (dest, path) = g.walk("France", &["capital", "river"]).unwrap();
    assert_eq!(dest, "Seine");
    assert_eq!(path.len(), 2);
}

// ══════════════════════════════════════════════════════════════
// WALK ALL PATHS
// ══════════════════════════════════════════════════════════════

#[test]
fn walk_all_paths_finds_multiple() {
    let g = chain_graph();
    let paths = walk_all_paths(&g, "A", &["leads_to", "leads_to"], 10);

    // A->B->C, A->B2->C, A->B2->C2 = 3 paths
    assert_eq!(paths.len(), 3, "should find 3 two-hop paths from A");

    // Best path first (highest min confidence)
    assert!(paths[0].min_confidence >= paths[1].min_confidence);
    assert!(paths[1].min_confidence >= paths[2].min_confidence);

    // Best path: A->B(0.9)->C(0.8), min=0.8
    assert_eq!(paths[0].destination, "C");
    assert!((paths[0].min_confidence - 0.8).abs() < 1e-10);
}

#[test]
fn walk_all_paths_max_limit() {
    let g = chain_graph();
    let paths = walk_all_paths(&g, "A", &["leads_to", "leads_to"], 2);
    assert_eq!(paths.len(), 2, "should respect max_paths limit");
}

#[test]
fn walk_all_paths_no_match() {
    let g = chain_graph();
    let paths = walk_all_paths(&g, "A", &["nonexistent"], 10);
    assert!(paths.is_empty());
}

#[test]
fn walk_all_paths_single_hop() {
    let g = chain_graph();
    let paths = walk_all_paths(&g, "A", &["leads_to"], 10);
    assert_eq!(paths.len(), 2); // A->B and A->B2
}

// ══════════════════════════════════════════════════════════════
// REMOVE EDGE + INDEX REBUILD
// ══════════════════════════════════════════════════════════════

#[test]
fn remove_edge_rebuilds_indexes() {
    let mut g = Graph::new();
    g.add_edge(Edge::new("France", "capital", "Paris").with_confidence(0.95));
    g.add_edge(Edge::new("France", "language", "French").with_confidence(0.90));
    g.add_edge(Edge::new("Germany", "capital", "Berlin").with_confidence(0.92));

    assert_eq!(g.edge_count(), 3);
    assert!(g.exists("France", "capital", "Paris"));

    // Remove an edge
    assert!(g.remove_edge("France", "capital", "Paris"));
    assert_eq!(g.edge_count(), 2);
    assert!(!g.exists("France", "capital", "Paris"));

    // Other edges still queryable
    assert!(g.exists("France", "language", "French"));
    assert!(g.exists("Germany", "capital", "Berlin"));

    // Select still works after rebuild
    let france_edges = g.select("France", None);
    assert_eq!(france_edges.len(), 1);
    assert_eq!(france_edges[0].relation, "language");
}

#[test]
fn remove_edge_nonexistent() {
    let mut g = geo_graph();
    let count_before = g.edge_count();
    assert!(!g.remove_edge("France", "capital", "Berlin"));
    assert_eq!(g.edge_count(), count_before);
}

// ══════════════════════════════════════════════════════════════
// SEARCH EDGE CASES
// ══════════════════════════════════════════════════════════════

#[test]
fn search_empty_query() {
    let g = geo_graph();
    let results = g.search("", 10);
    assert!(results.is_empty());
}

#[test]
fn search_no_match() {
    let g = geo_graph();
    let results = g.search("Narnia", 10);
    assert!(results.is_empty());
}

#[test]
fn search_case_insensitive() {
    let g = geo_graph();
    let r1 = g.search("france", 10);
    let r2 = g.search("FRANCE", 10);
    // Both should find the same edges (keyword index is lowercase)
    assert_eq!(r1.len(), r2.len());
}

// ══════════════════════════════════════════════════════════════
// DEDUPLICATION
// ══════════════════════════════════════════════════════════════

#[test]
fn deduplicate_after_merge() {
    // Merge two graphs with overlapping edges to create duplicates,
    // then test dedup with MaxConfidence strategy.
    let mut g1 = Graph::new();
    g1.add_edge(Edge::new("France", "capital", "Paris").with_confidence(0.5));
    g1.add_edge(Edge::new("France", "language", "French").with_confidence(0.8));

    let mut g2 = Graph::new();
    g2.add_edge(Edge::new("France", "capital", "Paris").with_confidence(0.95));
    g2.add_edge(Edge::new("Germany", "capital", "Berlin").with_confidence(0.9));

    // Union merge keeps both — g1 already has Paris so it won't add the duplicate
    let added = merge_graphs(&mut g1, &g2);
    // Only Berlin should be added (Paris is duplicate)
    assert_eq!(added, 1);
    assert_eq!(g1.edge_count(), 3);

    // MaxConfidence merge strategy replaces lower-confidence edges
    let mut g3 = Graph::new();
    g3.add_edge(Edge::new("France", "capital", "Paris").with_confidence(0.5));
    merge_graphs_with_strategy(&mut g3, &g2, MergeStrategy::MaxConfidence);
    // Paris should now be 0.95 (replaced by higher confidence)
    let edges = g3.select("France", Some("capital"));
    assert!((edges[0].confidence - 0.95).abs() < 1e-10);
}
