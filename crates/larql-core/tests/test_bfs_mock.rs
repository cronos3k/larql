use larql_core::engine::mock_provider::MockProvider;
use larql_core::engine::templates::PromptTemplate;
use larql_core::*;

fn geo_templates() -> TemplateRegistry {
    let mut reg = TemplateRegistry::new();
    reg.register(PromptTemplate {
        relation: "capital-of".to_string(),
        template: "The capital of {subject} is".to_string(),
        reverse_template: None,
        multi_token: true,
        stop_tokens: vec!['.', '\n', ',', ';'],
    });
    reg.register(PromptTemplate {
        relation: "currency".to_string(),
        template: "The currency of {subject} is".to_string(),
        reverse_template: None,
        multi_token: true,
        stop_tokens: vec!['.', '\n', ',', ';'],
    });
    reg.register(PromptTemplate {
        relation: "located-in".to_string(),
        template: "{subject} is located in".to_string(),
        reverse_template: None,
        multi_token: true,
        stop_tokens: vec!['.', '\n', ',', ';'],
    });
    reg
}

fn geo_provider() -> MockProvider {
    MockProvider::with_knowledge(vec![
        ("The capital of France is".into(), "Paris".into(), 0.89),
        ("The currency of France is".into(), "Euro".into(), 0.91),
        ("The capital of Germany is".into(), "Berlin".into(), 0.81),
        ("The currency of Germany is".into(), "Euro".into(), 0.88),
        ("Paris is located in".into(), "France".into(), 0.98),
        ("Berlin is located in".into(), "Germany".into(), 0.97),
    ])
}

#[test]
fn test_bfs_basic() {
    let provider = geo_provider();
    let templates = geo_templates();
    let seeds = vec!["France".to_string()];
    let config = BfsConfig {
        max_depth: 0,
        max_entities: 100,
        min_confidence: 0.3,
        ..Default::default()
    };

    let mut graph = Graph::new();
    let mut callbacks = larql_core::engine::bfs::SilentCallbacks;
    let result = extract_bfs(
        &provider,
        &templates,
        &seeds,
        &config,
        &mut graph,
        &mut callbacks,
    );

    assert_eq!(result.entities_visited, 1);
    assert!(result.edges_added >= 2); // capital-of and currency at minimum
    assert!(graph.exists("France", "capital-of", "Paris"));
    assert!(graph.exists("France", "currency", "Euro"));
}

#[test]
fn test_bfs_depth_1_follows_entities() {
    let provider = geo_provider();
    let templates = geo_templates();
    let seeds = vec!["France".to_string()];
    let config = BfsConfig {
        max_depth: 1,
        max_entities: 100,
        min_confidence: 0.3,
        ..Default::default()
    };

    let mut graph = Graph::new();
    let mut callbacks = larql_core::engine::bfs::SilentCallbacks;
    let result = extract_bfs(
        &provider,
        &templates,
        &seeds,
        &config,
        &mut graph,
        &mut callbacks,
    );

    // Should have visited France + discovered entities (Paris, Euro)
    assert!(result.entities_visited > 1);
    // Paris should have been probed and found located-in France
    assert!(graph.exists("Paris", "located-in", "France"));
}

#[test]
fn test_bfs_multiple_seeds() {
    let provider = geo_provider();
    let templates = geo_templates();
    let seeds = vec!["France".to_string(), "Germany".to_string()];
    let config = BfsConfig {
        max_depth: 0,
        max_entities: 100,
        min_confidence: 0.3,
        ..Default::default()
    };

    let mut graph = Graph::new();
    let mut callbacks = larql_core::engine::bfs::SilentCallbacks;
    extract_bfs(
        &provider,
        &templates,
        &seeds,
        &config,
        &mut graph,
        &mut callbacks,
    );

    assert!(graph.exists("France", "capital-of", "Paris"));
    assert!(graph.exists("Germany", "capital-of", "Berlin"));
}

#[test]
fn test_bfs_respects_max_entities() {
    let provider = geo_provider();
    let templates = geo_templates();
    let seeds = vec!["France".to_string(), "Germany".to_string()];
    let config = BfsConfig {
        max_depth: 3,
        max_entities: 1,
        min_confidence: 0.3,
        ..Default::default()
    };

    let mut graph = Graph::new();
    let mut callbacks = larql_core::engine::bfs::SilentCallbacks;
    let result = extract_bfs(
        &provider,
        &templates,
        &seeds,
        &config,
        &mut graph,
        &mut callbacks,
    );

    assert_eq!(result.entities_visited, 1);
}

#[test]
fn test_bfs_respects_min_confidence() {
    // Provider returns low confidence
    let provider = MockProvider::with_knowledge(vec![(
        "The capital of France is".into(),
        "Paris".into(),
        0.1,
    )]);
    let templates = geo_templates();
    let seeds = vec!["France".to_string()];
    let config = BfsConfig {
        max_depth: 0,
        max_entities: 100,
        min_confidence: 0.5,
        ..Default::default()
    };

    let mut graph = Graph::new();
    let mut callbacks = larql_core::engine::bfs::SilentCallbacks;
    extract_bfs(
        &provider,
        &templates,
        &seeds,
        &config,
        &mut graph,
        &mut callbacks,
    );

    // Edge should not be added (confidence 0.1 < threshold 0.5)
    assert!(!graph.exists("France", "capital-of", "Paris"));
}

#[test]
fn test_bfs_empty_provider() {
    let provider = MockProvider::new(); // no knowledge
    let templates = geo_templates();
    let seeds = vec!["France".to_string()];
    let config = BfsConfig::default();

    let mut graph = Graph::new();
    let mut callbacks = larql_core::engine::bfs::SilentCallbacks;
    let result = extract_bfs(
        &provider,
        &templates,
        &seeds,
        &config,
        &mut graph,
        &mut callbacks,
    );

    assert_eq!(result.entities_visited, 1);
    assert_eq!(result.edges_added, 0);
    assert_eq!(graph.edge_count(), 0);
}

#[test]
fn test_bfs_no_duplicate_visits() {
    let provider = geo_provider();
    let templates = geo_templates();
    let seeds = vec!["France".to_string()];
    let config = BfsConfig {
        max_depth: 3,
        max_entities: 100,
        min_confidence: 0.3,
        ..Default::default()
    };

    let mut graph = Graph::new();
    let mut callbacks = larql_core::engine::bfs::SilentCallbacks;
    extract_bfs(
        &provider,
        &templates,
        &seeds,
        &config,
        &mut graph,
        &mut callbacks,
    );

    // France -> Paris -> France loop: France should only be visited once
    // Check no duplicate edges
    let france_caps = graph.select("France", Some("capital-of"));
    assert_eq!(france_caps.len(), 1);
}

#[test]
fn test_bfs_edges_have_source_parametric() {
    let provider = geo_provider();
    let templates = geo_templates();
    let seeds = vec!["France".to_string()];
    let config = BfsConfig {
        max_depth: 0,
        max_entities: 100,
        min_confidence: 0.3,
        ..Default::default()
    };

    let mut graph = Graph::new();
    let mut callbacks = larql_core::engine::bfs::SilentCallbacks;
    extract_bfs(
        &provider,
        &templates,
        &seeds,
        &config,
        &mut graph,
        &mut callbacks,
    );

    for edge in graph.edges() {
        assert_eq!(edge.source, SourceType::Parametric);
    }
}

#[test]
fn test_bfs_edges_have_metadata() {
    let provider = geo_provider();
    let templates = geo_templates();
    let seeds = vec!["France".to_string()];
    let config = BfsConfig {
        max_depth: 0,
        max_entities: 100,
        min_confidence: 0.3,
        ..Default::default()
    };

    let mut graph = Graph::new();
    let mut callbacks = larql_core::engine::bfs::SilentCallbacks;
    extract_bfs(
        &provider,
        &templates,
        &seeds,
        &config,
        &mut graph,
        &mut callbacks,
    );

    for edge in graph.edges() {
        let meta = edge.metadata.as_ref().unwrap();
        assert!(meta.contains_key("forward_passes"));
        assert!(meta.contains_key("model"));
        assert_eq!(meta["model"], "mock/knowledge-base");
    }
}

#[test]
fn test_bfs_result_counts() {
    let provider = geo_provider();
    let templates = geo_templates();
    let seeds = vec!["France".to_string()];
    let config = BfsConfig {
        max_depth: 0,
        max_entities: 100,
        min_confidence: 0.3,
        ..Default::default()
    };

    let mut graph = Graph::new();
    let mut callbacks = larql_core::engine::bfs::SilentCallbacks;
    let result = extract_bfs(
        &provider,
        &templates,
        &seeds,
        &config,
        &mut graph,
        &mut callbacks,
    );

    assert_eq!(result.edges_added, graph.edge_count());
    assert!(result.total_forward_passes > 0);
}
