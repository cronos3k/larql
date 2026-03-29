use larql_core::engine::mock_provider::MockProvider;
use larql_core::*;

#[test]
fn test_chain_single_token() {
    let provider = MockProvider::with_knowledge(vec![(
        "The capital of France is".into(),
        "Paris".into(),
        0.89,
    )]);

    let result = chain_tokens(&provider, "The capital of France is", 1, 0.3, None).unwrap();
    assert_eq!(result.answer, "Paris");
    assert_eq!(result.num_passes, 1);
    assert!((result.avg_probability() - 0.89).abs() < 0.001);
}

#[test]
fn test_chain_stops_on_low_confidence() {
    let provider = MockProvider::with_knowledge(vec![("prompt".into(), "answer".into(), 0.1)]);

    let result = chain_tokens(&provider, "prompt", 5, 0.5, None).unwrap();
    assert!(result.answer.is_empty());
    assert_eq!(result.num_passes, 0);
}

#[test]
fn test_chain_stops_on_empty_response() {
    let provider = MockProvider::new(); // no knowledge

    let result = chain_tokens(&provider, "unknown prompt", 5, 0.1, None).unwrap();
    assert!(result.answer.is_empty());
    assert_eq!(result.num_passes, 0);
}

#[test]
fn test_chain_result_min_probability() {
    let result = ChainResult {
        answer: "test".to_string(),
        tokens: vec!["a".into(), "b".into()],
        probabilities: vec![0.9, 0.5],
        num_passes: 2,
    };

    assert!((result.min_probability() - 0.5).abs() < f64::EPSILON);
    assert!((result.avg_probability() - 0.7).abs() < f64::EPSILON);
}

#[test]
fn test_chain_result_empty() {
    let result = ChainResult {
        answer: String::new(),
        tokens: vec![],
        probabilities: vec![],
        num_passes: 0,
    };

    assert!((result.min_probability() - 0.0).abs() < f64::EPSILON);
    assert!((result.avg_probability() - 0.0).abs() < f64::EPSILON);
}
