//! Integration tests for walker modules using a tiny mock model.
//!
//! Creates a fake safetensors model with 2 layers, hidden_size=8,
//! intermediate_size=4, vocab_size=16, head_dim=4, 2 attention heads.
//! Tests the full pipeline: load → walk → edges.

mod walker_tests {
    use std::collections::HashMap;
    use std::path::Path;

    /// Create a tiny mock model directory with safetensors weights + config + tokenizer.
    fn create_mock_model(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();

        let hidden = 8usize;
        let intermediate = 4usize;
        let vocab = 16usize;
        let head_dim = 4usize;
        let num_q_heads = 2usize;
        let num_kv_heads = 2usize;
        let num_layers = 2usize;

        // Build weight tensors
        let mut tensors: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();

        // Embedding: (vocab, hidden)
        tensors.insert(
            "embed_tokens.weight".into(),
            (random_f32(vocab * hidden, 42), vec![vocab, hidden]),
        );

        // Final norm
        tensors.insert("norm.weight".into(), (vec![0.0f32; hidden], vec![hidden]));

        for layer in 0..num_layers {
            let p = format!("layers.{layer}");

            // Norms (1D)
            for norm in &[
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "pre_feedforward_layernorm.weight",
                "post_feedforward_layernorm.weight",
            ] {
                tensors.insert(format!("{p}.{norm}"), (vec![0.0f32; hidden], vec![hidden]));
            }

            // Q/K norms (head_dim)
            tensors.insert(
                format!("{p}.self_attn.q_norm.weight"),
                (vec![0.0f32; head_dim], vec![head_dim]),
            );
            tensors.insert(
                format!("{p}.self_attn.k_norm.weight"),
                (vec![0.0f32; head_dim], vec![head_dim]),
            );

            // Attention projections (2D)
            tensors.insert(
                format!("{p}.self_attn.q_proj.weight"),
                (
                    random_f32(num_q_heads * head_dim * hidden, layer * 100 + 1),
                    vec![num_q_heads * head_dim, hidden],
                ),
            );
            tensors.insert(
                format!("{p}.self_attn.k_proj.weight"),
                (
                    random_f32(num_kv_heads * head_dim * hidden, layer * 100 + 2),
                    vec![num_kv_heads * head_dim, hidden],
                ),
            );
            tensors.insert(
                format!("{p}.self_attn.v_proj.weight"),
                (
                    random_f32(num_kv_heads * head_dim * hidden, layer * 100 + 3),
                    vec![num_kv_heads * head_dim, hidden],
                ),
            );
            tensors.insert(
                format!("{p}.self_attn.o_proj.weight"),
                (
                    random_f32(hidden * num_q_heads * head_dim, layer * 100 + 4),
                    vec![hidden, num_q_heads * head_dim],
                ),
            );

            // FFN projections (2D)
            tensors.insert(
                format!("{p}.mlp.gate_proj.weight"),
                (
                    random_f32(intermediate * hidden, layer * 100 + 5),
                    vec![intermediate, hidden],
                ),
            );
            tensors.insert(
                format!("{p}.mlp.up_proj.weight"),
                (
                    random_f32(intermediate * hidden, layer * 100 + 6),
                    vec![intermediate, hidden],
                ),
            );
            tensors.insert(
                format!("{p}.mlp.down_proj.weight"),
                (
                    random_f32(hidden * intermediate, layer * 100 + 7),
                    vec![hidden, intermediate],
                ),
            );
        }

        // Write safetensors file
        write_safetensors(dir, &tensors);

        // Write config.json
        let config = serde_json::json!({
            "model_type": "gemma3",
            "text_config": {
                "model_type": "gemma3_text",
                "num_hidden_layers": num_layers,
                "hidden_size": hidden,
                "intermediate_size": intermediate,
                "head_dim": head_dim,
                "num_attention_heads": num_q_heads,
                "num_key_value_heads": num_kv_heads,
                "rope_theta": 10000.0
            }
        });
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_string_pretty(&config).unwrap(),
        )
        .unwrap();

        // Write minimal tokenizer.json
        write_mock_tokenizer(dir, vocab);
    }

    fn random_f32(n: usize, seed: usize) -> Vec<f32> {
        // Deterministic pseudo-random for reproducible tests
        let mut vals = Vec::with_capacity(n);
        let mut x = seed as u64 * 2654435761 + 1;
        for _ in 0..n {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let f = ((x >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
            vals.push(f * 0.1); // small values
        }
        vals
    }

    fn write_safetensors(dir: &Path, tensors: &HashMap<String, (Vec<f32>, Vec<usize>)>) {
        let mut data_map: HashMap<String, safetensors::tensor::TensorView<'_>> = HashMap::new();
        let mut byte_bufs: HashMap<String, Vec<u8>> = HashMap::new();

        // First pass: convert f32 to bytes
        for (name, (values, _shape)) in tensors {
            let bytes: Vec<u8> = values.iter().flat_map(|f| f.to_le_bytes()).collect();
            byte_bufs.insert(name.clone(), bytes);
        }

        // Second pass: create TensorView references
        for (name, (_, shape)) in tensors {
            let bytes = &byte_bufs[name];
            data_map.insert(
                name.clone(),
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                    .unwrap(),
            );
        }

        let serialized = safetensors::tensor::serialize(&data_map, &None).unwrap();
        std::fs::write(dir.join("model.safetensors"), serialized).unwrap();
    }

    fn write_mock_tokenizer(dir: &Path, vocab_size: usize) {
        let tokens = [
            "the", "a", "is", "of", "France", "Paris", "Germany", "Berlin", "capital", "Europe",
            "language", "French", "city", "country", "and", "in",
        ];

        let mut vocab = serde_json::Map::new();
        for (i, tok) in tokens.iter().enumerate().take(vocab_size) {
            vocab.insert(tok.to_string(), serde_json::json!(i));
        }

        // Format that the tokenizers crate's Tokenizer::from_file accepts
        let tokenizer_json = serde_json::json!({
            "version": "1.0",
            "model": {
                "type": "WordLevel",
                "vocab": vocab,
                "unk_token": "the"
            },
            "pre_tokenizer": {
                "type": "Whitespace"
            }
        });

        std::fs::write(
            dir.join("tokenizer.json"),
            serde_json::to_string_pretty(&tokenizer_json).unwrap(),
        )
        .unwrap();
    }

    // ── Weight Walker Tests ──

    #[test]
    fn test_weight_walker_loads() {
        let dir = std::env::temp_dir().join("larql_test_weight_load");
        let _ = std::fs::remove_dir_all(&dir);
        create_mock_model(&dir);

        let walker =
            larql_inference::walker::weight_walker::WeightWalker::load(dir.to_str().unwrap())
                .unwrap();
        assert_eq!(walker.num_layers(), 2);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_weight_walker_extracts_edges() {
        let dir = std::env::temp_dir().join("larql_test_weight_edges");
        let _ = std::fs::remove_dir_all(&dir);
        create_mock_model(&dir);

        let walker =
            larql_inference::walker::weight_walker::WeightWalker::load(dir.to_str().unwrap())
                .unwrap();
        let config = larql_inference::walker::weight_walker::WalkConfig {
            top_k: 3,
            min_score: 0.0,
        };
        let mut graph = larql_core::Graph::new();
        let mut callbacks = larql_inference::walker::weight_walker::SilentWalkCallbacks;

        let result = walker
            .walk_layer(0, &config, &mut graph, &mut callbacks)
            .unwrap();

        assert_eq!(result.layer, 0);
        assert_eq!(result.features_scanned, 4); // intermediate_size
        assert!(result.edges_found > 0);
        assert!(graph.edge_count() > 0);

        // Edges should have metadata
        for edge in graph.edges() {
            assert_eq!(edge.source, larql_core::SourceType::Parametric);
            let meta = edge.metadata.as_ref().unwrap();
            assert!(meta.contains_key("layer"));
            assert!(meta.contains_key("feature"));
            assert!(meta.contains_key("c_in"));
            assert!(meta.contains_key("c_out"));
            assert!(meta.contains_key("selectivity"));
        }

        // Confidence should be normalized [0, 1]
        for edge in graph.edges() {
            assert!(edge.confidence >= 0.0);
            assert!(edge.confidence <= 1.0);
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_weight_walker_all_layers() {
        let dir = std::env::temp_dir().join("larql_test_weight_all");
        let _ = std::fs::remove_dir_all(&dir);
        create_mock_model(&dir);

        let config = larql_inference::walker::weight_walker::WalkConfig {
            top_k: 2,
            min_score: 0.0,
        };
        let mut graph = larql_core::Graph::new();
        let mut callbacks = larql_inference::walker::weight_walker::SilentWalkCallbacks;

        let results = larql_inference::walk_model(
            dir.to_str().unwrap(),
            None,
            &config,
            &mut graph,
            &mut callbacks,
        )
        .unwrap();

        assert_eq!(results.len(), 2); // 2 layers
        assert!(graph.edge_count() > 0);

        // Both layers should have edges
        let l0_edges: Vec<_> = graph
            .edges()
            .iter()
            .filter(|e| {
                e.metadata
                    .as_ref()
                    .and_then(|m| m.get("layer"))
                    .and_then(|v| v.as_u64())
                    == Some(0)
            })
            .collect();
        let l1_edges: Vec<_> = graph
            .edges()
            .iter()
            .filter(|e| {
                e.metadata
                    .as_ref()
                    .and_then(|m| m.get("layer"))
                    .and_then(|v| v.as_u64())
                    == Some(1)
            })
            .collect();
        assert!(!l0_edges.is_empty());
        assert!(!l1_edges.is_empty());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_weight_walker_layer_stats() {
        let dir = std::env::temp_dir().join("larql_test_weight_stats");
        let _ = std::fs::remove_dir_all(&dir);
        create_mock_model(&dir);

        let walker =
            larql_inference::walker::weight_walker::WeightWalker::load(dir.to_str().unwrap())
                .unwrap();
        let config = larql_inference::walker::weight_walker::WalkConfig {
            top_k: 3,
            min_score: 0.0,
        };
        let mut graph = larql_core::Graph::new();
        let mut callbacks = larql_inference::walker::weight_walker::SilentWalkCallbacks;

        let result = walker
            .walk_layer(0, &config, &mut graph, &mut callbacks)
            .unwrap();

        let stats = &result.stats;
        assert!(stats.mean_confidence >= 0.0);
        assert!(stats.max_confidence <= 1.0);
        assert!(stats.mean_selectivity >= 0.0);
        assert!(stats.mean_c_in >= 0.0);
        assert!(stats.mean_c_out >= 0.0);

        std::fs::remove_dir_all(&dir).ok();
    }

    // ── Attention Walker Tests ──

    #[test]
    fn test_attention_walker_loads() {
        let dir = std::env::temp_dir().join("larql_test_attn_load");
        let _ = std::fs::remove_dir_all(&dir);
        create_mock_model(&dir);

        let walker =
            larql_inference::walker::attention_walker::AttentionWalker::load(dir.to_str().unwrap())
                .unwrap();
        assert_eq!(walker.num_layers(), 2);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_attention_walker_extracts_edges() {
        let dir = std::env::temp_dir().join("larql_test_attn_edges");
        let _ = std::fs::remove_dir_all(&dir);
        create_mock_model(&dir);

        let walker =
            larql_inference::walker::attention_walker::AttentionWalker::load(dir.to_str().unwrap())
                .unwrap();
        let config = larql_inference::walker::weight_walker::WalkConfig {
            top_k: 2,
            min_score: 0.0,
        };
        let mut graph = larql_core::Graph::new();
        let mut callbacks = larql_inference::walker::weight_walker::SilentWalkCallbacks;

        let result = walker
            .walk_layer(0, &config, &mut graph, &mut callbacks)
            .unwrap();

        assert_eq!(result.layer, 0);
        assert_eq!(result.heads_walked, 2); // num_kv_heads
        assert!(result.edges_found > 0);

        // Edges should have OV circuit metadata
        for edge in graph.edges() {
            let meta = edge.metadata.as_ref().unwrap();
            assert!(meta.contains_key("layer"));
            assert!(meta.contains_key("head"));
            assert_eq!(meta["circuit"], "OV");
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    // ── Vector Extractor Tests ──

    #[test]
    fn test_vector_extractor_ffn_down() {
        let dir = std::env::temp_dir().join("larql_test_vec_down");
        let _ = std::fs::remove_dir_all(&dir);
        create_mock_model(&dir);

        let extractor =
            larql_inference::walker::vector_extractor::VectorExtractor::load(dir.to_str().unwrap())
                .unwrap();
        assert_eq!(extractor.num_layers(), 2);
        assert_eq!(extractor.hidden_size(), 8);

        let output_dir = dir.join("output");
        std::fs::create_dir_all(&output_dir).unwrap();

        let config = larql_inference::walker::vector_extractor::ExtractConfig {
            components: vec!["ffn_down".to_string()],
            layers: Some(vec![0]),
            top_k: 3,
        };
        let mut callbacks = larql_inference::walker::vector_extractor::SilentExtractCallbacks;

        let summary = extractor
            .extract_all(&config, &output_dir, false, &mut callbacks)
            .unwrap();

        assert_eq!(summary.total_vectors, 4); // intermediate_size
        assert_eq!(summary.components.len(), 1);
        assert_eq!(summary.components[0].component, "ffn_down");

        // Verify output file exists
        let output_file = output_dir.join("ffn_down.vectors.jsonl");
        assert!(output_file.exists());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_vector_extractor_embeddings() {
        let dir = std::env::temp_dir().join("larql_test_vec_embed");
        let _ = std::fs::remove_dir_all(&dir);
        create_mock_model(&dir);

        let extractor =
            larql_inference::walker::vector_extractor::VectorExtractor::load(dir.to_str().unwrap())
                .unwrap();

        let output_dir = dir.join("output");
        std::fs::create_dir_all(&output_dir).unwrap();

        let config = larql_inference::walker::vector_extractor::ExtractConfig {
            components: vec!["embeddings".to_string()],
            layers: None,
            top_k: 3,
        };
        let mut callbacks = larql_inference::walker::vector_extractor::SilentExtractCallbacks;

        let summary = extractor
            .extract_all(&config, &output_dir, false, &mut callbacks)
            .unwrap();

        assert_eq!(summary.total_vectors, 16); // vocab_size

        std::fs::remove_dir_all(&dir).ok();
    }

    // ── Safetensors Loader Tests ──

    #[test]
    fn test_loader_loads_all_tensors() {
        let dir = std::env::temp_dir().join("larql_test_loader");
        let _ = std::fs::remove_dir_all(&dir);
        create_mock_model(&dir);

        let weights = larql_inference::model::load_model_dir(&dir).unwrap();

        assert_eq!(weights.num_layers, 2);
        assert_eq!(weights.hidden_size, 8);
        assert_eq!(weights.intermediate_size, 4);
        assert_eq!(weights.vocab_size, 16);
        assert_eq!(weights.head_dim, 4);
        assert_eq!(weights.num_q_heads, 2);
        assert_eq!(weights.num_kv_heads, 2);
        assert_eq!(weights.embed.shape(), &[16, 8]);

        // Check 2D tensors exist
        assert!(weights
            .tensors
            .contains_key("layers.0.mlp.gate_proj.weight"));
        assert!(weights
            .tensors
            .contains_key("layers.1.mlp.down_proj.weight"));
        assert!(weights
            .tensors
            .contains_key("layers.0.self_attn.q_proj.weight"));

        // Check 1D tensors (norms)
        assert!(weights
            .vectors
            .contains_key("layers.0.input_layernorm.weight"));
        assert!(weights.vectors.contains_key("norm.weight"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_loader_missing_directory() {
        let result = larql_inference::model::load_model_dir("/nonexistent/path/model");
        assert!(result.is_err());
    }

    // ── Forward Pass Tests ──

    #[test]
    fn test_forward_captures_residuals() {
        let dir = std::env::temp_dir().join("larql_test_forward");
        let _ = std::fs::remove_dir_all(&dir);
        create_mock_model(&dir);

        let weights = larql_inference::model::load_model_dir(&dir).unwrap();

        // Token IDs within vocab range
        let token_ids = vec![4u32, 5, 0]; // "France", "Paris", "the"

        let residuals = larql_inference::forward::capture_residuals(&weights, &token_ids, &[0, 1]);

        assert_eq!(residuals.len(), 2); // captured at layer 0 and 1
        assert_eq!(residuals[0].0, 0); // layer 0
        assert_eq!(residuals[1].0, 1); // layer 1
        assert_eq!(residuals[0].1.len(), 8); // hidden_size
        assert_eq!(residuals[1].1.len(), 8);

        // Residuals should be different between layers
        assert_ne!(residuals[0].1, residuals[1].1);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_forward_single_token() {
        let dir = std::env::temp_dir().join("larql_test_forward_single");
        let _ = std::fs::remove_dir_all(&dir);
        create_mock_model(&dir);

        let weights = larql_inference::model::load_model_dir(&dir).unwrap();
        let residuals = larql_inference::forward::capture_residuals(&weights, &[0], &[1]);

        assert_eq!(residuals.len(), 1);
        assert_eq!(residuals[0].1.len(), 8);

        std::fs::remove_dir_all(&dir).ok();
    }
}
