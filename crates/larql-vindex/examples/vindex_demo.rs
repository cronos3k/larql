//! Vindex Demo — create, query, mutate, and save a vindex in memory.
//!
//! Demonstrates the core larql-vindex API without needing a real model.
//!
//! Run: cargo run -p larql-vindex --example vindex_demo

use larql_models::TopKEntry;
use larql_vindex::{FeatureMeta, VectorIndex, VindexConfig};
use ndarray::{Array1, Array2};

fn main() {
    println!("=== Vindex Demo ===\n");

    // ── 1. Build an index in memory ──
    section("Build in-memory index");

    let hidden = 4;
    let num_features = 5;
    let num_layers = 2;

    // Create gate vectors: each feature responds to a different direction
    let mut gate0 = Array2::<f32>::zeros((num_features, hidden));
    gate0[[0, 0]] = 10.0; // "France" feature
    gate0[[1, 1]] = 10.0; // "Germany" feature
    gate0[[2, 2]] = 10.0; // "Japan" feature
    gate0[[3, 0]] = 5.0;
    gate0[[3, 1]] = 5.0;  // "European" feature (France + Germany)
    // feature 4 is empty (free slot)

    let gate1 = Array2::<f32>::zeros((num_features, hidden));

    let gate_vectors = vec![Some(gate0), Some(gate1)];

    // Create metadata: what each feature outputs
    let meta0 = vec![
        Some(meta("Paris", 100, 0.95, &["France", "french"])),
        Some(meta("Berlin", 101, 0.92, &["Germany", "german"])),
        Some(meta("Tokyo", 102, 0.88, &["Japan", "japanese"])),
        Some(meta("European", 103, 0.70, &["Europe", "EU"])),
        None, // free slot
    ];
    let meta1 = vec![None; num_features];

    let down_meta = vec![Some(meta0), Some(meta1)];

    let index = VectorIndex::new(gate_vectors, down_meta, num_layers, hidden);

    println!("  Created: {} layers, {} features/layer, {} hidden",
        index.num_layers, num_features, hidden);
    println!("  Total features: {}", index.total_gate_vectors());
    println!("  With metadata:  {}", index.total_down_meta());
    println!("  Loaded layers:  {:?}", index.loaded_layers());

    // ── 2. Gate KNN ──
    section("Gate KNN — find features for a query");

    // Query "France" (dim 0)
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    println!("  Query: [1, 0, 0, 0] (\"France\" direction)");
    let hits = index.gate_knn(0, &query, 3);
    for (feat, score) in &hits {
        let label = index.feature_meta(0, *feat)
            .map(|m| m.top_token.as_str())
            .unwrap_or("(none)");
        println!("    F{}: {} (score={:.1})", feat, label, score);
    }

    // Query "European" (dim 0 + dim 1)
    println!();
    let query2 = Array1::from_vec(vec![0.7, 0.7, 0.0, 0.0]);
    println!("  Query: [0.7, 0.7, 0, 0] (\"European\" direction)");
    let hits2 = index.gate_knn(0, &query2, 3);
    for (feat, score) in &hits2 {
        let label = index.feature_meta(0, *feat)
            .map(|m| m.top_token.as_str())
            .unwrap_or("(none)");
        println!("    F{}: {} (score={:.1})", feat, label, score);
    }

    // ── 3. Walk ──
    section("Walk — multi-layer feature scan");

    let query3 = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let trace = index.walk(&query3, &[0, 1], 3);
    for (layer, hits) in &trace.layers {
        if hits.is_empty() {
            println!("  L{}: (no hits)", layer);
        }
        for hit in hits {
            println!("  L{}: F{} → {} (gate={:.1})",
                layer, hit.feature, hit.meta.top_token, hit.gate_score);
        }
    }

    // ── 4. Mutate ──
    section("Mutate — insert an edge");

    let mut index = index; // make mutable

    // Find free slot
    let free = index.find_free_feature(0);
    println!("  Free slot at layer 0: {:?}", free);

    if let Some(slot) = free {
        // Set gate vector for "Australia"
        let gate_vec = Array1::from_vec(vec![0.0, 0.0, 0.0, 10.0]);
        index.set_gate_vector(0, slot, &gate_vec);
        index.set_feature_meta(0, slot, meta("Canberra", 104, 0.85, &["Australia"]));
        println!("  Inserted: F{} → Canberra (Australia direction)", slot);
    }

    // Verify it works
    let query4 = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0]);
    let hits4 = index.gate_knn(0, &query4, 1);
    let (feat, score) = hits4[0];
    let label = index.feature_meta(0, feat).unwrap();
    println!("  Query [0,0,0,1] → F{}: {} (score={:.1})", feat, label.top_token, score);

    // ── 5. Delete ──
    section("Delete — remove an edge");

    println!("  Before: F2 = {:?}", index.feature_meta(0, 2).map(|m| &m.top_token));
    index.delete_feature_meta(0, 2);
    println!("  After:  F2 = {:?}", index.feature_meta(0, 2).map(|m| &m.top_token));
    println!("  Free slots: {:?}", index.find_free_feature(0));

    // ── 6. Save round-trip ──
    section("Save + Load round-trip");

    let dir = std::env::temp_dir().join("larql_vindex_demo");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let layer_infos = index.save_gate_vectors(&dir).unwrap();
    let down_count = index.save_down_meta(&dir).unwrap();

    let config = VindexConfig {
        version: 1,
        model: "demo-model".into(),
        family: "demo".into(),
        num_layers,
        hidden_size: hidden,
        intermediate_size: num_features,
        vocab_size: 200,
        embed_scale: 1.0,
        layers: layer_infos,
        down_top_k: 2,
        has_model_weights: false,
        model_config: None,
    };
    VectorIndex::save_config(&config, &dir).unwrap();

    println!("  Saved to: {}", dir.display());
    println!("  Down meta records: {}", down_count);

    // Reload
    let mut cb = larql_vindex::SilentLoadCallbacks;
    let loaded = VectorIndex::load_vindex(&dir, &mut cb).unwrap();
    println!("  Reloaded: {} layers, {} features", loaded.num_layers, loaded.total_gate_vectors());

    // Verify
    let hits5 = loaded.gate_knn(0, &Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]), 1);
    let m = loaded.feature_meta(0, hits5[0].0).unwrap();
    println!("  Query [1,0,0,0] → {} (round-trip verified)", m.top_token);

    let _ = std::fs::remove_dir_all(&dir);

    println!("\n=== Done ===");
}

fn section(name: &str) {
    println!("\n── {} ──\n", name);
}

fn meta(token: &str, id: u32, score: f32, also: &[&str]) -> FeatureMeta {
    let mut top_k = vec![TopKEntry {
        token: token.to_string(),
        token_id: id,
        logit: score,
    }];
    for (i, t) in also.iter().enumerate() {
        top_k.push(TopKEntry {
            token: t.to_string(),
            token_id: id + 1000 + i as u32,
            logit: score * 0.5,
        });
    }
    FeatureMeta {
        top_token: token.to_string(),
        top_token_id: id,
        c_score: score,
        top_k,
    }
}
