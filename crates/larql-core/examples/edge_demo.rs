//! Edge demo — construction, metadata, compact serialization.
//!
//! Run: cargo run --release -p larql-core --example edge_demo

use larql_core::*;

fn main() {
    println!("=== LARQL Edge Demo ===\n");

    // ── Basic edge ──
    let e1 = Edge::new("France", "capital-of", "Paris")
        .with_confidence(0.89)
        .with_source(SourceType::Parametric);

    println!(
        "Edge: {} --{}--> {} (c={:.2}, src={})",
        e1.subject,
        e1.relation,
        e1.object,
        e1.confidence,
        e1.source.as_str()
    );
    println!("  Triple: {:?}", e1.triple());

    // ── Edge with metadata (like weight-walk produces) ──
    let e2 = Edge::new("France", "L26-F9298", "Paris")
        .with_confidence(0.89)
        .with_source(SourceType::Parametric)
        .with_metadata("layer", serde_json::json!(26))
        .with_metadata("feature", serde_json::json!(9298))
        .with_metadata("c_in", serde_json::json!(8.7))
        .with_metadata("c_out", serde_json::json!(12.4));

    println!("\nWeight-walk edge:");
    println!("  {} --{}--> {}", e2.subject, e2.relation, e2.object);
    if let Some(meta) = &e2.metadata {
        println!("  layer={}, feature={}", meta["layer"], meta["feature"]);
        println!("  c_in={}, c_out={}", meta["c_in"], meta["c_out"]);
    }

    // ── Equality is triple-based ──
    let e3 = Edge::new("France", "capital-of", "Paris").with_confidence(0.50);
    println!("\nEquality (ignores confidence):");
    println!("  e1(c=0.89) == e3(c=0.50): {}", e1 == e3);

    // ── Compact JSON serialization ──
    let compact = core::edge::CompactEdge::from(&e1);
    let json = serde_json::to_string(&compact).unwrap();
    println!("\nCompact JSON: {json}");

    // Roundtrip
    let parsed: core::edge::CompactEdge = serde_json::from_str(&json).unwrap();
    let restored = Edge::from(parsed);
    println!(
        "  Roundtrip: {} --{}--> {} (c={:.2})",
        restored.subject, restored.relation, restored.object, restored.confidence
    );

    // ── Confidence clamping ──
    let clamped = Edge::new("A", "r", "B").with_confidence(1.5);
    println!("\nConfidence clamping: 1.5 → {}", clamped.confidence);

    // ── Source types ──
    println!("\nSource types:");
    for src in &[
        SourceType::Parametric,
        SourceType::Document,
        SourceType::Installed,
        SourceType::Wikidata,
        SourceType::Manual,
        SourceType::Unknown,
    ] {
        println!("  {:?} → \"{}\"", src, src.as_str());
    }

    println!("\n=== Done ===");
}
