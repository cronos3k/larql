//! Demonstrate model architecture detection and configuration.
//!
//! Shows how larql-models reads config.json and produces architecture-specific
//! behavior: tensor keys, norm offsets, embed scaling, sliding window patterns.
//!
//! Run: cargo run -p larql-models --example architecture_demo

use larql_models::{detect_from_json, ModelArchitecture};

fn main() {
    println!("=== larql-models: Architecture Detection Demo ===\n");

    // ── Gemma 3 ──
    let gemma_config = serde_json::json!({
        "model_type": "gemma3",
        "text_config": {
            "model_type": "gemma3_text",
            "hidden_size": 2560,
            "num_hidden_layers": 34,
            "intermediate_size": 10240,
            "head_dim": 256,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "sliding_window": 1024
        }
    });

    let gemma = detect_from_json(&gemma_config);
    print_architecture(&*gemma);

    // ── Llama-style (generic) ──
    let llama_config = serde_json::json!({
        "model_type": "llama",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "intermediate_size": 11008,
        "head_dim": 128,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "rope_theta": 500000.0
    });

    let llama = detect_from_json(&llama_config);
    print_architecture(&*llama);

    // ── Tensor key comparison ──
    println!("=== Tensor Key Comparison (Layer 5) ===\n");
    println!("{:<25} {:<45} {}", "Key", "Gemma 3", "Generic/Llama");
    println!("{}", "-".repeat(100));

    let keys: Vec<(&str, Box<dyn Fn(&dyn ModelArchitecture) -> String>)> = vec![
        (
            "Q projection",
            Box::new(|a: &dyn ModelArchitecture| a.attn_q_key(5)),
        ),
        (
            "K projection",
            Box::new(|a: &dyn ModelArchitecture| a.attn_k_key(5)),
        ),
        (
            "V projection",
            Box::new(|a: &dyn ModelArchitecture| a.attn_v_key(5)),
        ),
        (
            "O projection",
            Box::new(|a: &dyn ModelArchitecture| a.attn_o_key(5)),
        ),
        (
            "Gate proj",
            Box::new(|a: &dyn ModelArchitecture| a.ffn_gate_key(5)),
        ),
        (
            "Up proj",
            Box::new(|a: &dyn ModelArchitecture| a.ffn_up_key(5)),
        ),
        (
            "Down proj",
            Box::new(|a: &dyn ModelArchitecture| a.ffn_down_key(5)),
        ),
        (
            "Input norm",
            Box::new(|a: &dyn ModelArchitecture| a.input_layernorm_key(5)),
        ),
    ];

    for (name, key_fn) in &keys {
        println!("{:<25} {:<45} {}", name, key_fn(&*gemma), key_fn(&*llama));
    }

    // ── Sliding window pattern ──
    println!("\n=== Gemma 3 Sliding Window Pattern ===\n");
    for layer in 0..12 {
        let attn_type = if gemma.is_sliding_window_layer(layer) {
            "sliding"
        } else {
            "FULL"
        };
        println!("  Layer {:2}: {}", layer, attn_type);
    }

    // ── Component constants ──
    println!("\n=== Vector Components ===\n");
    for comp in larql_models::ALL_COMPONENTS {
        println!("  {comp}");
    }
}

fn print_architecture(arch: &dyn ModelArchitecture) {
    let cfg = arch.config();
    println!("--- {} ---", arch.family());
    println!("  Layers:          {}", cfg.num_layers);
    println!("  Hidden size:     {}", cfg.hidden_size);
    println!("  Intermediate:    {}", cfg.intermediate_size);
    println!("  Head dim:        {}", cfg.head_dim);
    println!("  Q heads:         {}", cfg.num_q_heads);
    println!("  KV heads:        {}", cfg.num_kv_heads);
    println!("  RoPE base:       {:.0}", cfg.rope_base);
    println!("  Norm offset:     {}", arch.norm_weight_offset());
    println!("  Embed scale:     {:.2}", arch.embed_scale());
    println!("  Has post norms:  {}", arch.has_post_norms());
    println!("  Has QK norm:     {}", arch.attn_q_norm_key(0).is_some());
    println!("  Embed key:       {}", arch.embed_key());
    println!("  Final norm key:  {}", arch.final_norm_key());
    println!();
}
