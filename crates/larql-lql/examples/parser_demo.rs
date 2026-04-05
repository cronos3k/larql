//! LQL Parser Demo — parse every statement type from the spec v0.3 and display the AST.
//!
//! Run: cargo run -p larql-lql --example parser_demo

use larql_lql::parse;

fn main() {
    println!("=== LQL Parser Demo (spec v0.3) ===\n");

    // ── Lifecycle Statements ──
    section("Lifecycle");

    demo(
        "EXTRACT (minimal)",
        r#"EXTRACT MODEL "google/gemma-3-4b-it" INTO "gemma3-4b.vindex";"#,
    );

    demo(
        "EXTRACT (with weights)",
        r#"EXTRACT MODEL "google/gemma-3-4b-it" INTO "gemma3-4b.vindex" WITH ALL;"#,
    );

    demo(
        "EXTRACT (full)",
        r#"EXTRACT MODEL "google/gemma-3-4b-it" INTO "gemma3-4b.vindex" COMPONENTS FFN_GATE, FFN_DOWN, FFN_UP, EMBEDDINGS LAYERS 0-33 WITH ALL;"#,
    );

    demo(
        "COMPILE (safetensors)",
        r#"COMPILE CURRENT INTO MODEL "gemma3-4b-edited/" FORMAT safetensors;"#,
    );

    demo(
        "COMPILE (gguf, from path)",
        r#"COMPILE "gemma3.vindex" INTO MODEL "out/" FORMAT gguf;"#,
    );

    demo(
        "DIFF (path vs CURRENT)",
        r#"DIFF "gemma3-4b.vindex" CURRENT;"#,
    );

    demo(
        "DIFF (with RELATION + LIMIT)",
        r#"DIFF "a.vindex" "b.vindex" RELATION "capital" LIMIT 20;"#,
    );

    demo("USE (vindex)", r#"USE "gemma3-4b.vindex";"#);
    demo("USE MODEL", r#"USE MODEL "google/gemma-3-4b-it";"#);
    demo("USE MODEL AUTO_EXTRACT", r#"USE MODEL "google/gemma-3-4b-it" AUTO_EXTRACT;"#);

    // ── Query Statements ──
    section("Query");

    demo("WALK (minimal)", r#"WALK "The capital of France is";"#);
    demo(
        "WALK (full options)",
        r#"WALK "The capital of France is" TOP 5 LAYERS 25-33 MODE hybrid COMPARE;"#,
    );

    demo("SELECT (star)", "SELECT * FROM EDGES LIMIT 5;");
    demo(
        "SELECT (fields + WHERE + ORDER + LIMIT)",
        r#"SELECT entity, relation, target, confidence FROM EDGES WHERE entity = "France" ORDER BY confidence DESC LIMIT 10;"#,
    );
    demo(
        "SELECT (NEAREST TO)",
        r#"SELECT entity, target, distance FROM EDGES NEAREST TO "Mozart" AT LAYER 26 LIMIT 20;"#,
    );

    // ── DESCRIBE with layer bands ──
    section("DESCRIBE (Layer Bands)");

    demo("DESCRIBE (default = brief)", r#"DESCRIBE "France";"#);
    demo("DESCRIBE VERBOSE", r#"DESCRIBE "France" VERBOSE;"#);
    demo("DESCRIBE RAW", r#"DESCRIBE "France" RAW;"#);
    demo("DESCRIBE SYNTAX (L0-13)", r#"DESCRIBE "def" SYNTAX;"#);
    demo("DESCRIBE KNOWLEDGE (L14-27)", r#"DESCRIBE "France" KNOWLEDGE;"#);
    demo("DESCRIBE OUTPUT (L28-33)", r#"DESCRIBE "France" OUTPUT;"#);
    demo("DESCRIBE ALL LAYERS", r#"DESCRIBE "France" ALL LAYERS;"#);
    demo("DESCRIBE AT LAYER", r#"DESCRIBE "Mozart" AT LAYER 26;"#);
    demo("DESCRIBE RELATIONS ONLY", r#"DESCRIBE "France" RELATIONS ONLY;"#);
    demo(
        "DESCRIBE band + RELATIONS ONLY",
        r#"DESCRIBE "France" KNOWLEDGE RELATIONS ONLY;"#,
    );

    // ── EXPLAIN ──
    section("Explain");

    demo("EXPLAIN WALK", r#"EXPLAIN WALK "The capital of France is";"#);
    demo(
        "EXPLAIN WALK (with options)",
        r#"EXPLAIN WALK "prompt" LAYERS 24-33 TOP 3 VERBOSE;"#,
    );
    demo("EXPLAIN INFER", r#"EXPLAIN INFER "The capital of France is" TOP 5;"#);

    // ── Inference Statements ──
    section("Inference");

    demo("INFER (minimal)", r#"INFER "The capital of France is" TOP 5;"#);
    demo("INFER (with compare)", r#"INFER "The capital of France is" TOP 5 COMPARE;"#);

    // ── Mutation Statements ──
    section("Mutation");

    demo(
        "INSERT (minimal)",
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("John Coyle", "lives-in", "Colchester");"#,
    );
    demo(
        "INSERT (with layer + confidence)",
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("John", "occupation", "engineer") AT LAYER 26 CONFIDENCE 0.8;"#,
    );
    demo("DELETE", r#"DELETE FROM EDGES WHERE entity = "John Coyle" AND relation = "lives-in";"#);
    demo(
        "UPDATE",
        r#"UPDATE EDGES SET target = "London" WHERE entity = "John Coyle" AND relation = "lives-in";"#,
    );
    demo(
        "UPDATE (multiple SET)",
        r#"UPDATE EDGES SET target = "London", confidence = 0.9 WHERE entity = "John Coyle";"#,
    );
    demo("MERGE (minimal)", r#"MERGE "medical-knowledge.vindex";"#);
    demo(
        "MERGE (full)",
        r#"MERGE "medical-knowledge.vindex" INTO "gemma3-4b.vindex" ON CONFLICT HIGHEST_CONFIDENCE;"#,
    );

    // ── Introspection ──
    section("Introspection");

    demo("SHOW RELATIONS", "SHOW RELATIONS;");
    demo("SHOW RELATIONS VERBOSE", "SHOW RELATIONS VERBOSE;");
    demo("SHOW RELATIONS RAW", "SHOW RELATIONS RAW;");
    demo("SHOW RELATIONS WITH EXAMPLES", "SHOW RELATIONS WITH EXAMPLES;");
    demo("SHOW RELATIONS AT LAYER", "SHOW RELATIONS AT LAYER 26;");
    demo("SHOW LAYERS", "SHOW LAYERS;");
    demo("SHOW LAYERS (range)", "SHOW LAYERS RANGE 0-10;");
    demo("SHOW LAYERS (bare range)", "SHOW LAYERS 0-10;");
    demo("SHOW FEATURES", "SHOW FEATURES 26;");
    demo("SHOW MODELS", "SHOW MODELS;");
    demo("STATS", "STATS;");

    // ── Patches ──
    section("Patches");

    demo("BEGIN PATCH", r#"BEGIN PATCH "medical-knowledge.vlp";"#);
    demo("SAVE PATCH", "SAVE PATCH;");
    demo("APPLY PATCH", r#"APPLY PATCH "medical-knowledge.vlp";"#);
    demo("SHOW PATCHES", "SHOW PATCHES;");
    demo("REMOVE PATCH", r#"REMOVE PATCH "medical-knowledge.vlp";"#);
    demo(
        "DIFF INTO PATCH",
        r#"DIFF "a.vindex" "b.vindex" INTO PATCH "changes.vlp";"#,
    );

    // ── Pipe Operator ──
    section("Pipe Operator");

    demo(
        "WALK |> EXPLAIN",
        r#"WALK "The capital of France is" TOP 5 |> EXPLAIN WALK "The capital of France is";"#,
    );

    // ── Demo Script (spec v0.3) ──
    section("Full Demo Script (spec v0.3)");

    let demo_stmts = vec![
        // ACT 1: DECOMPILE
        r#"EXTRACT MODEL "google/gemma-3-4b-it" INTO "gemma3-4b.vindex" WITH ALL;"#,
        r#"USE "gemma3-4b.vindex";"#,
        "STATS;",
        // ACT 2: INSPECT
        "SHOW RELATIONS WITH EXAMPLES;",
        r#"DESCRIBE "France";"#,
        r#"DESCRIBE "Einstein";"#,
        r#"DESCRIBE "def" SYNTAX;"#,
        // ACT 3: WALK + INFER
        r#"WALK "France" TOP 10;"#,
        r#"EXPLAIN WALK "The capital of France is";"#,
        r#"INFER "The capital of France is" TOP 5 COMPARE;"#,
        // ACT 4: EDIT
        r#"DESCRIBE "John Coyle";"#,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("John Coyle", "lives-in", "Colchester");"#,
        r#"DESCRIBE "John Coyle";"#,
        // ACT 5: PATCH
        r#"BEGIN PATCH "medical.vlp";"#,
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("aspirin", "treats", "headache");"#,
        "SAVE PATCH;",
        r#"APPLY PATCH "medical.vlp";"#,
        "SHOW PATCHES;",
        // ACT 6: RECOMPILE
        r#"DIFF "gemma3-4b.vindex" CURRENT;"#,
        r#"DIFF "gemma3-4b.vindex" CURRENT INTO PATCH "changes.vlp";"#,
        r#"COMPILE CURRENT INTO MODEL "gemma3-4b-edited/" FORMAT safetensors;"#,
    ];

    let mut ok = 0;
    let mut fail = 0;
    for (i, input) in demo_stmts.iter().enumerate() {
        match parse(input) {
            Ok(_) => {
                println!("  {:2}. OK   {}", i + 1, truncate(input, 70));
                ok += 1;
            }
            Err(e) => {
                println!("  {:2}. FAIL {} — {}", i + 1, truncate(input, 50), e);
                fail += 1;
            }
        }
    }
    println!("\n  {ok}/{} statements parsed successfully.", ok + fail);

    println!("\n=== Done ===");
}

fn section(name: &str) {
    println!("\n── {} ──\n", name);
}

fn demo(label: &str, input: &str) {
    match parse(input) {
        Ok(stmt) => {
            println!("  {label}:");
            println!("    Input: {}", truncate(input, 80));
            println!("    AST:   {:?}", stmt);
            println!();
        }
        Err(e) => {
            println!("  {label}: PARSE ERROR");
            println!("    Input: {}", truncate(input, 80));
            println!("    Error: {e}");
            println!();
        }
    }
}

fn truncate(s: &str, max: usize) -> String {
    let s = s.replace('\n', " ");
    if s.len() > max {
        format!("{}...", &s[..max])
    } else {
        s
    }
}
