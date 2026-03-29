use larql_models::{TopKEntry, VectorRecord};
use larql_surreal::{
    batch_insert_sql, completed_layers_sql, count_sql, mark_layer_done_sql, progress_table_sql,
    schema_sql, setup_sql, single_insert_sql,
};

fn sample_record(id: &str, layer: usize, feature: usize) -> VectorRecord {
    VectorRecord {
        id: id.to_string(),
        layer,
        feature,
        vector: vec![0.1, 0.2, 0.3],
        dim: 3,
        top_token: "hello".to_string(),
        top_token_id: 42,
        c_score: 0.95,
        top_k: vec![TopKEntry {
            token: "hello".to_string(),
            token_id: 42,
            logit: 2.75,
        }],
    }
}

// -------------------------------------------------------
// 1. setup_sql generates valid SQL with ns/db substituted
// -------------------------------------------------------

#[test]
fn setup_sql_substitutes_ns_and_db() {
    let sql = setup_sql("test_ns", "test_db");
    assert!(sql.contains("test_ns"), "namespace not found in setup SQL");
    assert!(sql.contains("test_db"), "database not found in setup SQL");
    assert!(sql.contains("DEFINE NAMESPACE"), "missing DEFINE NAMESPACE");
    assert!(sql.contains("DEFINE DATABASE"), "missing DEFINE DATABASE");
}

#[test]
fn setup_sql_no_raw_placeholders() {
    let sql = setup_sql("my_ns", "my_db");
    assert!(
        !sql.contains("{ns}"),
        "raw {{ns}} placeholder still present"
    );
    assert!(
        !sql.contains("{db}"),
        "raw {{db}} placeholder still present"
    );
}

// -------------------------------------------------------
// 2. schema_sql for a valid component returns SQL with table name and dimension
// -------------------------------------------------------

#[test]
fn schema_sql_valid_component() {
    let sql = schema_sql("ffn_down", 128).expect("should succeed for known component");
    assert!(sql.contains("ffn_down"), "table name not found");
    assert!(sql.contains("128"), "dimension not found");
    assert!(sql.contains("DEFINE TABLE"), "missing DEFINE TABLE");
    assert!(sql.contains("HNSW DIMENSION"), "missing HNSW DIMENSION");
}

#[test]
fn schema_sql_all_known_components() {
    for component in larql_models::ALL_COMPONENTS {
        let result = schema_sql(component, 64);
        assert!(
            result.is_ok(),
            "schema_sql should succeed for known component: {component}"
        );
        let sql = result.unwrap();
        assert!(sql.contains(component));
    }
}

// -------------------------------------------------------
// 3. schema_sql for an invalid component returns UnknownComponent error
// -------------------------------------------------------

#[test]
fn schema_sql_unknown_component_returns_error() {
    let result = schema_sql("nonexistent_component", 128);
    assert!(result.is_err(), "should fail for unknown component");
    let err = result.unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("unknown component"),
        "error message should mention unknown component, got: {msg}"
    );
}

// -------------------------------------------------------
// 4. progress_table_sql is non-empty
// -------------------------------------------------------

#[test]
fn progress_table_sql_is_nonempty() {
    let sql = progress_table_sql();
    assert!(!sql.is_empty(), "progress_table_sql should not be empty");
    assert!(
        sql.contains("load_progress"),
        "should reference load_progress table"
    );
    assert!(sql.contains("DEFINE TABLE"), "should contain DEFINE TABLE");
}

// -------------------------------------------------------
// 5. single_insert_sql generates CREATE statement with correct table and record id
// -------------------------------------------------------

#[test]
fn single_insert_sql_contains_table_and_id() {
    let record = sample_record("L0_F42", 0, 42);
    let sql = single_insert_sql("ffn_down", &record);
    assert!(
        sql.contains("CREATE ffn_down:L0_F42"),
        "should contain CREATE table:id, got: {sql}"
    );
    assert!(sql.contains("CONTENT"), "should contain CONTENT keyword");
    assert!(sql.ends_with(';'), "should end with semicolon");
}

#[test]
fn single_insert_sql_includes_record_fields() {
    let record = sample_record("L1_F10", 1, 10);
    let sql = single_insert_sql("attn_ov", &record);
    assert!(sql.contains("\"layer\":1"), "should include layer");
    assert!(sql.contains("\"feature\":10"), "should include feature");
    assert!(
        sql.contains("\"top_token\":\"hello\""),
        "should include top_token"
    );
    assert!(
        sql.contains("\"top_token_id\":42"),
        "should include top_token_id"
    );
}

// -------------------------------------------------------
// 6. batch_insert_sql wraps records in BEGIN/COMMIT TRANSACTION
// -------------------------------------------------------

#[test]
fn batch_insert_sql_transaction_wrapper() {
    let records = vec![sample_record("L0_F1", 0, 1), sample_record("L0_F2", 0, 2)];
    let sql = batch_insert_sql("ffn_up", &records);
    assert!(
        sql.starts_with("BEGIN TRANSACTION;"),
        "should start with BEGIN TRANSACTION"
    );
    assert!(
        sql.ends_with("COMMIT TRANSACTION;"),
        "should end with COMMIT TRANSACTION"
    );
}

#[test]
fn batch_insert_sql_contains_all_records() {
    let records = vec![
        sample_record("L0_F1", 0, 1),
        sample_record("L0_F2", 0, 2),
        sample_record("L1_F3", 1, 3),
    ];
    let sql = batch_insert_sql("embeddings", &records);
    assert!(
        sql.contains("CREATE embeddings:L0_F1"),
        "should contain first record"
    );
    assert!(
        sql.contains("CREATE embeddings:L0_F2"),
        "should contain second record"
    );
    assert!(
        sql.contains("CREATE embeddings:L1_F3"),
        "should contain third record"
    );
}

#[test]
fn batch_insert_sql_empty_records() {
    let sql = batch_insert_sql("ffn_gate", &[]);
    assert!(sql.contains("BEGIN TRANSACTION;"));
    assert!(sql.contains("COMMIT TRANSACTION;"));
}

// -------------------------------------------------------
// 7. mark_layer_done_sql includes table, layer, count
// -------------------------------------------------------

#[test]
fn mark_layer_done_sql_contains_fields() {
    let sql = mark_layer_done_sql("ffn_down", 5, 1000);
    assert!(sql.contains("ffn_down"), "should contain table name");
    assert!(sql.contains("5"), "should contain layer number");
    assert!(sql.contains("1000"), "should contain count");
    assert!(
        sql.contains("load_progress"),
        "should reference load_progress table"
    );
    assert!(sql.contains("CREATE"), "should be a CREATE statement");
}

#[test]
fn mark_layer_done_sql_includes_completed_true() {
    let sql = mark_layer_done_sql("attn_qk", 0, 500);
    assert!(
        sql.contains("\"completed\": true") || sql.contains("\"completed\":true"),
        "should mark completed as true"
    );
}

// -------------------------------------------------------
// 8. completed_layers_sql includes table name
// -------------------------------------------------------

#[test]
fn completed_layers_sql_includes_table() {
    let sql = completed_layers_sql("ffn_gate");
    assert!(sql.contains("ffn_gate"), "should include the table name");
    assert!(sql.contains("SELECT"), "should be a SELECT statement");
    assert!(
        sql.contains("load_progress"),
        "should query load_progress table"
    );
}

// -------------------------------------------------------
// 9. count_sql includes table name
// -------------------------------------------------------

#[test]
fn count_sql_includes_table() {
    let sql = count_sql("embeddings");
    assert!(sql.contains("embeddings"), "should include table name");
    assert!(sql.contains("SELECT"), "should be a SELECT statement");
    assert!(sql.contains("count()"), "should use count() function");
}
