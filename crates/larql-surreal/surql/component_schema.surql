-- Schema for a vector component table
-- Parameters: $table, $dimension (substituted by the loader)
-- All component tables share this schema matching VectorRecord

DEFINE TABLE IF NOT EXISTS {table} SCHEMAFULL;

-- Core fields
DEFINE FIELD IF NOT EXISTS layer       ON {table} TYPE int;
DEFINE FIELD IF NOT EXISTS feature     ON {table} TYPE int;
DEFINE FIELD IF NOT EXISTS vector      ON {table} TYPE array<float>;
DEFINE FIELD IF NOT EXISTS top_token   ON {table} TYPE string;
DEFINE FIELD IF NOT EXISTS top_token_id ON {table} TYPE int;
DEFINE FIELD IF NOT EXISTS c_score     ON {table} TYPE float;

-- Top-k token metadata (nested objects)
DEFINE FIELD IF NOT EXISTS top_k          ON {table} TYPE array<object>;
DEFINE FIELD IF NOT EXISTS top_k[*].token    ON {table} TYPE string;
DEFINE FIELD IF NOT EXISTS top_k[*].token_id ON {table} TYPE int;
DEFINE FIELD IF NOT EXISTS top_k[*].logit    ON {table} TYPE float;

-- Indexes
DEFINE INDEX IF NOT EXISTS idx_layer     ON {table} FIELDS layer;
DEFINE INDEX IF NOT EXISTS idx_top_token ON {table} FIELDS top_token;
DEFINE INDEX IF NOT EXISTS idx_c_score   ON {table} FIELDS c_score;
DEFINE INDEX IF NOT EXISTS idx_vec       ON {table} FIELDS vector
    HNSW DIMENSION {dimension} DIST COSINE;
