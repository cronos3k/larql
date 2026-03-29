-- Attention→FFN routing: which gate features does head 0 at L26 connect to?
-- This traces the signal path from attention output to FFN input.

LET $v = (SELECT vector FROM ONLY attn_ov:L26_H0).vector;
SELECT id, feature, top_token, c_score, vector::distance::knn() AS dist
FROM ffn_gate
WHERE layer = 26 AND vector <|10, COSINE|> $v
ORDER BY dist;
