-- Find similar features across layers.
-- Does the "France" feature at L26 have analogues at other layers?

LET $v = (SELECT vector FROM ONLY ffn_gate:L26_F67).vector;
SELECT id, layer, feature, top_token, c_score, vector::distance::knn() AS dist
FROM ffn_gate
WHERE vector <|10, COSINE|> $v
ORDER BY dist;
