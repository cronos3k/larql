-- Which FFN gate features at layer 26 fire for France's embedding?
-- This discovers which FFN features are activated by an entity.
--
-- The gate feature's top_token tells you what activates it.
-- Then look up the paired down vector to see what it produces.

LET $v = (SELECT vector FROM ONLY embeddings:T7001).vector;
SELECT id, feature, top_token, c_score, vector::distance::knn() AS dist
FROM ffn_gate
WHERE layer = 26 AND vector <|10, COSINE|> $v
ORDER BY dist;
