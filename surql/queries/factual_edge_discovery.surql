-- Factual edge discovery: gate KNN → down KNN → edge
-- No forward pass. No model loaded. Pure SQL on extracted weights.
--
-- Method:
--   1. Gate KNN against embeddings → what entity triggers the feature
--   2. Down KNN against embeddings → what entity/concept it produces
--   3. Pair them → factual edge
--
-- Example: F5040 at L26
--   Gate: nearest to Toulouse embedding
--   Down KNN: French, France, Frenchman, フランス
--   Edge: Toulouse → French/France

-- Step 1: Find gate's input entity
LET $g = (SELECT vector FROM ONLY ffn_gate:L26_F5040).vector;
SELECT id, top_token, vector::distance::knn() AS dist
FROM embeddings
WHERE vector <|5, COSINE|> $g
ORDER BY dist;

-- Step 2: Find down's output entity via embedding KNN
LET $d = (SELECT vector FROM ONLY ffn_down:L26_F5040).vector;
SELECT id, top_token, vector::distance::knn() AS dist
FROM embeddings
WHERE vector <|10, COSINE|> $d
ORDER BY dist;

-- More examples:
--   F943:  € → euros, EU, 欧盟, Spain, EUR, France
--   F918:  Rome → Roman, Roma, Romano
--   F2230: Dutch → Netherlands, Amsterdam, 荷兰, Nederlandse

-- To discover edges for an entity, chain:
-- 1. Entity embedding → gate KNN (find features that fire for this entity)
-- 2. For each matched feature → down KNN (find what it writes)
-- 3. Each (entity, feature_relation, output_entity) triple is an edge
