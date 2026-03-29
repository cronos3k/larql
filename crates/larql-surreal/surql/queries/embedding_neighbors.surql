-- Find nearest neighbors of a token in embedding space.
-- Shows all language variants, case variants, and morphological forms.
--
-- Usage: replace T7001 with the token ID you want to explore.
-- To find a token ID: SELECT id, token FROM embeddings WHERE token = 'France';

LET $v = (SELECT vector FROM ONLY embeddings:T7001).vector;
SELECT id, top_token, vector::distance::knn() AS dist
FROM embeddings
WHERE vector <|10, COSINE|> $v
ORDER BY dist;
