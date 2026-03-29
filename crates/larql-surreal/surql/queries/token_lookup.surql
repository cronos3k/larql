-- Look up a token's embedding by name.
-- Useful for finding the token ID before running other queries.

SELECT id, token, c_score FROM embeddings WHERE token = 'France';

-- Or search by prefix (if supported):
-- SELECT id, token FROM embeddings WHERE token CONTAINS 'Fran' LIMIT 20;
