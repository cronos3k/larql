-- What does a specific FFN feature produce?
-- Replace L26_F#### with the feature ID from entity_to_ffn_gate results.
--
-- If the gate fires for "France" and the down produces "Paris",
-- you've found a factual edge from pure database queries.

SELECT id, top_token, c_score, top_k FROM ffn_down WHERE id = ffn_down:L26_F67;
