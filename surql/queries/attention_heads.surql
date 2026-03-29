-- What do the attention heads at layer 26 route?
-- Each head's top_token shows what it amplifies most.

SELECT id, layer, feature AS head, top_token, c_score
FROM attn_ov
WHERE layer = 26;
