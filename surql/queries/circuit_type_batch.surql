-- Classify all features at a layer as amplifier/rotator/inverter.
-- Computes cosine(gate, down) for each feature.
--
-- Amplifier:  cosine > 0.5   (reads X, writes X back)
-- Rotator:    -0.5 < cosine < 0.5  (reads X, writes something different)
-- Inverter:   cosine < -0.5  (reads X, writes opposite)

-- Single feature test:
LET $g = (SELECT vector FROM ONLY ffn_gate:L26_F67).vector;
LET $d = (SELECT vector FROM ONLY ffn_down:L26_F67).vector;
RETURN {
  feature: 67,
  gate_token: (SELECT top_token FROM ONLY ffn_gate:L26_F67).top_token,
  down_token: (SELECT top_token FROM ONLY ffn_down:L26_F67).top_token,
  cosine: vector::similarity::cosine($g, $d)
};

-- Batch version (top 50 amplifiers at L26):
-- SELECT
--   id, feature, top_token, c_score,
--   vector::similarity::cosine(
--     vector,
--     (SELECT vector FROM ONLY type::thing('ffn_down',
--       string::concat('L26_F', <string> $parent.feature))).vector
--   ) AS gate_down_cos
-- FROM ffn_gate
-- WHERE layer = 26
-- ORDER BY gate_down_cos DESC
-- LIMIT 50;
