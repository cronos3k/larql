-- Classify a feature as amplifier, inverter, or rotator.
--
-- Cosine similarity between gate and down vectors:
--   > 0.8  = amplifier (reads X, writes X back stronger)
--   < -0.8 = inverter (reads X, writes opposite)
--   middle = rotator (reads X, writes something different)
--
-- Replace L26_F67 with the feature you want to classify.

LET $g = (SELECT vector FROM ONLY ffn_gate:L26_F67).vector;
LET $d = (SELECT vector FROM ONLY ffn_down:L26_F67).vector;
RETURN vector::similarity::cosine($g, $d);
