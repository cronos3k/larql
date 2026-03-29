-- Circuit type distribution for a layer.
-- Classify every feature as identity/transform/projector/suppressor/inverter
-- based on cosine(gate, down).
--
-- For full classification, use: python scripts/circuit_classify.py --layer 26
--
-- Quick sample (10 features):

LET $results = [];
-- Check a few representative features manually:

LET $g0 = (SELECT vector FROM ONLY ffn_gate:L26_F67).vector;
LET $d0 = (SELECT vector FROM ONLY ffn_down:L26_F67).vector;
RETURN {
  feature: 67,
  gate: (SELECT top_token FROM ONLY ffn_gate:L26_F67).top_token,
  down: (SELECT top_token FROM ONLY ffn_down:L26_F67).top_token,
  cosine: vector::similarity::cosine($g0, $d0),
  type: IF vector::similarity::cosine($g0, $d0) > 0.5 THEN 'identity'
        ELSE IF vector::similarity::cosine($g0, $d0) > 0.2 THEN 'transform'
        ELSE IF vector::similarity::cosine($g0, $d0) > -0.2 THEN 'projector'
        ELSE IF vector::similarity::cosine($g0, $d0) > -0.5 THEN 'suppressor'
        ELSE 'inverter'
        END END END END
};

-- L26 distribution summary (from scripts/circuit_classify.py):
-- identity:     95  (0.9%)  — same concept, same form
-- transform:   487  (4.8%)  — same concept, different form
-- projector:  9064  (88.5%) — factual bridges
-- suppressor:  500  (4.9%)  — interference/gating
-- inverter:     94  (0.9%)  — negation circuits
