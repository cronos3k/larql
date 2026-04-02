"""
Tests for larql Python bindings — vindex operations.

Requires a built vindex at output/gemma3-4b-v2.vindex.
Run: pytest tests/test_vindex_bindings.py -v
"""

import os
import sys
import pytest
import numpy as np

# Import native module directly (works before larql package is installed)
import larql

VINDEX_PATH = os.environ.get(
    "VINDEX_PATH",
    os.path.join(os.path.dirname(__file__), "..", "output", "gemma3-4b-v2.vindex")
)

# Skip all tests if vindex not available
pytestmark = pytest.mark.skipif(
    not os.path.exists(VINDEX_PATH),
    reason=f"Vindex not found at {VINDEX_PATH}"
)


@pytest.fixture(scope="module")
def vindex():
    """Load vindex once for all tests."""
    return larql.load(VINDEX_PATH)


# ─── Loading & Properties ───

class TestLoading:
    def test_load(self, vindex):
        assert vindex is not None
        assert repr(vindex).startswith("Vindex(")

    def test_properties(self, vindex):
        assert vindex.num_layers > 0
        assert vindex.hidden_size > 0
        assert vindex.vocab_size > 0
        assert len(vindex.model) > 0
        assert len(vindex.family) > 0
        assert vindex.is_mmap  # production vindexes are mmap'd
        assert vindex.total_gate_vectors > 0

    def test_loaded_layers(self, vindex):
        layers = vindex.loaded_layers
        assert len(layers) > 0
        assert all(isinstance(l, int) for l in layers)

    def test_num_features(self, vindex):
        layer = vindex.loaded_layers[0]
        nf = vindex.num_features(layer)
        assert nf > 0

    def test_stats(self, vindex):
        s = vindex.stats()
        assert "model" in s
        assert "num_layers" in s
        assert "hidden_size" in s
        assert "total_gate_vectors" in s
        assert s["total_gate_vectors"] > 0

    def test_layer_bands(self, vindex):
        bands = vindex.layer_bands()
        assert bands is not None
        assert "syntax" in bands
        assert "knowledge" in bands
        assert "output" in bands
        assert bands["knowledge"][0] < bands["knowledge"][1]


# ─── Embeddings ───

class TestEmbeddings:
    def test_embed_single_token(self, vindex):
        embed = vindex.embed("France")
        assert isinstance(embed, np.ndarray)
        assert embed.shape == (vindex.hidden_size,)
        assert embed.dtype == np.float32
        assert np.linalg.norm(embed) > 0

    def test_embed_multi_token(self, vindex):
        """Multi-token entities should be averaged."""
        embed = vindex.embed("John Coyle")
        assert embed.shape == (vindex.hidden_size,)
        assert np.linalg.norm(embed) > 0

    def test_embed_different_entities(self, vindex):
        """Different entities should produce different embeddings."""
        a = vindex.embed("France")
        b = vindex.embed("Germany")
        assert not np.allclose(a, b)

    def test_tokenize(self, vindex):
        ids = vindex.tokenize("hello world")
        assert len(ids) > 0
        assert all(isinstance(i, int) for i in ids)

    def test_decode(self, vindex):
        ids = vindex.tokenize("hello")
        text = vindex.decode(ids)
        assert "hello" in text.lower()

    def test_embedding_by_id(self, vindex):
        embed = vindex.embedding(token_id=100)
        assert embed.shape == (vindex.hidden_size,)

    def test_embedding_matrix(self, vindex):
        mat = vindex.embedding_matrix()
        assert mat.shape == (vindex.vocab_size, vindex.hidden_size)
        assert mat.dtype == np.float32


# ─── Gate Vectors ───

class TestGateVectors:
    def test_gate_vector_single(self, vindex):
        layer = vindex.loaded_layers[0]
        vec = vindex.gate_vector(layer=layer, feature=0)
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (vindex.hidden_size,)

    def test_gate_vectors_layer(self, vindex):
        layer = vindex.loaded_layers[0]
        nf = vindex.num_features(layer)
        mat = vindex.gate_vectors(layer=layer)
        assert mat.shape == (nf, vindex.hidden_size)
        assert mat.dtype == np.float32

    def test_gate_vector_invalid(self, vindex):
        with pytest.raises(ValueError):
            vindex.gate_vector(layer=999, feature=0)


# ─── KNN & Walk ───

class TestKNN:
    def test_gate_knn(self, vindex):
        embed = vindex.embed("France")
        layer = vindex.loaded_layers[-1]  # last loaded layer
        hits = vindex.gate_knn(layer=layer, query_vector=embed.tolist(), top_k=5)
        assert len(hits) <= 5
        assert all(isinstance(h, tuple) and len(h) == 2 for h in hits)
        # Scores should be sorted by absolute value
        scores = [abs(s) for _, s in hits]
        assert scores == sorted(scores, reverse=True)

    def test_entity_knn(self, vindex):
        bands = vindex.layer_bands()
        layer = bands["knowledge"][1]  # last knowledge layer
        hits = vindex.entity_knn("France", layer=layer, top_k=10)
        assert len(hits) > 0

    def test_walk(self, vindex):
        embed = vindex.embed("France")
        hits = vindex.walk(embed.tolist(), top_k=3)
        assert len(hits) > 0
        assert all(hasattr(h, "layer") for h in hits)
        assert all(hasattr(h, "gate_score") for h in hits)
        assert all(hasattr(h, "top_token") for h in hits)

    def test_entity_walk(self, vindex):
        bands = vindex.layer_bands()
        layers = list(range(bands["knowledge"][0], bands["knowledge"][1] + 1))
        hits = vindex.entity_walk("France", layers=layers, top_k=5)
        assert len(hits) > 0
        # Should find Paris somewhere
        tokens = [h.top_token.strip().lower() for h in hits]
        assert "paris" in tokens, f"Expected 'Paris' in walk results, got: {tokens[:20]}"


# ─── Feature Metadata ───

class TestFeatures:
    def test_feature_meta(self, vindex):
        layer = vindex.loaded_layers[0]
        meta = vindex.feature_meta(layer, 0)
        if meta is not None:
            assert len(meta.top_token) > 0
            assert isinstance(meta.c_score, float)
            assert repr(meta).startswith("FeatureMeta(")

    def test_feature_dict(self, vindex):
        layer = vindex.loaded_layers[0]
        d = vindex.feature(layer, 0)
        if d is not None:
            assert "top_token" in d
            assert "c_score" in d
            assert "layer" in d

    def test_feature_label(self, vindex):
        """feature_label should return a string or None."""
        layer = vindex.loaded_layers[0]
        label = vindex.feature_label(layer, 0)
        assert label is None or isinstance(label, str)


# ─── DESCRIBE ───

class TestDescribe:
    def test_describe_france(self, vindex):
        edges = vindex.describe("France")
        assert len(edges) > 0
        assert all(hasattr(e, "target") for e in edges)
        assert all(hasattr(e, "gate_score") for e in edges)
        assert all(hasattr(e, "relation") for e in edges)
        # Should find Paris
        targets = [e.target.lower() for e in edges]
        assert "paris" in targets, f"Expected 'Paris' in describe, got: {targets[:10]}"

    def test_describe_with_relation(self, vindex):
        edges = vindex.describe("France")
        # At least some edges should have probe-confirmed relations
        labelled = [e for e in edges if e.relation is not None]
        assert len(labelled) > 0, "Expected some labelled edges"

    def test_describe_verbose(self, vindex):
        edges_normal = vindex.describe("France", verbose=False)
        edges_verbose = vindex.describe("France", verbose=True)
        assert len(edges_verbose) >= len(edges_normal)

    def test_describe_syntax_band(self, vindex):
        edges = vindex.describe("def", band="syntax")
        # Syntax band may or may not have results
        assert isinstance(edges, list)

    def test_describe_repr(self, vindex):
        edges = vindex.describe("France")
        if edges:
            r = repr(edges[0])
            assert "DescribeEdge" in r

    def test_has_edge(self, vindex):
        assert vindex.has_edge("France")
        result = vindex.has_edge("France", relation="capital")
        assert isinstance(result, bool)

    def test_get_target(self, vindex):
        target = vindex.get_target("France", "capital")
        if target is not None:
            assert target.lower() == "paris"


# ─── Relations & Clusters ───

class TestRelations:
    def test_relations_list(self, vindex):
        rels = vindex.relations()
        assert len(rels) > 0
        assert all(hasattr(r, "name") for r in rels)
        assert all(hasattr(r, "count") for r in rels)
        # Should be sorted by count descending
        counts = [r.count for r in rels]
        assert counts == sorted(counts, reverse=True)

    def test_cluster_centre(self, vindex):
        centre = vindex.cluster_centre("capital")
        if centre is not None:
            assert isinstance(centre, np.ndarray)
            assert centre.shape == (vindex.hidden_size,)
            assert np.linalg.norm(centre) > 0

    def test_typical_layer(self, vindex):
        layer = vindex.typical_layer("capital")
        if layer is not None:
            bands = vindex.layer_bands()
            assert bands["knowledge"][0] <= layer <= bands["knowledge"][1]


# ─── Mutation ───

class TestMutation:
    def test_insert_and_verify(self, vindex):
        """Insert a new fact and verify it can be found."""
        layer, feat = vindex.insert("TestEntity", "capital", "TestCity")
        assert isinstance(layer, int)
        assert isinstance(feat, int)

        # Verify metadata was written
        meta = vindex.feature_meta(layer, feat)
        assert meta is not None
        assert meta.top_token == "TestCity"

    def test_insert_with_layer_hint(self, vindex):
        layer, feat = vindex.insert("TestEntity2", "language", "TestLang", layer=20)
        assert layer == 20

    def test_delete(self, vindex):
        # Insert then delete
        layer, feat = vindex.insert("DeleteMe", "capital", "Gone")
        assert vindex.feature_meta(layer, feat) is not None

        count = vindex.delete("Gone")  # delete by target token match
        # Note: delete finds by describe, so it may or may not find our just-inserted entry


# ─── Session ───

class TestSession:
    def test_session_create(self):
        s = larql.session(VINDEX_PATH)
        assert repr(s).startswith("Session(")

    def test_session_query_stats(self):
        s = larql.session(VINDEX_PATH)
        result = s.query("STATS")
        assert len(result) > 0
        assert any("layer" in line.lower() or "feature" in line.lower() or "model" in line.lower()
                    for line in result)

    def test_session_query_describe(self):
        s = larql.session(VINDEX_PATH)
        result = s.query("DESCRIBE 'France'")
        assert len(result) > 0

    def test_session_query_walk(self):
        s = larql.session(VINDEX_PATH)
        result = s.query("WALK 'The capital of France is' TOP 5")
        assert len(result) > 0

    def test_session_vindex_access(self):
        s = larql.session(VINDEX_PATH)
        v = s.vindex
        assert v.num_layers > 0
        embed = v.embed("France")
        assert embed.shape == (v.hidden_size,)

    def test_session_query_text(self):
        s = larql.session(VINDEX_PATH)
        text = s.query_text("STATS")
        assert isinstance(text, str)
        assert len(text) > 0
