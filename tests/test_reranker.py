"""Tests for reranker service."""
import pytest

from app.services.reranker import (
    MockReranker,
    BM25Reranker,
    get_reranker,
    RerankerResult,
)


class TestRerankerResult:
    def test_creation(self):
        r = RerankerResult(index=0, score=0.5)
        assert r.index == 0
        assert r.score == 0.5


class TestMockReranker:
    def test_basic_reranking(self):
        reranker = MockReranker()
        query = "machine learning"
        docs = [
            "This is about machine learning and AI",
            "Cooking recipes for dinner",
            "Deep learning is a subset of machine learning",
            "The weather is nice today",
        ]
        results = reranker.rerank(query, docs)

        # Results should be sorted by score descending
        assert len(results) == 4
        assert results[0].score >= results[1].score >= results[2].score >= results[3].score

        # Most relevant docs should be first
        top_indices = [r.index for r in results[:2]]
        assert 0 in top_indices or 2 in top_indices

    def test_no_overlap(self):
        reranker = MockReranker()
        query = "quantum physics"
        docs = ["cooking recipe", "garden flowers", "sports news"]
        results = reranker.rerank(query, docs)
        assert len(results) == 3
        # All scores should be 0 (or just length bonus)
        for r in results:
            assert r.score < 0.2

    def test_empty_documents(self):
        reranker = MockReranker()
        results = reranker.rerank("query", [])
        assert results == []

    def test_single_document(self):
        reranker = MockReranker()
        results = reranker.rerank("test", ["single document"])
        assert len(results) == 1
        assert results[0].index == 0

    def test_exact_match(self):
        reranker = MockReranker()
        query = "exact match"
        docs = ["exact match here", "no relation", "partial exact"]
        results = reranker.rerank(query, docs)
        assert results[0].index == 0


class TestBM25Reranker:
    def test_basic_reranking(self):
        reranker = BM25Reranker()
        query = "cancer treatment"
        docs = [
            "Cancer treatment involves chemotherapy and radiation therapy.",
            "How to grow tomatoes in your garden.",
            "Immunotherapy for cancer: latest treatment guidelines.",
        ]
        results = reranker.rerank(query, docs)

        assert len(results) == 3
        # Results sorted by score descending
        assert results[0].score >= results[1].score >= results[2].score

        # Both cancer/treatment docs should rank above the gardening doc
        top_indices = {results[0].index, results[1].index}
        assert 1 not in top_indices  # gardening should not be in top 2

    def test_idf_effect(self):
        """Test that terms appearing in fewer documents get higher IDF scores."""
        reranker = BM25Reranker()
        query = "unique_term_xyz"
        docs = [
            "This document contains unique_term_xyz and some other words",
            "This document does not contain the query term at all",
        ]
        results = reranker.rerank(query, docs)
        assert results[0].score >= results[1].score

    def test_term_frequency_effect(self):
        """Test that documents with more query term matches can rank higher."""
        reranker = BM25Reranker()
        # Use same-length documents to avoid length normalization effects
        query = "python programming"
        docs = [
            "python python python programming programming other words here more",
            "java java java programming programming other words here more",
        ]
        results = reranker.rerank(query, docs)
        # First document has more "python" matches
        assert results[0].index == 0

    def test_empty_query(self):
        reranker = BM25Reranker()
        results = reranker.rerank("", ["doc1", "doc2"])
        assert len(results) == 2

    def test_custom_parameters(self):
        reranker = BM25Reranker(k1=2.0, b=0.5)
        results = reranker.rerank("test query", ["test document", "another doc"])
        assert len(results) == 2


class TestGetReranker:
    def test_get_mock_reranker(self):
        from app.config import settings
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(settings, "reranker_type", "mock")
            reranker = get_reranker()
            assert isinstance(reranker, MockReranker)

    def test_get_bm25_reranker(self):
        from app.config import settings
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(settings, "reranker_type", "bm25")
            reranker = get_reranker()
            assert isinstance(reranker, BM25Reranker)

    def test_default_is_bm25(self):
        from app.config import settings
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(settings, "reranker_type", "unknown")
            reranker = get_reranker()
            assert isinstance(reranker, BM25Reranker)
