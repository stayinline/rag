"""Tests for RAG pipeline v2 with reranking and query rewriting."""
import pytest
from unittest.mock import patch, MagicMock

from app.services.rag import (
    RAGSource,
    _build_where_filter,
    _weaviate_to_source,
    rerank_sources,
    build_context,
)


class TestRAGSourceV2:
    def test_with_document_type(self):
        source = RAGSource(
            chunk_id="c1",
            document_id="d1",
            document_title="Test Paper",
            section_path="Test/Abstract",
            page_start=None,
            page_end=None,
            score=0.9,
            content_preview="Abstract content",
            document_type="paper",
            section_type="abstract",
        )
        assert source.document_type == "paper"
        assert source.section_type == "abstract"

    def test_to_dict_includes_new_fields(self):
        source = RAGSource(
            chunk_id="c1",
            document_id="d1",
            document_title="Test",
            section_path="Test/Intro",
            page_start=1,
            page_end=2,
            score=0.8,
            content_preview="Content",
            document_type="paper",
            section_type="introduction",
        )
        d = source.to_dict()
        assert d["document_type"] == "paper"
        assert d["section_type"] == "introduction"

    def test_default_document_type(self):
        source = RAGSource(
            chunk_id="c1",
            document_id="d1",
            document_title="Test",
            section_path=None,
            page_start=None,
            page_end=None,
            score=0.5,
            content_preview="Content",
        )
        assert source.document_type == "general"
        assert source.section_type is None


class TestBuildWhereFilter:
    def test_without_kb_ids(self):
        from weaviate.classes.query import Filter
        where = _build_where_filter("org1", [])
        assert where is not None

    def test_with_kb_ids(self):
        where = _build_where_filter("org1", ["kb1", "kb2"])
        assert where is not None


class TestWeaviateToSource:
    def test_conversion(self):
        mock_obj = MagicMock()
        mock_obj.uuid = "test-uuid"
        mock_obj.properties = {
            "document_id": "doc1",
            "title": "Test Paper",
            "section_path": "Test/Abstract",
            "page_start": 1,
            "page_end": 5,
            "content": "Content preview",
            "document_type": "paper",
        }
        mock_obj.metadata = {"score": 0.85}

        source = _weaviate_to_source(mock_obj, 0.85)

        assert source.chunk_id == "test-uuid"
        assert source.document_id == "doc1"
        assert source.document_title == "Test Paper"
        assert source.score == 0.85
        assert source.document_type == "paper"


class TestRerankSources:
    def test_empty_sources(self):
        reranked = rerank_sources("query", [])
        assert reranked == []

    def test_reranking_changes_order(self):
        sources = [
            RAGSource("c1", "d1", "Doc1", None, None, None, 0.9, "random unrelated text here"),
            RAGSource("c2", "d2", "Doc2", None, None, None, 0.8, "machine learning is a subset of machine learning"),
        ]
        reranked = rerank_sources("machine learning", sources)

        # After reranking, the ML document should score higher
        assert reranked[0].document_title == "Doc2"

    def test_reranking_respects_top_n(self):
        from app.config import settings
        sources = [
            RAGSource(f"c{i}", f"d{i}", f"Doc{i}", None, None, None, 0.5, f"content {i}")
            for i in range(10)
        ]
        with patch("app.services.rag.settings") as mock_settings:
            mock_settings.reranker_top_n = 3
            reranked = rerank_sources("query", sources, top_n=3)
            assert len(reranked) == 3

    def test_reranking_updates_scores(self):
        sources = [
            RAGSource("c1", "d1", "Doc1", None, None, None, 0.9, "machine learning deep learning"),
            RAGSource("c2", "d2", "Doc2", None, None, None, 0.8, "cooking dinner"),
        ]
        reranked = rerank_sources("machine learning", sources)

        # Scores should have been updated
        assert reranked[0].score != 0.9  # Score updated by reranker


class TestBuildContextV2:
    def test_includes_document_type(self):
        source = RAGSource(
            chunk_id="c1", document_id="d1", document_title="Paper",
            section_path="Paper/Abstract", page_start=None, page_end=None,
            score=0.9, content_preview="Abstract text",
            document_type="paper",
        )
        context, citations = build_context([source])
        assert "[paper]" in context

    def test_general_documents_no_type_label(self):
        source = RAGSource(
            chunk_id="c1", document_id="d1", document_title="Doc",
            section_path="Doc/Section", page_start=None, page_end=None,
            score=0.8, content_preview="General content",
            document_type="general",
        )
        context, citations = build_context([source])
        assert "[general]" not in context

    def test_multiple_sources(self):
        sources = [
            RAGSource("c1", "d1", "Paper1", None, None, None, 0.9, "Content 1", "paper"),
            RAGSource("c2", "d2", "Paper2", None, None, None, 0.8, "Content 2", "paper"),
        ]
        context, citations = build_context(sources)
        assert "[1]" in context
        assert "[2]" in context
        assert len(citations) == 2
