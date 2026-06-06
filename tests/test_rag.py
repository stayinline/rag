"""Tests for RAG pipeline."""
from unittest.mock import MagicMock, patch
import uuid

from app.services.rag import (
    RAGSource,
    _clear_rag_caches,
    hybrid_search,
    _retrieve_plan_sources,
    retrieve_sources,
    rerank_sources,
    build_context,
    assemble_context_and_generate,
)
from app.services.planner import QueryPlan


def test_rag_source():
    long_content = "x" * 400
    source = RAGSource(
        chunk_id="c1",
        document_id="d1",
        document_title="Test Doc",
        section_path="Section A",
        page_start=1,
        page_end=3,
        score=0.85,
        content_preview=long_content,
    )
    assert source.chunk_id == "c1"
    assert source.score == 0.85
    assert source.content == long_content
    assert source.content_preview == long_content[:300]


def test_rag_source_to_dict():
    source = RAGSource(
        chunk_id="c1",
        document_id="d1",
        document_title="Doc",
        section_path=None,
        page_start=None,
        page_end=None,
        score=0.9,
        content_preview="Preview",
        vector_score=0.7,
        bm25_score=0.3,
        combined_score=0.8,
    )
    d = source.to_dict()
    assert d["chunk_id"] == "c1"
    assert d["document_id"] == "d1"
    assert d["score"] == 0.9
    assert d["vector_score"] == 0.7
    assert d["bm25_score"] == 0.3
    assert d["combined_score"] == 0.8
    assert "content_preview" in d


def test_build_context_empty():
    context, citations = build_context([])
    assert context.strip() == ""
    assert citations == []


def test_build_context_single_source():
    source = RAGSource(
        chunk_id="c1",
        document_id="d1",
        document_title="Test Document",
        section_path="Methods",
        page_start=5,
        page_end=7,
        score=0.9,
        content_preview="The methodology involves...",
    )
    context, citations = build_context([source])
    assert "[1] Test Document" in context
    assert "Methods" in context
    assert len(citations) == 1
    assert citations[0]["document_title"] == "Test Document"


def test_build_context_uses_full_source_content_not_preview():
    long_content = "start " + ("middle " * 80) + "tail_answer"
    source = RAGSource(
        chunk_id="c1",
        document_id="d1",
        document_title="Long Document",
        section_path="Body",
        page_start=None,
        page_end=None,
        score=0.9,
        content_preview=long_content,
    )

    context, citations = build_context([source])

    assert "tail_answer" in context
    assert len(source.content_preview) == 300
    assert "tail_answer" not in citations[0]["content_preview"]


def test_rerank_sources_uses_full_source_content_not_preview():
    from app.services.reranker import RerankerResult

    long_content = ("prefix " * 60) + "tail_keyword"
    sources = [
        RAGSource("c1", "d1", "Long Doc", None, None, None, 0.1, long_content),
        RAGSource("c2", "d2", "Short Doc", None, None, None, 0.9, "unrelated"),
    ]

    def rerank(query, documents):
        assert query == "tail_keyword"
        assert documents[0] == long_content
        assert len(sources[0].content_preview) == 300
        assert "tail_keyword" not in sources[0].content_preview
        return [RerankerResult(index=0, score=1.0), RerankerResult(index=1, score=0.0)]

    with patch("app.services.rag.get_reranker") as mock_get_reranker:
        mock_get_reranker.return_value.rerank.side_effect = rerank
        reranked = rerank_sources("tail_keyword", sources)

    assert reranked[0].document_title == "Long Doc"


def test_build_context_multiple_sources():
    sources = [
        RAGSource(
            chunk_id="c1",
            document_id="d1",
            document_title="Doc A",
            section_path=None,
            page_start=None,
            page_end=None,
            score=0.9,
            content_preview="Content A",
        ),
        RAGSource(
            chunk_id="c2",
            document_id="d2",
            document_title="Doc B",
            section_path="Results",
            page_start=10,
            page_end=12,
            score=0.8,
            content_preview="Content B",
        ),
    ]
    context, citations = build_context(sources)
    assert "[1] Doc A" in context
    assert "[2] Doc B" in context
    assert len(citations) == 2


def test_build_context_respects_token_budget():
    sources = [
        RAGSource("c1", "d1", "Doc A", None, None, None, 0.9, "alpha " * 80),
        RAGSource("c2", "d2", "Doc B", None, None, None, 0.8, "beta " * 80),
    ]

    context, citations = build_context(sources, max_tokens=40)

    assert "[1] Doc A" in context
    assert "[2] Doc B" not in context
    assert len(citations) == 1


def test_retrieve_sources_runs_hybrid_then_rerank():
    sources = [
        RAGSource("c1", "d1", "Doc1", None, None, None, 0.1, "first"),
        RAGSource("c2", "d2", "Doc2", None, None, None, 0.2, "second"),
    ]

    with patch("app.services.rag.hybrid_search", return_value=sources) as mock_hybrid, \
         patch("app.services.rag.rerank_sources", return_value=[sources[1]]) as mock_rerank, \
         patch("app.services.rag.expand_parent_child_context", side_effect=lambda items, trace=None: items), \
         patch("app.services.rag._write_retrieval_hits_to_clickhouse") as mock_hits:
        result = retrieve_sources("query", "org", ["kb"], top_k=1, expand_query=True)

    assert result == [sources[1]]
    assert mock_hybrid.call_args.kwargs["expand_query"] is True
    assert mock_rerank.call_args.args[0] == "query"
    assert mock_rerank.call_args.kwargs["top_n"] == 1
    assert sources[0].rank_before == 1
    assert sources[1].rank_before == 2
    assert sources[1].rank_after == 1
    mock_hits.assert_called_once()


def test_hybrid_search_empty_results():
    org_id = str(uuid.uuid4())
    with patch("app.services.rag.get_client") as mock_client, \
         patch("app.services.rag.embed_text") as mock_embed:
        mock_embed.return_value = [0.1] * 1536
        mock_w = MagicMock()
        mock_collection = MagicMock()
        mock_w.collections.get = MagicMock(return_value=mock_collection)
        mock_collection.query.hybrid = MagicMock(
            return_value=MagicMock(objects=[])
        )
        mock_client.return_value = mock_w

        results = hybrid_search("test query", org_id, [])
        assert results == []
        call_kwargs = mock_collection.query.hybrid.call_args.kwargs
        assert not isinstance(call_kwargs["return_metadata"], set)
        assert call_kwargs["vector"] == [0.1] * 1536


def test_hybrid_search_with_results():
    org_id = str(uuid.uuid4())
    kb_id = str(uuid.uuid4())

    with patch("app.services.rag.get_client") as mock_client, \
         patch("app.services.rag.embed_text") as mock_embed:
        mock_embed.return_value = [0.1] * 1536
        mock_w = MagicMock()
        mock_collection = MagicMock()
        mock_w.collections.get = MagicMock(return_value=mock_collection)

        # Create mock search result
        mock_obj = MagicMock()
        mock_obj.uuid = uuid.uuid4()
        mock_obj.properties = {
            "org_id": org_id,
            "kb_id": kb_id,
            "document_id": "doc-1",
            "document_version_id": "ver-1",
            "chunk_id": "chunk-1",
            "security_level": "internal",
            "status": "ready",
            "content": "Test content from search",
            "title": "Test Document",
            "section_path": "Introduction",
            "page_start": 1,
            "page_end": 2,
            "document_type": "general",
            "embedding_model": "text-embedding-v3",
        }
        mock_obj.metadata = {"score": 0.85}

        mock_response = MagicMock()
        mock_response.objects = [mock_obj]
        mock_collection.query.hybrid = MagicMock(return_value=mock_response)
        mock_client.return_value = mock_w

        results = hybrid_search("test query", org_id, [kb_id], top_k=5)
        assert len(results) == 1
        assert results[0].document_id == "doc-1"
        assert results[0].document_title == "Test Document"
        assert results[0].score == 0.85
        assert results[0].page_start == 1


def test_hybrid_search_returns_separate_component_scores():
    org_id = str(uuid.uuid4())
    kb_id = str(uuid.uuid4())
    chunk_id = uuid.uuid4()

    with patch("app.services.rag.get_client") as mock_client, \
         patch("app.services.rag.embed_text") as mock_embed:
        mock_embed.return_value = [0.1] * 1536
        mock_w = MagicMock()
        mock_collection = MagicMock()
        mock_w.collections.get = MagicMock(return_value=mock_collection)

        mock_obj = MagicMock()
        mock_obj.uuid = chunk_id
        mock_obj.properties = {
            "kb_id": kb_id,
            "document_id": "doc-1",
            "content": "Hybrid content",
            "title": "Hybrid Document",
        }
        mock_obj.metadata = {"score": 0.85}

        dense_obj = MagicMock()
        dense_obj.uuid = chunk_id
        dense_obj.metadata = {"distance": 0.25}

        bm25_obj = MagicMock()
        bm25_obj.uuid = chunk_id
        bm25_obj.metadata = {"score": 0.42}

        mock_collection.query.hybrid.return_value = MagicMock(objects=[mock_obj])
        mock_collection.query.near_vector.return_value = MagicMock(objects=[dense_obj])
        mock_collection.query.bm25.return_value = MagicMock(objects=[bm25_obj])
        mock_client.return_value = mock_w

        results = hybrid_search("test query", org_id, [kb_id], top_k=5)

    assert len(results) == 1
    assert results[0].hybrid_score == 0.85
    assert abs(results[0].combined_score - 0.8) < 1e-6
    assert results[0].vector_score == 0.75
    assert results[0].bm25_score == 0.42


def test_hybrid_search_fuses_metadata_results():
    org_id = str(uuid.uuid4())
    kb_id = str(uuid.uuid4())
    metadata_chunk_id = uuid.uuid4()

    with patch("app.services.rag.get_client") as mock_client, \
         patch("app.services.rag.embed_text") as mock_embed:
        mock_embed.return_value = [0.1] * 1536
        mock_w = MagicMock()
        mock_collection = MagicMock()
        mock_w.collections.get = MagicMock(return_value=mock_collection)

        metadata_obj = MagicMock()
        metadata_obj.uuid = metadata_chunk_id
        metadata_obj.properties = {
            "kb_id": kb_id,
            "document_id": "doc-meta",
            "content": "PD-1 metadata content",
            "title": "Metadata Document",
        }
        metadata_obj.metadata = {"score": 0.0}

        mock_collection.query.hybrid.return_value = MagicMock(objects=[])
        mock_collection.query.near_vector.return_value = MagicMock(objects=[])
        mock_collection.query.bm25.return_value = MagicMock(objects=[])
        mock_collection.query.fetch_objects.return_value = MagicMock(objects=[metadata_obj])
        mock_client.return_value = mock_w

        results = hybrid_search("PD-1 adverse events", org_id, [kb_id], top_k=5)

    assert len(results) == 1
    assert results[0].document_id == "doc-meta"
    assert results[0].metadata_score > 0


def test_hybrid_search_reads_object_metadata_score():
    org_id = str(uuid.uuid4())

    with patch("app.services.rag.get_client") as mock_client, \
         patch("app.services.rag.embed_text") as mock_embed:
        mock_embed.return_value = [0.1] * 1536
        mock_w = MagicMock()
        mock_collection = MagicMock()
        mock_w.collections.get = MagicMock(return_value=mock_collection)

        mock_obj = MagicMock()
        mock_obj.uuid = uuid.uuid4()
        mock_obj.properties = {
            "document_id": "doc-1",
            "content": "Search content",
            "title": "Doc",
        }
        mock_obj.metadata = MagicMock()
        mock_obj.metadata.score = 0.72

        mock_response = MagicMock()
        mock_response.objects = [mock_obj]
        mock_collection.query.hybrid = MagicMock(return_value=mock_response)
        mock_client.return_value = mock_w

        results = hybrid_search("test query", org_id, [], top_k=5)

        assert len(results) == 1
        assert results[0].score == 0.72


def test_hybrid_search_fallback_relaxes_kb_filter_when_zero_results():
    org_id = str(uuid.uuid4())
    kb_id = str(uuid.uuid4())

    with patch("app.services.rag.get_client") as mock_client, \
         patch("app.services.rag.embed_text") as mock_embed:
        mock_embed.return_value = [0.1] * 1536
        mock_w = MagicMock()
        mock_collection = MagicMock()
        mock_w.collections.get = MagicMock(return_value=mock_collection)

        fallback_obj = MagicMock()
        fallback_obj.uuid = uuid.uuid4()
        fallback_obj.properties = {
            "kb_id": "other-kb",
            "document_id": "doc-1",
            "content": "Fallback content",
            "title": "Fallback Document",
        }
        fallback_obj.metadata = {"score": 0.55}

        mock_collection.query.hybrid.side_effect = [
            MagicMock(objects=[]),
            MagicMock(objects=[fallback_obj]),
        ]
        mock_collection.query.near_vector.return_value = MagicMock(objects=[])
        mock_collection.query.bm25.return_value = MagicMock(objects=[])
        mock_client.return_value = mock_w

        results = hybrid_search("test query", org_id, [kb_id], top_k=5)

    assert len(results) == 1
    assert results[0].combined_score == 0.55
    assert mock_collection.query.hybrid.call_count == 2


def test_retrieve_plan_sources_runs_multiple_queries_and_deduplicates():
    source_a = RAGSource("c1", "d1", "Doc1", None, None, None, 0.9, "first", combined_score=0.9)
    source_b = RAGSource("c2", "d2", "Doc2", None, None, None, 0.8, "second", combined_score=0.8)
    source_a_lower = RAGSource("c1", "d1", "Doc1", None, None, None, 0.1, "first", combined_score=0.1)
    plan = QueryPlan(
        original="q",
        queries=["q", "sub query"],
        enabled=True,
        strategy="deterministic",
        reason="decomposed",
    )

    with patch("app.services.rag.hybrid_search", side_effect=[[source_a], [source_a_lower, source_b]]) as mock_hybrid:
        result = _retrieve_plan_sources(
            plan=plan,
            query="q",
            org_id="org",
            kb_ids=["kb"],
            top_k=5,
            expand_query=True,
        )

    assert mock_hybrid.call_count == 2
    assert [source.chunk_id for source in result] == ["c1", "c2"]


def test_hybrid_search_uses_retrieval_cache_for_repeated_query():
    _clear_rag_caches()
    org_id = str(uuid.uuid4())
    kb_id = str(uuid.uuid4())

    with patch("app.services.rag.settings") as mock_settings, \
         patch("app.services.rag.get_client") as mock_client, \
         patch("app.services.rag.embed_text") as mock_embed:
        mock_settings.embedding_model = "text-embedding-v3"
        mock_settings.query_cache_ttl = 300
        mock_settings.retrieval_cache_ttl = 1800
        mock_settings.query_expansion = False
        mock_embed.return_value = [0.1] * 1536

        mock_w = MagicMock()
        mock_collection = MagicMock()
        mock_w.collections.get.return_value = mock_collection
        mock_obj = MagicMock()
        mock_obj.uuid = uuid.uuid4()
        mock_obj.properties = {
            "document_id": "doc-1",
            "content": "Cached search content",
            "title": "Doc",
        }
        mock_obj.metadata = {"score": 0.88}
        mock_collection.query.hybrid.return_value = MagicMock(objects=[mock_obj])
        mock_client.return_value = mock_w

        first = hybrid_search("same query", org_id, [kb_id], top_k=5)
        first[0].score = 0.01
        second = hybrid_search("same query", org_id, [kb_id], top_k=5)

    assert mock_embed.call_count == 1
    assert mock_collection.query.hybrid.call_count == 1
    assert second[0].score == 0.88
    assert second[0] is not first[0]
    _clear_rag_caches()


def test_hybrid_search_reuses_query_embedding_when_retrieval_cache_disabled():
    _clear_rag_caches()
    org_id = str(uuid.uuid4())

    with patch("app.services.rag.settings") as mock_settings, \
         patch("app.services.rag.get_client") as mock_client, \
         patch("app.services.rag.embed_text") as mock_embed:
        mock_settings.embedding_model = "text-embedding-v3"
        mock_settings.query_cache_ttl = 300
        mock_settings.retrieval_cache_ttl = 0
        mock_settings.query_expansion = False
        mock_embed.return_value = [0.1] * 1536

        mock_w = MagicMock()
        mock_collection = MagicMock()
        mock_w.collections.get.return_value = mock_collection
        mock_collection.query.hybrid.return_value = MagicMock(objects=[])
        mock_client.return_value = mock_w

        hybrid_search("same query", org_id, [], top_k=5)
        hybrid_search("same query", org_id, [], top_k=5)

    assert mock_embed.call_count == 1
    assert mock_collection.query.hybrid.call_count == 2
    _clear_rag_caches()


def test_assemble_context_and_generate_no_results():
    with patch("app.services.rag.retrieve_sources") as mock_search:
        mock_search.return_value = []

        items = list(assemble_context_and_generate(
            query="test",
            org_id="org-1",
            kb_ids=[],
        ))
        assert len(items) == 1
        assert items[0]["done"] is True
        assert items[0]["sources"] == []
        assert items[0]["delta"]


def test_assemble_context_and_generate_with_results():
    org_id = str(uuid.uuid4())
    source = RAGSource(
        chunk_id="c1",
        document_id="d1",
        document_title="Test Doc",
        section_path=None,
        page_start=None,
        page_end=None,
        score=0.9,
        content_preview="RAG is a technique that combines retrieval and generation.",
    )

    with patch("app.services.rag.retrieve_sources") as mock_search, \
         patch("app.services.rag.compress_sources_for_query") as mock_compress, \
         patch("app.services.rag.generate_stream") as mock_gen:
        mock_search.return_value = [source]
        mock_compress.return_value = (
            [source],
            MagicMock(input_count=1, compressed_count=0, original_chars=10, compressed_chars=10),
        )
        mock_resp = MagicMock()
        mock_resp.output = MagicMock()
        mock_resp.output.choices = [{"message": {"content": "RAG is useful."}}]
        mock_gen.return_value = iter([mock_resp])

        items = list(assemble_context_and_generate(
            query="What is RAG?",
            org_id=org_id,
            kb_ids=["kb-1"],
        ))

        # Should have at least the answer chunks + final done chunk
        done_items = [i for i in items if i["done"]]
        assert len(done_items) == 1
        assert len(done_items[0]["sources"]) == 1
        assert done_items[0]["sources"][0]["document_title"] == "Test Doc"
        assert mock_gen.call_args.kwargs["messages"] is None
        mock_compress.assert_called_once()


def test_assemble_context_and_generate_passes_history_to_llm_and_retrieval():
    source = RAGSource(
        chunk_id="c1",
        document_id="d1",
        document_title="Test Doc",
        section_path=None,
        page_start=None,
        page_end=None,
        score=0.9,
        content_preview="contraindication context",
    )
    messages = [
        {"role": "system", "content": "Earlier summary: user asks about immunotherapy."},
        {"role": "user", "content": "Who is it suitable for?"},
        {"role": "assistant", "content": "It depends on patient subtype."},
    ]

    with patch("app.services.rag.retrieve_sources") as mock_search, \
         patch("app.services.rag.generate_stream") as mock_gen:
        mock_search.return_value = [source]
        mock_resp = MagicMock()
        mock_resp.output = MagicMock()
        mock_resp.output.choices = [{"message": {"content": "Contraindications include..."}}]
        mock_gen.return_value = iter([mock_resp])

        list(assemble_context_and_generate(
            query="What are the contraindications?",
            org_id="org-1",
            kb_ids=["kb-1"],
            messages=messages,
        ))

    retrieval_query = mock_search.call_args.args[0]
    assert "Earlier summary" in retrieval_query
    assert "What are the contraindications?" in retrieval_query
    assert mock_gen.call_args.kwargs["messages"] == messages


def test_assemble_context_and_generate_truncates_chunks():
    sources = [RAGSource(
        chunk_id=f"c{i}",
        document_id=f"d{i}",
        document_title=f"Doc {i}",
        section_path=None,
        page_start=None,
        page_end=None,
        score=0.9 - i * 0.1,
        content_preview=f"Content {i}",
    ) for i in range(10)]

    with patch("app.services.rag.retrieve_sources") as mock_search, \
         patch("app.services.rag.generate_stream") as mock_gen:
        mock_search.return_value = sources
        mock_resp = MagicMock()
        mock_resp.output = MagicMock()
        mock_resp.output.choices = [{"message": {"content": "done"}}]
        mock_gen.return_value = iter([mock_resp])

        # max_chunks=5 should limit to first 5 sources
        items = list(assemble_context_and_generate(
            query="test",
            org_id="org-1",
            kb_ids=["kb-1"],
            max_chunks=5,
        ))

        done_items = [i for i in items if i["done"]]
        assert len(done_items[0]["sources"]) == 5


def test_assemble_context_and_generate_appends_citation_validation_note():
    source = RAGSource(
        chunk_id="c1",
        document_id="d1",
        document_title="Test Doc",
        section_path=None,
        page_start=None,
        page_end=None,
        score=0.9,
        content_preview="Evidence.",
    )

    with patch("app.services.rag.retrieve_sources") as mock_search, \
         patch("app.services.rag.compress_sources_for_query") as mock_compress, \
         patch("app.services.rag.generate_stream") as mock_gen:
        mock_search.return_value = [source]
        mock_compress.return_value = (
            [source],
            MagicMock(input_count=1, compressed_count=0, original_chars=10, compressed_chars=10),
        )
        mock_resp = MagicMock()
        mock_resp.output = MagicMock()
        mock_resp.output.choices = [{"message": {"content": "Answer cites missing [2]."}}]
        mock_gen.return_value = iter([mock_resp])

        items = list(assemble_context_and_generate(
            query="What?",
            org_id="org-1",
            kb_ids=["kb-1"],
        ))

    assert items[-1]["done"] is True
    assert "引用校验提示" in items[-1]["delta"]
    assert "[2]" in items[-1]["delta"]
