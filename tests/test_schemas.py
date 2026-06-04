"""Tests for Pydantic schemas."""
import uuid

import pytest
from pydantic import ValidationError

from app.schemas.kb import KBCreate, KBUpdate
from app.schemas.document import IngestionJobResponse, DocumentVersionInfo
from app.schemas.chat import ChatRequest, ChatSource, ChatResponse, ChatStreamChunk
from app.schemas.search import SearchRequest, SearchResultItem, SearchResponse


# KB Schemas
def test_kb_create_valid():
    kb = KBCreate(name="Test KB")
    assert kb.name == "Test KB"
    assert kb.description == ""


def test_kb_create_with_description():
    kb = KBCreate(name="Test KB", description="A test knowledge base")
    assert kb.description == "A test knowledge base"


def test_kb_create_name_too_long():
    with pytest.raises(ValidationError):
        KBCreate(name="x" * 201)


def test_kb_update_partial():
    update = KBUpdate(name="New Name")
    assert update.name == "New Name"
    assert update.description is None
    assert update.is_active is None


# Chat Schemas
def test_chat_request_minimal():
    req = ChatRequest(query="What is RAG?")
    assert req.query == "What is RAG?"
    assert req.kb_ids == []
    assert req.stream is True
    assert req.conversation_id is None


def test_chat_request_full():
    kb_id = uuid.uuid4()
    conv_id = uuid.uuid4()
    req = ChatRequest(
        query="Explain vector databases",
        kb_ids=[kb_id],
        conversation_id=conv_id,
        stream=False,
    )
    assert req.query == "Explain vector databases"
    assert len(req.kb_ids) == 1
    assert req.conversation_id == conv_id
    assert req.stream is False


def test_chat_request_empty_query():
    with pytest.raises(ValidationError):
        ChatRequest(query="")


def test_chat_request_query_too_long():
    with pytest.raises(ValidationError):
        ChatRequest(query="x" * 5001)


def test_chat_source():
    source = ChatSource(
        chunk_id="chunk-1",
        document_id="doc-1",
        document_title="Test Doc",
        section_path="Test/Section",
        page_start=1,
        page_end=3,
        score=0.85,
        content_preview="This is a preview of the chunk content.",
    )
    assert source.chunk_id == "chunk-1"
    assert source.score == 0.85
    assert source.page_start == 1


def test_chat_response():
    kb_id = uuid.uuid4()
    source = ChatSource(
        chunk_id="c1",
        document_id="d1",
        document_title="Doc",
        section_path=None,
        page_start=None,
        page_end=None,
        score=0.9,
        content_preview="Preview",
    )
    resp = ChatResponse(
        answer="Test answer",
        trace_id="trace-1",
        conversation_id=kb_id,
        sources=[source],
        model="qwen-plus",
        prompt_version="v1",
    )
    assert resp.answer == "Test answer"
    assert len(resp.sources) == 1


def test_chat_stream_chunk():
    chunk = ChatStreamChunk(delta="Hello", done=False)
    assert chunk.delta == "Hello"
    assert chunk.done is False


def test_chat_stream_chunk_done():
    chunk = ChatStreamChunk(delta="", done=True, trace_id="t1")
    assert chunk.done is True
    assert chunk.trace_id == "t1"


# Search Schemas
def test_search_request_minimal():
    req = SearchRequest(query="test")
    assert req.query == "test"
    assert req.top_k == 10


def test_search_request_custom_top_k():
    req = SearchRequest(query="test", top_k=25)
    assert req.top_k == 25


def test_search_request_top_k_too_high():
    with pytest.raises(ValidationError):
        SearchRequest(query="test", top_k=51)


def test_search_request_top_k_zero():
    with pytest.raises(ValidationError):
        SearchRequest(query="test", top_k=0)


def test_search_result_item():
    doc_id = uuid.uuid4()
    item = SearchResultItem(
        chunk_id="c1",
        document_id=doc_id,
        document_title="Doc",
        section_path="Section",
        page_start=1,
        page_end=2,
        content_preview="Preview text",
        vector_score=0.9,
        bm25_score=0.8,
        combined_score=0.85,
    )
    assert item.combined_score == 0.85


def test_search_response():
    resp = SearchResponse(query="test", total=5, results=[])
    assert resp.query == "test"
    assert resp.total == 5


# Document Schemas
def test_document_version_info():
    doc_id = uuid.uuid4()
    info = DocumentVersionInfo(
        id=doc_id,
        version=1,
        index_status="pending",
        chunk_count=0,
        storage_path="/path/to/file",
        created_at="2024-01-01T00:00:00Z",
    )
    assert info.version == 1
    assert info.index_status == "pending"


def test_ingestion_job_response():
    job = IngestionJobResponse(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        job_type="parse",
        status="pending",
        retry_count=0,
        error_code=None,
        error_message=None,
        started_at=None,
        finished_at=None,
        created_at="2024-01-01T00:00:00Z",
    )
    assert job.job_type == "parse"
    assert job.status == "pending"
