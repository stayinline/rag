"""Tests for Celery workers and tasks."""
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

from app.workers.celery_app import celery_app


def test_celery_config():
    assert celery_app.conf.task_serializer == "json"
    assert celery_app.conf.accept_content == ["json"]
    assert celery_app.conf.result_serializer == "json"
    assert celery_app.conf.timezone == "Asia/Shanghai"


def test_parse_document_task_success():
    from app.workers.tasks import parse_document_task

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Test document content")
        f.flush()
        result = parse_document_task(
            "test-org",
            "test-doc",
            "test-version",
            f.name,
        )
    assert result["org_id"] == "test-org"
    assert result["document_id"] == "test-doc"
    assert result["version_id"] == "test-version"
    assert result["text_length"] == 21
    assert "parsed_path" in result
    os.unlink(f.name)


def test_parse_document_task_unsupported_file():
    """Task should retry on unsupported file type."""
    from app.workers.tasks import parse_document_task
    from celery.exceptions import Retry

    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
        f.write(b"content")
        f.flush()

        # When called directly, task's self.retry raises Retry
        try:
            parse_document_task("org", "doc", "ver", f.name)
            assert False, "Should have raised Retry"
        except (Retry, ValueError):
            pass  # Either Retry from retry() or ValueError from parse_file
    os.unlink(f.name)


def test_parse_document_task_file_not_found():
    from app.workers.tasks import parse_document_task
    from celery.exceptions import Retry

    try:
        parse_document_task("org", "doc", "ver", "/nonexistent/file.txt")
        assert False, "Should have raised Retry or FileNotFoundError"
    except (Retry, FileNotFoundError):
        pass  # Expected


def test_chunk_and_embed_task():
    from app.workers.tasks import chunk_and_embed_task

    content = """# Introduction
This is the introduction section with some content about RAG systems.

# Methods
Here we describe the methods used in this study about retrieval.
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        f.flush()
        parsed_path = f.name

    with patch("app.workers.tasks.get_client") as mock_client, \
         patch("app.workers.tasks.embed_texts") as mock_embed:
        mock_embed.return_value = [[0.1] * 1536, [0.2] * 1536]

        mock_w_client = MagicMock()
        mock_collection = MagicMock()
        mock_w_client.collections.get = MagicMock(return_value=mock_collection)
        mock_w_client.connect = MagicMock()
        mock_w_client.close = MagicMock()
        mock_client.return_value = mock_w_client

        # Do NOT pass None as first arg - Celery auto-injects self
        result = chunk_and_embed_task(
            "test-org",
            "test-doc",
            "test-version",
            "test-kb",
            "Test Document",
            parsed_path,
        )

    assert result["org_id"] == "test-org"
    assert result["document_id"] == "test-doc"
    assert result["chunk_count"] >= 2
    assert len(result["chunk_ids"]) == result["chunk_count"]
    os.unlink(parsed_path)


def test_chunk_and_embed_task_missing_file():
    from app.workers.tasks import chunk_and_embed_task
    from celery.exceptions import Retry

    try:
        chunk_and_embed_task(
            "test-org",
            "test-doc",
            "test-version",
            "test-kb",
            "Test",
            "/nonexistent/parsed.txt",
        )
        assert False, "Should have raised Retry or FileNotFoundError"
    except (Retry, FileNotFoundError):
        pass  # Expected


def test_publish_document_task():
    from app.workers.tasks import publish_document_task

    chunk_ids = ["chunk-1", "chunk-2"]

    # Create a proper async context manager mock for async_session
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock()
    mock_session.commit = AsyncMock()

    mock_session_factory = MagicMock()
    mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)

    with patch("app.workers.tasks.get_client") as mock_client, \
         patch("app.workers.tasks.async_session", mock_session_factory):
        mock_w_client = MagicMock()
        mock_collection = MagicMock()
        mock_w_client.collections.get = MagicMock(return_value=mock_collection)
        mock_w_client.connect = MagicMock()
        mock_w_client.close = MagicMock()
        mock_client.return_value = mock_w_client

        result = publish_document_task(
            "test-org",
            "test-doc",
            "test-version",
            "test-kb",
            2,
            chunk_ids,
        )

    assert result["document_id"] == "test-doc"
    assert result["status"] == "ready"
    assert result["chunk_count"] == 2


def test_task_names():
    """Verify task names are properly configured."""
    from app.workers.tasks import parse_document_task, chunk_and_embed_task, publish_document_task

    assert parse_document_task.name == "parse_document"
    assert chunk_and_embed_task.name == "chunk_and_embed"
    assert publish_document_task.name == "publish_document"


def test_parse_document_task_with_pdf():
    from app.workers.tasks import parse_document_task
    import fitz

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "PDF document content for testing")
        doc.save(f.name)
        doc.close()

        result = parse_document_task(
            "test-org",
            "test-doc",
            "test-version",
            f.name,
        )
        assert result["text_length"] > 0
        assert "PDF document content" in open(result["parsed_path"]).read()
    os.unlink(f.name)
