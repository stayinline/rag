"""Tests for Celery workers and tasks."""
import asyncio
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock, patch

from app.workers.celery_app import celery_app


def _clear_worker_async_context(tasks_module):
    loop = getattr(tasks_module._worker_async_context, "loop", None)
    if loop is not None:
        if not loop.is_closed():
            loop.close()
        asyncio.set_event_loop(None)
        delattr(tasks_module._worker_async_context, "loop")


def test_celery_config():
    assert celery_app.conf.task_serializer == "json"
    assert celery_app.conf.accept_content == ["json"]
    assert celery_app.conf.result_serializer == "json"
    assert celery_app.conf.timezone == "Asia/Shanghai"


def test_run_async_uses_thread_local_event_loops_for_concurrent_sync_calls():
    from app.workers import tasks

    async def capture_loop(caller_thread_id):
        return {
            "loop_id": id(asyncio.get_running_loop()),
            "thread_id": threading.get_ident(),
            "caller_thread_id": caller_thread_id,
        }

    barrier = threading.Barrier(2)

    def run_one():
        barrier.wait(timeout=5)
        caller_thread_id = threading.get_ident()
        try:
            return tasks._run_async(capture_loop(caller_thread_id))
        finally:
            _clear_worker_async_context(tasks)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_one) for _ in range(2)]
        results = [future.result(timeout=5) for future in futures]

    assert len({result["loop_id"] for result in results}) == 2
    assert len({result["thread_id"] for result in results}) == 2
    assert all(result["thread_id"] == result["caller_thread_id"] for result in results)


def test_run_async_propagates_coroutine_exception():
    from app.workers import tasks

    class WorkerAsyncError(RuntimeError):
        pass

    async def fail():
        raise WorkerAsyncError("db update failed")

    try:
        tasks._run_async(fail())
        assert False, "Should propagate the coroutine exception"
    except WorkerAsyncError as exc:
        assert str(exc) == "db update failed"
    finally:
        _clear_worker_async_context(tasks)


def test_parse_document_task_success(tmp_path):
    from app.workers.tasks import parse_document_task

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Test document content")
        f.flush()
        with patch("app.workers.tasks.settings") as mock_settings:
            mock_settings.storage_path = str(tmp_path)
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
         patch("app.workers.tasks.embed_texts") as mock_embed, \
         patch("app.workers.tasks.settings") as mock_settings:
        mock_settings.embedding_model = "configured-embedding-model"
        mock_embed.return_value = [[0.1] * 1536, [0.2] * 1536]

        mock_w_client = MagicMock()
        mock_collection = MagicMock()
        mock_w_client.collections.get = MagicMock(return_value=mock_collection)
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
    first_insert = mock_collection.data.insert.call_args_list[0]
    assert first_insert.kwargs["properties"]["embedding_model"] == "configured-embedding-model"
    os.unlink(parsed_path)


def test_upsert_weaviate_object_updates_existing_object():
    from app.workers.tasks import _upsert_weaviate_object

    collection = MagicMock()
    collection.data.exists.return_value = True
    properties = {"status": "draft"}
    vector = [0.1, 0.2]

    result = _upsert_weaviate_object(
        collection,
        object_id="chunk-id",
        properties=properties,
        vector=vector,
        task_id="task-id",
        owner_type="document_id",
        owner_id="doc-id",
    )

    assert result == "updated"
    collection.data.insert.assert_not_called()
    collection.data.update.assert_called_once_with(uuid="chunk-id", properties=properties, vector=vector)


def test_upsert_weaviate_object_updates_duplicate_insert():
    from app.workers.tasks import _upsert_weaviate_object

    class DuplicateObjectError(Exception):
        status_code = 422

        def __str__(self):
            return "id 'chunk-id' already exists"

    collection = MagicMock()
    collection.data.exists.return_value = False
    collection.data.insert.side_effect = DuplicateObjectError()
    properties = {"status": "draft"}
    vector = [0.1, 0.2]

    result = _upsert_weaviate_object(
        collection,
        object_id="chunk-id",
        properties=properties,
        vector=vector,
        task_id="task-id",
        owner_type="document_id",
        owner_id="doc-id",
    )

    assert result == "updated"
    collection.data.insert.assert_called_once_with(uuid="chunk-id", properties=properties, vector=vector)
    collection.data.update.assert_called_once_with(uuid="chunk-id", properties=properties, vector=vector)


def test_batch_publish_weaviate_chunks_preserves_existing_properties_and_vector():
    from app.workers.tasks import _batch_publish_weaviate_chunks

    collection = MagicMock()
    obj_1 = MagicMock()
    obj_1.uuid = "chunk-1"
    obj_1.properties = {"status": "draft", "content": "A"}
    obj_1.vector = {"default": [0.1, 0.2]}
    obj_2 = MagicMock()
    obj_2.uuid = "chunk-2"
    obj_2.properties = {"status": "draft", "content": "B"}
    obj_2.vector = {"default": [0.3, 0.4]}
    collection.query.fetch_objects_by_ids.return_value = MagicMock(objects=[obj_1, obj_2])

    batch = MagicMock()
    collection.batch.fixed_size.return_value.__enter__.return_value = batch
    collection.batch.fixed_size.return_value.__exit__.return_value = False
    collection.batch.failed_objects = []

    updated_count = _batch_publish_weaviate_chunks(collection, ["chunk-1", "chunk-2"])

    assert updated_count == 2
    collection.query.fetch_objects_by_ids.assert_called_once_with(
        ["chunk-1", "chunk-2"],
        include_vector=True,
        return_properties=True,
    )
    collection.batch.fixed_size.assert_called_once_with(batch_size=2, concurrent_requests=2)
    batch.add_object.assert_any_call(
        uuid="chunk-1",
        properties={"status": "ready", "content": "A"},
        vector={"default": [0.1, 0.2]},
    )
    batch.add_object.assert_any_call(
        uuid="chunk-2",
        properties={"status": "ready", "content": "B"},
        vector={"default": [0.3, 0.4]},
    )
    collection.data.update.assert_not_called()


def test_batch_publish_weaviate_chunks_fails_when_chunk_missing():
    from app.workers.tasks import _batch_publish_weaviate_chunks

    collection = MagicMock()
    obj_1 = MagicMock()
    obj_1.uuid = "chunk-1"
    obj_1.properties = {"status": "draft", "content": "A"}
    obj_1.vector = {"default": [0.1, 0.2]}
    collection.query.fetch_objects_by_ids.return_value = MagicMock(objects=[obj_1])

    try:
        _batch_publish_weaviate_chunks(collection, ["chunk-1", "chunk-2"])
        assert False, "Should fail when Weaviate does not return every requested chunk"
    except RuntimeError as exc:
        assert "missing 1 chunks" in str(exc)

    collection.batch.fixed_size.assert_not_called()


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
        obj_1 = MagicMock()
        obj_1.uuid = "chunk-1"
        obj_1.properties = {"status": "draft", "content": "A"}
        obj_1.vector = {"default": [0.1, 0.2]}
        obj_2 = MagicMock()
        obj_2.uuid = "chunk-2"
        obj_2.properties = {"status": "draft", "content": "B"}
        obj_2.vector = {"default": [0.3, 0.4]}
        mock_collection.query.fetch_objects_by_ids.return_value = MagicMock(objects=[obj_1, obj_2])
        mock_batch = MagicMock()
        mock_collection.batch.fixed_size.return_value.__enter__.return_value = mock_batch
        mock_collection.batch.fixed_size.return_value.__exit__.return_value = False
        mock_collection.batch.failed_objects = []
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
    mock_collection.query.fetch_objects_by_ids.assert_called_once_with(
        chunk_ids,
        include_vector=True,
        return_properties=True,
    )
    assert mock_batch.add_object.call_count == 2
    mock_collection.data.update.assert_not_called()
    mock_w_client.close.assert_not_called()


def test_chunk_and_embed_from_parse_task_continues_pipeline():
    from app.workers.tasks import chunk_and_embed_from_parse_task

    parse_result = {
        "org_id": "test-org",
        "document_id": "test-doc",
        "version_id": "test-version",
        "parsed_path": "/tmp/parsed.txt",
    }

    with patch("app.workers.tasks.chunk_and_embed_task") as mock_chunk:
        mock_chunk.return_value = {"status": "ok"}
        result = chunk_and_embed_from_parse_task(
            parse_result,
            "test-kb",
            "Test Document",
        )

    assert result == {"status": "ok"}
    mock_chunk.assert_called_once_with(
        "test-org",
        "test-doc",
        "test-version",
        "test-kb",
        "Test Document",
        "/tmp/parsed.txt",
        10,
    )


def test_publish_document_from_chunks_task_continues_pipeline():
    from app.workers.tasks import publish_document_from_chunks_task

    embed_result = {
        "org_id": "test-org",
        "document_id": "test-doc",
        "version_id": "test-version",
        "kb_id": "test-kb",
        "chunk_count": 2,
        "chunk_ids": ["chunk-1", "chunk-2"],
    }

    with patch("app.workers.tasks.publish_document_task") as mock_publish:
        mock_publish.return_value = {"status": "ready"}
        result = publish_document_from_chunks_task(embed_result)

    assert result == {"status": "ready"}
    mock_publish.assert_called_once_with(
        "test-org",
        "test-doc",
        "test-version",
        "test-kb",
        2,
        ["chunk-1", "chunk-2"],
    )


def test_parse_paper_task_writes_kb_id_to_weaviate():
    from app.services.paper_parser import PaperParseResult, PaperSection
    from app.workers.tasks import parse_paper_task

    parse_result = PaperParseResult(
        title="Google File System",
        abstract="A scalable distributed file system.",
        sections=[
            PaperSection(
                section_type="introduction",
                heading="Introduction",
                content="GFS is designed for large distributed data-intensive applications.",
            ),
        ],
    )

    mock_session = AsyncMock()
    mock_session.get = AsyncMock(return_value=None)
    mock_session.execute = AsyncMock()
    mock_session.commit = AsyncMock()
    mock_session_factory = MagicMock()
    mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)

    mock_w_client = MagicMock()
    mock_collection = MagicMock()
    mock_w_client.collections.get.return_value = mock_collection

    kb_id = "kb-123"
    with patch("app.workers.tasks.parse_paper_local", return_value=parse_result), \
         patch("app.workers.tasks.extract_medical_entities", return_value={"diseases": [], "drugs": [], "targets": []}), \
         patch("app.workers.tasks.embed_texts", return_value=[[0.1] * 1536, [0.2] * 1536]), \
         patch("app.workers.tasks.async_session", mock_session_factory), \
         patch("app.workers.tasks.get_client", return_value=mock_w_client), \
         patch("app.workers.tasks.settings") as mock_settings:
        mock_settings.embedding_model = "configured-embedding-model"
        result = parse_paper_task(
            "test-org",
            "test-doc",
            "test-version",
            "test-paper",
            "paper.pdf",
            kb_id=kb_id,
        )

    assert result["kb_id"] == kb_id
    first_insert = mock_collection.data.insert.call_args_list[0]
    assert first_insert.kwargs["properties"]["kb_id"] == kb_id
    assert first_insert.kwargs["properties"]["embedding_model"] == "configured-embedding-model"


def test_task_names():
    """Verify task names are properly configured."""
    from app.workers.tasks import (
        parse_document_task,
        chunk_and_embed_task,
        chunk_and_embed_from_parse_task,
        publish_document_task,
        publish_document_from_chunks_task,
        parse_paper_task,
        run_evaluation_task,
    )

    assert parse_document_task.name == "parse_document"
    assert chunk_and_embed_task.name == "chunk_and_embed"
    assert chunk_and_embed_from_parse_task.name == "chunk_and_embed_from_parse"
    assert publish_document_task.name == "publish_document"
    assert publish_document_from_chunks_task.name == "publish_document_from_chunks"
    assert parse_paper_task.name == "parse_paper"
    assert run_evaluation_task.name == "run_evaluation"


def test_parse_document_task_with_pdf(tmp_path):
    from app.workers.tasks import parse_document_task
    import fitz

    fd, pdf_path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    try:
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "PDF document content for testing")
        doc.save(pdf_path)
        doc.close()

        with patch("app.workers.tasks.settings") as mock_settings:
            mock_settings.storage_path = str(tmp_path)
            result = parse_document_task(
                "test-org",
                "test-doc",
                "test-version",
                pdf_path,
            )
        assert result["text_length"] > 0
        assert "PDF document content" in open(result["parsed_path"]).read()
    finally:
        os.unlink(pdf_path)
