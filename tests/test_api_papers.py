"""Tests for paper API endpoints."""
import uuid
from unittest.mock import patch

import pytest

from app.models.document import DocumentVersion
from tests.conftest import make_mock_session


@pytest.fixture
def papers_client(mock_settings):
    """Create test client for paper endpoints."""
    # Import app module properly
    import app.main as app_main

    with patch.object(app_main, "ensure_collection"):
        from app.main import app
        from app.api.deps import get_current_user
        from app.database import get_db

        test_user = {
            "user_id": str(uuid.uuid4()),
            "org_id": str(uuid.uuid4()),
            "roles": ["viewer"],
        }
        mock_sess = make_mock_session()

        async def _override_db():
            yield mock_sess

        app.dependency_overrides[get_current_user] = lambda: test_user
        app.dependency_overrides[get_db] = _override_db

        from fastapi.testclient import TestClient
        client = TestClient(app)
        yield client

        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_db, None)


class TestPaperSchemas:
    def test_paper_create_valid(self):
        from app.schemas.paper import PaperCreate
        kb = uuid.uuid4()
        data = PaperCreate(kb_id=kb, title="Test Paper")
        assert data.kb_id == kb
        assert data.title == "Test Paper"
        assert data.doi is None

    def test_paper_create_with_doi(self):
        from app.schemas.paper import PaperCreate
        kb = uuid.uuid4()
        data = PaperCreate(kb_id=kb, title="Test", doi="10.1234/test")
        assert data.doi == "10.1234/test"

    def test_doi_import_request(self):
        from app.schemas.paper import DoiImportRequest
        kb = uuid.uuid4()
        data = DoiImportRequest(doi="10.1234/test", kb_id=kb)
        assert data.doi == "10.1234/test"

    def test_pmid_import_request(self):
        from app.schemas.paper import PmidImportRequest
        kb = uuid.uuid4()
        data = PmidImportRequest(pmid="12345", kb_id=kb)
        assert data.pmid == "12345"

    def test_paper_evidence_response(self):
        from app.schemas.paper import PaperEvidenceResponse
        data = PaperEvidenceResponse(
            paper_id=uuid.uuid4(),
            title="Test",
            study_type="RCT",
            sample_size=100,
            has_randomization=True,
            has_blinding=False,
            pico={"population": "adults", "intervention": "drug A", "comparator": "placebo", "outcome": "survival"},
            evidence_level="A",
        )
        assert data.study_type == "RCT"
        assert data.pico["population"] == "adults"

    def test_paper_reference_response(self):
        from app.schemas.paper import PaperReferencesResponse, PaperReference
        data = PaperReferencesResponse(
            paper_id=uuid.uuid4(),
            title="Test",
            references=[PaperReference(authors="Smith", title="Paper", year=2023)],
            total=1,
        )
        assert data.total == 1
        assert data.references[0].title == "Paper"

    def test_similar_paper_response(self):
        from app.schemas.paper import SimilarPapersResponse, SimilarPaperItem
        data = SimilarPapersResponse(
            paper_id=uuid.uuid4(),
            similar_papers=[
                SimilarPaperItem(
                    paper_id=uuid.uuid4(),
                    title="Similar Paper",
                    doi="10.1234/similar",
                    similarity_score=0.75,
                    shared_mesh=["Neoplasms"],
                )
            ],
        )
        assert len(data.similar_papers) == 1
        assert data.similar_papers[0].similarity_score == 0.75

    def test_paper_response_full(self):
        from app.schemas.paper import PaperResponse
        from datetime import datetime, timezone
        data = PaperResponse(
            id=uuid.uuid4(),
            org_id=uuid.uuid4(),
            kb_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            title="Full Paper",
            status="ready",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        assert data.title == "Full Paper"
        assert data.status == "ready"

    def test_upload_response(self):
        from app.schemas.paper import PaperUploadResponse
        data = PaperUploadResponse(
            paper_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            title="Uploaded Paper",
            status="draft",
            ingestion_job_id=uuid.uuid4(),
        )
        assert data.title == "Uploaded Paper"
        assert data.status == "draft"


def test_upload_paper_uses_configured_storage_and_embedding_model(papers_client, tmp_path):
    client = papers_client

    from app.database import get_db
    from app.main import app

    added_objects = []

    class Session:
        def add(self, obj):
            added_objects.append(obj)

        async def commit(self):
            return None

    async def db_override():
        yield Session()

    app.dependency_overrides[get_db] = db_override
    kb_id = str(uuid.uuid4())
    try:
        with patch("app.api.v1.papers.settings") as mock_settings, \
         patch("app.workers.celery_app.celery_app.send_task") as mock_send_task:
            mock_settings.storage_path = str(tmp_path)
            mock_settings.embedding_model = "configured-embedding-model"

            response = client.post(
                "/api/v1/papers/upload",
                data={"kb_id": kb_id},
                files={"file": ("paper.pdf", b"%PDF-1.4\n", "application/pdf")},
            )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    version = next(obj for obj in added_objects if isinstance(obj, DocumentVersion))
    assert version.embedding_model == "configured-embedding-model"
    assert str(tmp_path) in version.storage_path
    mock_send_task.assert_called_once()
    assert mock_send_task.call_args.args[0] == "parse_paper"
    sent_args = mock_send_task.call_args.kwargs["args"]
    assert sent_args[-1] == kb_id
