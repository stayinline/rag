"""Shared test fixtures."""
import tempfile
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def make_mock_session(scalar_one_or_none=None, scalars_all=None, scalar_val=0):
    """Create a fully mocked async session."""
    session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = scalar_one_or_none
    mock_result.scalars.return_value.all.return_value = scalars_all or []
    mock_result.scalar.return_value = scalar_val
    session.execute = AsyncMock(return_value=mock_result)
    session.commit = AsyncMock()
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    session.add = MagicMock()
    session.rollback = AsyncMock()
    return session


@pytest.fixture(autouse=True)
def mock_settings():
    """Use test settings for all tests."""
    with patch("app.config.Settings") as mock:
        mock.return_value.app_name = "RAG Test"
        mock.return_value.debug = False
        mock.return_value.secret_key = "test-secret-key"
        mock.return_value.api_prefix = "/api/v1"
        mock.return_value.database_url = "sqlite+aiosqlite:///./test.db"
        mock.return_value.redis_url = "redis://localhost:6379/15"
        mock.return_value.weaviate_url = "http://localhost:8080"
        mock.return_value.weaviate_grpc_port = 50051
        mock.return_value.dashscope_api_key = "test-api-key"
        mock.return_value.llm_model = "qwen-plus"
        mock.return_value.embedding_model = "text-embedding-v3"
        mock.return_value.storage_path = tempfile.mkdtemp()
        mock.return_value.jwt_algorithm = "HS256"
        mock.return_value.jwt_expire_minutes = 60
        mock.return_value.rag_max_chunks = 5
        mock.return_value.rag_chunk_overlap = 80
        mock.return_value.rag_chunk_size = 600
        mock.return_value.rag_top_k = 10
        mock.return_value.celery_broker_url = "redis://localhost:6379/15"
        mock.return_value.celery_result_backend = "redis://localhost:6379/15"
        # Phase 2 settings
        mock.return_value.grobid_url = "http://localhost:8070"
        mock.return_value.reranker_type = "mock"
        mock.return_value.reranker_top_n = 5
        mock.return_value.query_expansion = True
        # Phase 3 settings
        mock.return_value.clickhouse_url = "http://localhost:8123"
        mock.return_value.enable_trace_logging = True
        yield mock


@pytest.fixture
def mock_dashscope():
    """Mock DashScope API calls."""
    with patch("app.services.embedding.dashscope"), \
         patch("app.services.embedding.TextEmbedding") as mock_embed, \
         patch("app.services.llm.dashscope"), \
         patch("app.services.llm.Generation") as mock_gen:
        embed_resp = MagicMock()
        embed_resp.status_code = 200
        embed_resp.output = {
            "embeddings": [
                {"text_index": 0, "embedding": [0.1] * 1536},
                {"text_index": 1, "embedding": [0.2] * 1536},
            ]
        }
        mock_embed.call = MagicMock(return_value=embed_resp)

        llm_resp = MagicMock()
        llm_resp.output = MagicMock()
        llm_resp.output.choices = [{"message": {"content": "test answer"}}]
        mock_gen.call = MagicMock(return_value=iter([llm_resp]))

        yield {"embed": mock_embed, "generation": mock_gen}


@pytest.fixture
def test_user():
    return {
        "user_id": str(uuid.uuid4()),
        "org_id": str(uuid.uuid4()),
        "roles": ["viewer"],
    }


@pytest.fixture
def test_user_admin():
    return {
        "user_id": str(uuid.uuid4()),
        "org_id": str(uuid.uuid4()),
        "roles": ["admin"],
    }


@pytest.fixture
def test_token():
    from app.auth import create_access_token
    return create_access_token({
        "sub": str(uuid.uuid4()),
        "org_id": str(uuid.uuid4()),
        "roles": ["viewer"],
    })


@pytest.fixture
def auth_headers(test_token):
    return {"Authorization": f"Bearer {test_token}"}


@pytest.fixture
def test_client():
    """Create a TestClient with auth and DB dependency overrides."""
    with patch("app.main.ensure_collection"):
        from app.main import app
        from app.api.deps import get_current_user, get_current_user_optional
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
        app.dependency_overrides[get_current_user_optional] = lambda: test_user
        app.dependency_overrides[get_db] = _override_db

        client = TestClient(app)
        client.mock_session = mock_sess
        client.test_user = test_user
        yield client

        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_current_user_optional, None)
        app.dependency_overrides.pop(get_db, None)


@pytest.fixture
def test_client_with_db(mock_session):
    """Create a TestClient with a specific DB session."""
    with patch("app.main.ensure_collection"):
        from app.main import app
        from app.api.deps import get_current_user, get_current_user_optional
        from app.database import get_db

        test_user = {
            "user_id": str(uuid.uuid4()),
            "org_id": str(uuid.uuid4()),
            "roles": ["viewer"],
        }

        async def _override_db():
            yield mock_session

        app.dependency_overrides[get_current_user] = lambda: test_user
        app.dependency_overrides[get_current_user_optional] = lambda: test_user
        app.dependency_overrides[get_db] = _override_db

        client = TestClient(app)
        yield client

        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_current_user_optional, None)
        app.dependency_overrides.pop(get_db, None)


@pytest.fixture
def mock_session():
    """Standalone mock session fixture."""
    return make_mock_session()


@pytest.fixture
def test_org_id():
    return str(uuid.uuid4())


@pytest.fixture
def test_kb_id():
    return str(uuid.uuid4())


@pytest.fixture
def test_document_id():
    return str(uuid.uuid4())


@pytest.fixture
def test_trace_id():
    return str(uuid.uuid4())
