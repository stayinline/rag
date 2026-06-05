"""Tests for Weaviate client."""
from unittest.mock import MagicMock, patch

from app.services.weaviate_client import (
    COLLECTION_NAME,
    COLLECTION_PROPERTIES,
    close_client,
    ensure_collection,
    get_client,
)


def test_collection_name():
    assert COLLECTION_NAME == "KnowledgeChunk"


def test_collection_properties():
    assert len(COLLECTION_PROPERTIES) > 0
    prop_names = [p.name for p in COLLECTION_PROPERTIES]
    assert "org_id" in prop_names
    assert "kb_id" in prop_names
    assert "document_id" in prop_names
    assert "content" in prop_names
    assert "status" in prop_names
    assert "security_level" in prop_names


def test_ensure_collection_creates():
    client = MagicMock()
    client.collections.exists = MagicMock(return_value=False)
    client.collections.create = MagicMock()

    ensure_collection(client)
    client.collections.create.assert_called_once()
    call_kwargs = client.collections.create.call_args[1]
    assert call_kwargs["name"] == "KnowledgeChunk"


def test_ensure_collection_skips_existing():
    client = MagicMock()
    client.collections.exists = MagicMock(return_value=True)
    client.collections.create = MagicMock()
    collection = MagicMock()
    client.collections.get.return_value = collection

    ensure_collection(client)

    client.collections.create.assert_not_called()
    client.collections.get.assert_called_once_with(COLLECTION_NAME)
    assert collection.config.add_property.call_count == len(COLLECTION_PROPERTIES)


def test_get_client(mock_settings):
    close_client()
    with patch("app.services.weaviate_client.ConnectionParams") as mock_params, \
         patch("app.services.weaviate_client.weaviate.WeaviateClient") as mock_client_cls, \
         patch("app.services.weaviate_client.settings") as mock_settings:
        mock_settings.weaviate_url = "http://configured-weaviate:18080"
        mock_settings.weaviate_grpc_port = 50052
        connection_params = MagicMock()
        mock_params.from_url.return_value = connection_params
        mock_client = MagicMock()
        mock_client.is_connected.side_effect = [False, True]
        mock_client_cls.return_value = mock_client

        client = get_client()
        same_client = get_client()

        mock_params.from_url.assert_called_once_with(
            "http://configured-weaviate:18080",
            grpc_port=50052,
        )
        mock_client_cls.assert_called_once_with(connection_params=connection_params)
        assert client is mock_client
        assert same_client is mock_client
        mock_client.connect.assert_called_once()
    close_client()


def test_close_client_resets_singleton(mock_settings):
    close_client()
    with patch("app.services.weaviate_client.weaviate.WeaviateClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        mock_client_cls.return_value = mock_client

        client = get_client()
        close_client()

    assert client is mock_client
    mock_client.close.assert_called_once()
