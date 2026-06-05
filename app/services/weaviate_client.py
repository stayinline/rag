import logging
import threading

import weaviate
import weaviate.classes.config as wc
from weaviate.connect import ConnectionParams
from weaviate.classes.config import Configure

from app.config import settings

logger = logging.getLogger(__name__)

_client: weaviate.WeaviateClient | None = None
_client_lock = threading.Lock()


def get_client() -> weaviate.WeaviateClient:
    global _client

    with _client_lock:
        if _client is None:
            logger.info(
                "Create shared Weaviate client url=%s grpc_port=%s",
                settings.weaviate_url,
                settings.weaviate_grpc_port,
            )
            _client = weaviate.WeaviateClient(
                connection_params=ConnectionParams.from_url(
                    settings.weaviate_url,
                    grpc_port=settings.weaviate_grpc_port,
                ),
            )

        if not _client.is_connected():
            logger.debug("Connect shared Weaviate client")
            _client.connect()

    return _client


def close_client() -> None:
    global _client

    with _client_lock:
        if _client is not None:
            _client.close()
            _client = None


COLLECTION_NAME = "KnowledgeChunk"

COLLECTION_PROPERTIES = [
    wc.Property(name="org_id", data_type=wc.DataType.TEXT, index_filterable=True),
    wc.Property(name="kb_id", data_type=wc.DataType.TEXT, index_filterable=True),
    wc.Property(name="document_id", data_type=wc.DataType.TEXT, index_filterable=True),
    wc.Property(name="document_version_id", data_type=wc.DataType.TEXT, index_filterable=True),
    wc.Property(name="chunk_id", data_type=wc.DataType.TEXT, index_filterable=True),
    wc.Property(name="chunk_index", data_type=wc.DataType.INT, index_filterable=True),
    wc.Property(name="parent_chunk_id", data_type=wc.DataType.TEXT, index_filterable=True),
    wc.Property(name="child_chunk_ids", data_type=wc.DataType.TEXT_ARRAY, index_filterable=True),
    wc.Property(name="acl_hash", data_type=wc.DataType.TEXT, index_filterable=True),
    wc.Property(name="security_level", data_type=wc.DataType.TEXT, index_filterable=True),
    wc.Property(name="status", data_type=wc.DataType.TEXT, index_filterable=True),
    wc.Property(name="content", data_type=wc.DataType.TEXT, index_searchable=True),
    wc.Property(name="title", data_type=wc.DataType.TEXT, index_searchable=True),
    wc.Property(name="section_path", data_type=wc.DataType.TEXT, index_searchable=True),
    wc.Property(name="page_start", data_type=wc.DataType.INT),
    wc.Property(name="page_end", data_type=wc.DataType.INT),
    wc.Property(name="document_type", data_type=wc.DataType.TEXT, index_filterable=True),
    wc.Property(name="domain_tags", data_type=wc.DataType.TEXT_ARRAY, index_filterable=True),
    wc.Property(name="entities", data_type=wc.DataType.TEXT_ARRAY, index_filterable=True),
    wc.Property(name="publication_date", data_type=wc.DataType.DATE),
    wc.Property(name="embedding_model", data_type=wc.DataType.TEXT, index_filterable=True),
    wc.Property(name="created_at", data_type=wc.DataType.DATE),
    wc.Property(name="section_type", data_type=wc.DataType.TEXT, index_filterable=True),
]


def ensure_collection(client: weaviate.WeaviateClient) -> None:
    logger.info("Ensure Weaviate collection start collection=%s", COLLECTION_NAME)
    if not client.collections.exists(COLLECTION_NAME):
        logger.info(
            "Weaviate collection missing; creating collection=%s property_count=%s",
            COLLECTION_NAME,
            len(COLLECTION_PROPERTIES),
        )
        client.collections.create(
            name=COLLECTION_NAME,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=COLLECTION_PROPERTIES,
        )
        logger.info("Weaviate collection created collection=%s", COLLECTION_NAME)
    else:
        collection = client.collections.get(COLLECTION_NAME)
        _ensure_collection_properties(collection)
        logger.info("Weaviate collection exists collection=%s", COLLECTION_NAME)


def _ensure_collection_properties(collection) -> None:
    for prop in COLLECTION_PROPERTIES:
        try:
            collection.config.add_property(prop)
            logger.info("Weaviate collection property added collection=%s property=%s", COLLECTION_NAME, prop.name)
        except Exception as exc:
            message = str(exc).lower()
            if "already" in message or "exist" in message:
                continue
            logger.warning(
                "Weaviate collection property add skipped collection=%s property=%s error=%s",
                COLLECTION_NAME,
                prop.name,
                exc,
            )
