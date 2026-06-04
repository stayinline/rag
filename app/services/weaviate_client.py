import logging

import weaviate
import weaviate.classes.config as wc
from weaviate.connect import ConnectionParams
from weaviate.classes.config import Configure

from app.config import settings

logger = logging.getLogger(__name__)


def get_client() -> weaviate.WeaviateClient:
    logger.debug("Create Weaviate client url=%s grpc_port=%s", settings.weaviate_url, settings.weaviate_grpc_port)
    return weaviate.WeaviateClient(
        connection_params=ConnectionParams.from_url(
            settings.weaviate_url,
            grpc_port=settings.weaviate_grpc_port,
        ),
    )


COLLECTION_NAME = "KnowledgeChunk"

COLLECTION_PROPERTIES = [
    wc.Property(name="org_id", data_type=wc.DataType.TEXT, index_filterable=True),
    wc.Property(name="kb_id", data_type=wc.DataType.TEXT, index_filterable=True),
    wc.Property(name="document_id", data_type=wc.DataType.TEXT, index_filterable=True),
    wc.Property(name="document_version_id", data_type=wc.DataType.TEXT, index_filterable=True),
    wc.Property(name="chunk_id", data_type=wc.DataType.TEXT, index_filterable=True),
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
        logger.info("Weaviate collection exists collection=%s", COLLECTION_NAME)
