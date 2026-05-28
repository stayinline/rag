from urllib.parse import urlparse

import weaviate
import weaviate.classes.config as wc
from weaviate.classes.config import Configure

from app.config import settings


def get_client() -> weaviate.WeaviateClient:
    parsed = urlparse(settings.weaviate_url)
    return weaviate.connect_to_local(
        host=parsed.hostname or "localhost",
        port=parsed.port or 8080,
        http_secure=parsed.scheme == "https",
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
    wc.Property(name="embedding_model", data_type=wc.DataType.TEXT, index_filterable=True),
    wc.Property(name="created_at", data_type=wc.DataType.DATE),
]


def ensure_collection(client: weaviate.WeaviateClient) -> None:
    if not client.collections.exists(COLLECTION_NAME):
        client.collections.create(
            name=COLLECTION_NAME,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=COLLECTION_PROPERTIES,
        )
