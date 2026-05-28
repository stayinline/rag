"""Initialize Weaviate collection."""
import sys

sys.path.insert(0, ".")

from app.services.weaviate_client import get_client, ensure_collection


def main():
    print("Initializing Weaviate collection...")
    client = get_client()
    client.connect()
    try:
        ensure_collection(client)
        print("Weaviate collection 'KnowledgeChunk' ready.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
