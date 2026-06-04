"""Initialize Weaviate collection."""
import sys

sys.path.insert(0, ".")

from app.services.weaviate_client import close_client, ensure_collection, get_client


def main():
    print("Initializing Weaviate collection...")
    client = get_client()
    try:
        ensure_collection(client)
        print("Weaviate collection 'KnowledgeChunk' ready.")
    finally:
        close_client()


if __name__ == "__main__":
    main()
