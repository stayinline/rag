import dashscope
from dashscope import TextEmbedding

from app.config import settings

dashscope.api_key = settings.dashscope_api_key


def embed_texts(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Batch embed texts using DashScope. Returns list of vectors."""
    model = model or settings.embedding_model
    resp = TextEmbedding.call(model=model, input=texts)
    if resp.status_code != 200:
        raise RuntimeError(f"Embedding failed: {resp.code} {resp.message}")
    sorted_items = sorted(resp.output["embeddings"], key=lambda x: x["text_index"])
    return [item["embedding"] for item in sorted_items]


def embed_text(text: str, model: str | None = None) -> list[float]:
    return embed_texts([text], model)[0]
