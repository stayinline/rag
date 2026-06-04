import logging
import time

import dashscope
from dashscope import TextEmbedding

from app.config import settings

logger = logging.getLogger(__name__)
dashscope.api_key = settings.dashscope_api_key


def embed_texts(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Batch embed texts using DashScope. Returns list of vectors."""
    model = model or settings.embedding_model
    t0 = time.monotonic()
    total_chars = sum(len(text or "") for text in texts)
    logger.info("Embedding request start model=%s batch_size=%s total_chars=%s", model, len(texts), total_chars)
    resp = TextEmbedding.call(
        model=model,
        input=texts,
        api_key=settings.dashscope_api_key or None,
        request_timeout=settings.llm_timeout,
    )
    if resp.status_code != 200:
        logger.error(
            "Embedding request failed model=%s batch_size=%s status_code=%s code=%s message=%s duration_ms=%.2f",
            model,
            len(texts),
            resp.status_code,
            getattr(resp, "code", None),
            getattr(resp, "message", None),
            (time.monotonic() - t0) * 1000,
        )
        raise RuntimeError(f"Embedding failed: {resp.code} {resp.message}")
    sorted_items = sorted(resp.output["embeddings"], key=lambda x: x["text_index"])
    vectors = [item["embedding"] for item in sorted_items]
    logger.info(
        "Embedding request complete model=%s batch_size=%s vector_count=%s vector_dims=%s duration_ms=%.2f",
        model,
        len(texts),
        len(vectors),
        len(vectors[0]) if vectors else 0,
        (time.monotonic() - t0) * 1000,
    )
    return vectors


def embed_text(text: str, model: str | None = None) -> list[float]:
    return embed_texts([text], model)[0]
