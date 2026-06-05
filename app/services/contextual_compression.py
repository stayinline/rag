"""Contextual compression for retrieved RAG sources."""
import logging
import re
from dataclasses import dataclass

from app.config import settings

logger = logging.getLogger(__name__)
_SENTENCE_RE = re.compile(r"[^。！？.!?\n]+[。！？.!?]?")
_TOKEN_RE = re.compile(r"[\w\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")


@dataclass(frozen=True)
class CompressionStats:
    input_count: int
    compressed_count: int
    original_chars: int
    compressed_chars: int


def compress_sources_for_query(query: str, sources) -> tuple[list, CompressionStats]:
    """Extract query-relevant sentences from each source while preserving source identity."""
    if not getattr(settings, "contextual_compression", True):
        return sources, CompressionStats(len(sources), 0, _total_chars(sources), _total_chars(sources))

    query_terms = _terms(query)
    max_sentences = max(int(getattr(settings, "contextual_compression_max_sentences", 3) or 3), 1)
    compressed_count = 0
    original_chars = _total_chars(sources)

    for source in sources:
        original = source.content or ""
        compressed = _compress_text(original, query_terms, max_sentences)
        if compressed and len(compressed) < len(original):
            source.content = compressed
            compressed_count += 1

    compressed_chars = _total_chars(sources)
    logger.info(
        "Contextual compression complete query_terms=%s input_count=%s compressed_count=%s original_chars=%s compressed_chars=%s",
        len(query_terms),
        len(sources),
        compressed_count,
        original_chars,
        compressed_chars,
    )
    return sources, CompressionStats(
        input_count=len(sources),
        compressed_count=compressed_count,
        original_chars=original_chars,
        compressed_chars=compressed_chars,
    )


def _compress_text(text: str, query_terms: set[str], max_sentences: int) -> str:
    sentences = [s.strip() for s in _SENTENCE_RE.findall(text or "") if s.strip()]
    if len(sentences) <= max_sentences:
        return text

    scored = []
    for idx, sentence in enumerate(sentences):
        sentence_terms = _terms(sentence)
        overlap = len(query_terms & sentence_terms)
        scored.append((overlap, -idx, sentence))

    selected = [item for item in scored if item[0] > 0]
    if not selected:
        selected = scored[:max_sentences]
    selected = sorted(selected, reverse=True)[:max_sentences]
    selected_sentences = [sentence for _, _, sentence in sorted(selected, key=lambda item: -item[1])]
    return "\n".join(selected_sentences).strip()


def _terms(text: str) -> set[str]:
    return {term.lower() for term in _TOKEN_RE.findall(text or "") if len(term.strip()) >= 2}


def _total_chars(sources) -> int:
    return sum(len(getattr(source, "content", "") or "") for source in sources)
