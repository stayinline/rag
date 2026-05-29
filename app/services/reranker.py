"""Reranker service with abstract layer for pluggable backends."""
import logging
from abc import ABC, abstractmethod

from app.config import settings

logger = logging.getLogger(__name__)


class RerankerResult:
    """Result from reranking."""
    def __init__(self, index: int, score: float):
        self.index = index  # Original index in input list
        self.score = score


class BaseReranker(ABC):
    """Abstract reranker interface."""

    @abstractmethod
    def rerank(self, query: str, documents: list[str]) -> list[RerankerResult]:
        """Rerank documents by relevance to query. Returns sorted by score desc."""
        ...


class MockReranker(BaseReranker):
    """Mock reranker for development/testing.

    Uses a simple heuristic: longer documents with more query term overlap
    get higher scores.
    """

    def rerank(self, query: str, documents: list[str]) -> list[RerankerResult]:
        query_terms = set(query.lower().split())
        results = []
        for i, doc in enumerate(documents):
            doc_terms = set(doc.lower().split())
            overlap = len(query_terms & doc_terms)
            # Normalize by query length
            score = overlap / max(len(query_terms), 1)
            # Bonus for document length (more content = potentially more relevant)
            length_bonus = min(len(doc) / 1000, 0.1)
            results.append(RerankerResult(index=i, score=score + length_bonus))
        results.sort(key=lambda r: r.score, reverse=True)
        return results


class BM25Reranker(BaseReranker):
    """Simple BM25-based reranker (no external model dependency)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def rerank(self, query: str, documents: list[str]) -> list[RerankerResult]:
        import math

        query_terms = query.lower().split()
        n_docs = len(documents)

        # Compute document term frequency and document frequency
        doc_term_counts = []
        doc_lengths = []
        doc_freq = {}  # Number of documents containing each term

        for doc in documents:
            terms = doc.lower().split()
            doc_lengths.append(len(terms))
            term_counts = {}
            doc_terms_seen = set()
            for t in terms:
                term_counts[t] = term_counts.get(t, 0) + 1
                if t not in doc_terms_seen:
                    doc_terms_seen.add(t)
                    doc_freq[t] = doc_freq.get(t, 0) + 1
            doc_term_counts.append(term_counts)

        avg_doc_len = sum(doc_lengths) / max(n_docs, 1)

        results = []
        for i, term_counts in enumerate(doc_term_counts):
            score = 0.0
            for term in query_terms:
                tf = term_counts.get(term, 0)
                if tf == 0:
                    continue
                df = doc_freq.get(term, 0)
                idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_lengths[i] / avg_doc_len)
                score += idf * numerator / max(denominator, 1e-6)
            results.append(RerankerResult(index=i, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results


def get_reranker() -> BaseReranker:
    """Get configured reranker instance."""
    reranker_type = getattr(settings, "reranker_type", "bm25")
    if reranker_type == "mock":
        return MockReranker()
    elif reranker_type == "bm25":
        return BM25Reranker()
    else:
        logger.warning("Unknown reranker type '%s', falling back to BM25", reranker_type)
        return BM25Reranker()
