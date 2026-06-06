from app.services.feedback_learning import apply_feedback_weights, build_feedback_weights
from app.services.rag import RAGSource


def test_build_feedback_weights_from_rated_sources():
    weights = build_feedback_weights([
        (5, [{"chunk_id": "c1", "document_id": "d1"}]),
        (1, [{"chunk_id": "c2", "document_id": "d2"}]),
        (3, [{"chunk_id": "ignored", "document_id": "ignored"}]),
    ])

    assert weights.sample_count == 2
    assert weights.chunk_weights["c1"] == 1.0
    assert weights.document_weights["d2"] == -1.0
    assert "ignored" not in weights.chunk_weights


def test_apply_feedback_weights_adjusts_order():
    good = RAGSource("c1", "d1", "Good", None, None, None, 0.2, "good")
    bad = RAGSource("c2", "d2", "Bad", None, None, None, 0.8, "bad")
    weights = build_feedback_weights([
        (5, [{"chunk_id": "c1", "document_id": "d1"}]),
        (1, [{"chunk_id": "c2", "document_id": "d2"}]),
    ])

    reranked = apply_feedback_weights([bad, good], weights, strength=0.5)

    assert reranked[0].chunk_id == "c1"
    assert reranked[0].feedback_score > 0
    assert reranked[1].feedback_score < 0
