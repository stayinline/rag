"""Citation post-processing for generated RAG answers."""
import re
from dataclasses import dataclass
from typing import Any

from app.config import settings

_CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class CitationValidationResult:
    answer: str
    used_citation_numbers: list[int]
    invalid_citation_numbers: list[int]
    low_confidence_citation_numbers: list[int]
    citation_count: int
    is_valid: bool


def validate_answer_citations(answer: str, citations: list[dict[str, Any]]) -> CitationValidationResult:
    """Validate numeric citations in an answer against the provided citation list."""
    citation_count = len(citations)
    used_numbers = sorted({int(match) for match in _CITATION_RE.findall(answer or "")})
    invalid_numbers = [number for number in used_numbers if number < 1 or number > citation_count]
    low_confidence_numbers = _low_confidence_numbers(used_numbers, citations)

    notes = []
    if invalid_numbers:
        notes.append(f"未能对齐的引用编号：{', '.join(f'[{number}]' for number in invalid_numbers)}。")
    if low_confidence_numbers:
        notes.append(f"低置信度引用编号：{', '.join(f'[{number}]' for number in low_confidence_numbers)}。")

    validated_answer = answer or ""
    if notes:
        validated_answer = validated_answer.rstrip()
        if validated_answer:
            validated_answer += "\n\n"
        validated_answer += "引用校验提示：" + " ".join(notes)

    return CitationValidationResult(
        answer=validated_answer,
        used_citation_numbers=used_numbers,
        invalid_citation_numbers=invalid_numbers,
        low_confidence_citation_numbers=low_confidence_numbers,
        citation_count=citation_count,
        is_valid=not invalid_numbers and not low_confidence_numbers,
    )


def _low_confidence_numbers(used_numbers: list[int], citations: list[dict[str, Any]]) -> list[int]:
    min_score = float(getattr(settings, "citation_min_score", 0.0) or 0.0)
    if min_score <= 0:
        return []

    low_confidence = []
    for number in used_numbers:
        if number < 1 or number > len(citations):
            continue
        score = _to_float(citations[number - 1].get("score"))
        if score < min_score:
            low_confidence.append(number)
    return low_confidence


def _to_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
