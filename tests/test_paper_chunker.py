"""Tests for paper-specific chunker."""

from app.services.paper_chunker import (
    chunk_paper,
    get_paper_evidence_summary,
    _chunk_references,
)
from app.services.paper_parser import PaperParseResult, PaperSection


class TestChunkPaper:
    def _make_paper(self):
        return PaperParseResult(
            title="Test Paper Title",
            abstract="This is a test abstract with some content.",
            sections=[
                PaperSection(
                    section_type="introduction",
                    heading="Introduction",
                    content="Background information and study rationale.",
                ),
                PaperSection(
                    section_type="methods",
                    heading="Methods",
                    content="Study design, participants, and statistical analysis methods used in this research.",
                ),
                PaperSection(
                    section_type="results",
                    heading="Results",
                    content="Primary and secondary outcomes showed significant differences between groups.",
                ),
                PaperSection(
                    section_type="discussion",
                    heading="Discussion",
                    content="These findings support the hypothesis.",
                ),
                PaperSection(
                    section_type="conclusion",
                    heading="Conclusion",
                    content="In conclusion, the treatment was effective.",
                ),
            ],
            references=[
                {"authors": "Smith J", "title": "Ref One", "year": "2023"},
                {"authors": "Doe A", "title": "Ref Two", "year": "2022"},
            ],
        )

    def test_abstract_is_single_chunk(self):
        paper = self._make_paper()
        chunks = chunk_paper(paper)

        # Abstract should be its own chunk
        abstract_chunks = [c for c in chunks if c.get("section_type") == "abstract"]
        assert len(abstract_chunks) == 1
        assert "Abstract" in abstract_chunks[0]["content"]

    def test_sections_produce_chunks(self):
        paper = self._make_paper()
        chunks = chunk_paper(paper)

        # Should have abstract + 5 sections + reference chunks
        assert len(chunks) >= 5

    def test_section_type_preserved(self):
        paper = self._make_paper()
        chunks = chunk_paper(paper)

        section_types = {c.get("section_type") for c in chunks}
        assert "abstract" in section_types
        assert "introduction" in section_types
        assert "methods" in section_types
        assert "results" in section_types

    def test_section_path_includes_title(self):
        paper = self._make_paper()
        chunks = chunk_paper(paper)

        # Non-reference chunks should include the title in section_path
        non_ref_chunks = [c for c in chunks if c.get("section_type") != "references"]
        for chunk in non_ref_chunks:
            assert "Test Paper Title" in chunk["section_path"]

    def test_abstract_has_highest_boost(self):
        paper = self._make_paper()
        chunks = chunk_paper(paper)

        abstract_chunks = [c for c in chunks if c.get("section_type") == "abstract"]
        assert len(abstract_chunks) == 1
        assert abstract_chunks[0]["boost"] >= 1.0

    def test_references_chunked(self):
        paper = self._make_paper()
        chunks = chunk_paper(paper)

        ref_chunks = [c for c in chunks if c.get("section_type") == "references"]
        assert len(ref_chunks) >= 1

    def test_acknowledgements_skipped(self):
        paper = PaperParseResult(
            title="Test",
            sections=[
                PaperSection(
                    section_type="acknowledgements",
                    heading="Acknowledgements",
                    content="Thanks to everyone.",
                ),
            ],
        )
        chunks = chunk_paper(paper)
        assert len(chunks) == 0  # Only acknowledgements, all skipped

    def test_empty_paper(self):
        paper = PaperParseResult()
        chunks = chunk_paper(paper)
        assert len(chunks) == 0


class TestChunkReferences:
    def test_single_reference(self):
        refs = [{"authors": "Smith J", "title": "Test Paper", "year": "2023"}]
        chunks = _chunk_references(refs)
        assert len(chunks) == 1
        assert "Smith J" in chunks[0]["content"]
        assert "Test Paper" in chunks[0]["content"]
        assert chunks[0]["section_type"] == "references"

    def test_multiple_references(self):
        refs = [
            {"authors": "A", "title": "Paper 1", "year": "2023"},
            {"authors": "B", "title": "Paper 2", "year": "2022"},
            {"authors": "C", "title": "Paper 3", "year": "2021"},
        ]
        chunks = _chunk_references(refs)
        assert len(chunks) == 3
        assert "[1]" in chunks[0]["content"]
        assert "[3]" in chunks[2]["content"]

    def test_empty_references(self):
        chunks = _chunk_references([])
        assert len(chunks) == 0


class TestGetPaperEvidenceSummary:
    def test_basic_summary(self):
        paper = PaperParseResult(
            title="Test",
            abstract="This is the abstract.",
            sections=[
                PaperSection(
                    section_type="methods",
                    heading="Methods",
                    content="Randomized controlled trial with 100 participants.",
                ),
                PaperSection(
                    section_type="results",
                    heading="Results",
                    content="Treatment group showed 50% improvement. P-value was 0.01. Effect size was large.",
                ),
                PaperSection(
                    section_type="conclusion",
                    heading="Conclusion",
                    content="The treatment is effective and safe.",
                ),
            ],
        )

        summary = get_paper_evidence_summary(paper)

        assert summary["abstract"] == "This is the abstract."
        assert summary["study_design"] is not None
        assert len(summary["key_findings"]) > 0
        assert "effective" in summary["conclusion"]

    def test_empty_paper(self):
        paper = PaperParseResult()
        summary = get_paper_evidence_summary(paper)
        assert summary["abstract"] == ""
        assert summary["study_design"] is None
        assert summary["key_findings"] == []
