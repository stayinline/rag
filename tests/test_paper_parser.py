"""Tests for paper parser service."""
import pytest
from unittest.mock import patch, MagicMock

from app.services.paper_parser import (
    PaperParseResult,
    PaperSection,
    _classify_section,
    parse_paper_local,
    paper_to_chunkable_text,
    paper_references_to_text,
)


class TestClassifySection:
    def test_abstract(self):
        assert _classify_section("Abstract") == "abstract"
        assert _classify_section("ABSTRACT") == "abstract"
        assert _classify_section("Summary") == "abstract"

    def test_introduction(self):
        assert _classify_section("Introduction") == "introduction"
        assert _classify_section("Background") == "introduction"

    def test_methods(self):
        assert _classify_section("Methods") == "methods"
        assert _classify_section("Materials and Methods") == "methods"
        assert _classify_section("Study Design") == "methods"

    def test_results(self):
        assert _classify_section("Results") == "results"
        assert _classify_section("Findings") == "results"

    def test_discussion(self):
        assert _classify_section("Discussion") == "discussion"

    def test_conclusion(self):
        assert _classify_section("Conclusion") == "conclusion"
        assert _classify_section("Conclusions") == "conclusion"

    def test_references(self):
        assert _classify_section("References") == "references"
        assert _classify_section("Bibliography") == "references"

    def test_other(self):
        assert _classify_section("Supplementary Material") == "other"
        assert _classify_section("Appendix") == "other"


class TestPaperParseResult:
    def test_default_values(self):
        result = PaperParseResult()
        assert result.title == ""
        assert result.authors == []
        assert result.abstract == ""
        assert result.sections == []
        assert result.references == []
        assert result.parser == "local_fallback"

    def test_with_data(self):
        result = PaperParseResult(
            title="Test Paper",
            abstract="Test abstract",
            sections=[PaperSection(section_type="abstract", heading="Abstract", content="Content")],
            parser="grobid",
            grobid_confidence=0.95,
        )
        assert result.title == "Test Paper"
        assert len(result.sections) == 1
        assert result.parser == "grobid"
        assert result.grobid_confidence == 0.95


class TestPaperToChunkableText:
    def test_basic_conversion(self):
        result = PaperParseResult(
            title="Test Paper",
            abstract="This is the abstract.",
            sections=[
                PaperSection(section_type="introduction", heading="Introduction", content="Intro content"),
                PaperSection(section_type="methods", heading="Methods", content="Method content"),
            ],
        )
        text = paper_to_chunkable_text(result)
        assert "# Test Paper" in text
        assert "## Abstract" in text
        assert "This is the abstract" in text
        assert "## Introduction" in text
        assert "## Methods" in text

    def test_skips_references(self):
        result = PaperParseResult(
            title="Test",
            sections=[
                PaperSection(section_type="references", heading="References", content="Ref 1"),
                PaperSection(section_type="results", heading="Results", content="Result content"),
            ],
        )
        text = paper_to_chunkable_text(result)
        assert "## References" not in text
        assert "## Results" in text

    def test_with_authors_and_journal(self):
        result = PaperParseResult(
            title="Test",
            authors=[{"name": "Smith, J"}, {"name": "Doe, A"}],
            journal="Nature",
        )
        text = paper_to_chunkable_text(result)
        assert "Authors:" in text
        assert "Smith" in text
        assert "Journal: Nature" in text


class TestPaperReferencesToText:
    def test_empty_references(self):
        result = PaperParseResult()
        text = paper_references_to_text(result)
        assert text == ""

    def test_with_references(self):
        result = PaperParseResult(
            references=[
                {"authors": "Smith J", "title": "Paper One", "year": "2023"},
                {"authors": "Doe A", "title": "Paper Two", "year": "2022"},
            ]
        )
        text = paper_references_to_text(result)
        assert "## References" in text
        assert "[1]" in text
        assert "[2]" in text
        assert "Paper One" in text
        assert "2023" in text


class TestParsePaperLocal:
    def test_creates_result(self):
        """Test that local parser creates a PaperParseResult."""
        import fitz

        with patch.object(fitz, "open") as mock_open:
            mock_doc = MagicMock()
            mock_page = MagicMock()
            mock_page.get_text.return_value = (
                "# Title of the Paper\n\n"
                "Abstract\n"
                "This is the abstract of the paper.\n"
                "It contains multiple sentences.\n\n"
                "Introduction\n"
                "This is the introduction section.\n\n"
                "METHODS\n"
                "Study design and methodology.\n\n"
                "RESULTS\n"
                "Key findings and data.\n\n"
                "DISCUSSION\n"
                "Interpretation of results.\n\n"
                "CONCLUSION\n"
                "Final remarks.\n"
            )
            mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
            mock_doc.__enter__ = MagicMock(return_value=mock_doc)
            mock_doc.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_doc

            result = parse_paper_local("/fake/path.pdf")

            assert isinstance(result, PaperParseResult)
            assert result.parser == "local_fallback"

    def test_section_classification(self):
        """Test that sections are properly classified."""
        import fitz

        with patch.object(fitz, "open") as mock_open:
            mock_doc = MagicMock()
            mock_page = MagicMock()
            mock_page.get_text.return_value = (
                "Abstract\nThis is a test abstract.\n"
                "Introduction\nBackground info.\n"
                "Results\nFindings here.\n"
                "References\nBibliography here.\n"
            )
            mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
            mock_doc.__enter__ = MagicMock(return_value=mock_doc)
            mock_doc.__exit__ = MagicMock(return_value=False)
            mock_open.return_value = mock_doc

            result = parse_paper_local("/fake/path.pdf")

            # Verify sections were found
            section_types = [s.section_type for s in result.sections]
            # At least some sections should be classified
            assert "results" in section_types or "introduction" in section_types
