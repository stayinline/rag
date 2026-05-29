"""Tests for metadata enhancement service."""
import pytest
from unittest.mock import patch, MagicMock

from app.services.metadata_enhancer import (
    EnhancedMetadata,
    extract_medical_entities,
    enhance_via_crossref,
    enhance_via_pubmed,
)


class TestEnhancedMetadata:
    def test_default_values(self):
        meta = EnhancedMetadata()
        assert meta.title is None
        assert meta.authors == []
        assert meta.mesh_terms == []
        assert meta.source == "none"


class TestCrossRefEnhancement:
    @patch("app.services.metadata_enhancer.httpx.Client")
    def test_successful_enhancement(self, mock_client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {
                "title": ["Test Paper Title"],
                "author": [
                    {"given": "John", "family": "Smith", "affiliation": [{"name": "University A"}]},
                    {"given": "Jane", "family": "Doe"},
                ],
                "container-title": ["Nature"],
                "published": {"date-parts": [[2023, 5, 15]]},
                "abstract": "This is the abstract.",
            }
        }
        mock_client.return_value.__enter__.return_value.get.return_value = mock_resp

        result = enhance_via_crossref("10.1234/test")

        assert result is not None
        assert result.title == "Test Paper Title"
        assert len(result.authors) == 2
        assert result.authors[0]["name"] == "John Smith"
        assert result.authors[0]["affiliation"] == "University A"
        assert result.journal == "Nature"
        assert result.publication_date == "2023-05-15"
        assert result.abstract == "This is the abstract."
        assert result.source == "crossref"

    @patch("app.services.metadata_enhancer.httpx.Client")
    def test_not_found(self, mock_client):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_client.return_value.__enter__.return_value.get.return_value = mock_resp

        result = enhance_via_crossref("10.9999/notfound")
        assert result is None

    @patch("app.services.metadata_enhancer.httpx.Client")
    def test_connection_error(self, mock_client):
        import httpx
        mock_client.return_value.__enter__.return_value.get.side_effect = httpx.ConnectError("fail")

        result = enhance_via_crossref("10.1234/test")
        assert result is None

    @patch("app.services.metadata_enhancer.httpx.Client")
    def test_partial_date(self, mock_client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {
                "title": ["Test"],
                "published": {"date-parts": [[2023]]},
            }
        }
        mock_client.return_value.__enter__.return_value.get.return_value = mock_resp

        result = enhance_via_crossref("10.1234/test")
        assert result.publication_date == "2023"


class TestPubMedEnhancement:
    @patch("app.services.metadata_enhancer.httpx.Client")
    def test_successful_enhancement(self, mock_client):
        mock_summary = MagicMock()
        mock_summary.status_code = 200
        mock_summary.json.return_value = {
            "result": {
                "12345": {
                    "title": "PubMed Paper Title",
                    "authors": ["Smith J", "Doe A"],
                    "source": "The Lancet",
                    "pubdate": "2023 Jun",
                }
            }
        }

        mock_fetch = MagicMock()
        mock_fetch.status_code = 200
        mock_fetch.text = "<AbstractText>PubMed abstract text.</AbstractText>"

        mock_client_instance = MagicMock()
        mock_client_instance.get.side_effect = [mock_summary, mock_fetch]
        mock_client.return_value.__enter__.return_value = mock_client_instance

        result = enhance_via_pubmed("12345")

        assert result is not None
        assert result.title == "PubMed Paper Title"
        assert len(result.authors) == 2
        assert result.journal == "The Lancet"
        assert result.source == "pubmed"

    @patch("app.services.metadata_enhancer.httpx.Client")
    def test_not_found(self, mock_client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"result": {}}
        mock_client.return_value.__enter__.return_value.get.return_value = mock_resp

        result = enhance_via_pubmed("99999")
        assert result is None


class TestMedicalEntityExtraction:
    def test_disease_detection(self):
        text = "The patient was diagnosed with lung cancer and hypertension."
        entities = extract_medical_entities(text)
        assert "cancer" in " ".join(entities["diseases"]).lower() or "Cancer" in entities["diseases"]

    def test_drug_detection(self):
        text = "Patients received pembrolizumab as first-line treatment."
        entities = extract_medical_entities(text)
        assert "pembrolizumab" in entities["drugs"]

    def test_target_detection(self):
        text = "EGFR mutation status was assessed. PD-1 inhibitors were used."
        entities = extract_medical_entities(text)
        assert "EGFR" in entities["targets"]
        assert "PD-1" in entities["targets"]

    def test_empty_text(self):
        entities = extract_medical_entities("")
        assert entities["diseases"] == []
        assert entities["drugs"] == []
        assert entities["targets"] == []

    def test_no_entities(self):
        text = "This is a general text without medical terms."
        entities = extract_medical_entities(text)
        assert entities["diseases"] == []
        assert entities["drugs"] == []
        assert entities["targets"] == []

    def test_multiple_diseases(self):
        text = "The study included patients with diabetes, obesity, and hypertension."
        entities = extract_medical_entities(text)
        disease_str = " ".join(entities["diseases"]).lower()
        assert "diabetes" in disease_str or "hypertension" in disease_str
