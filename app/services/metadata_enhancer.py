"""CrossRef / PubMed / MeSH metadata enhancement for papers."""
import json
import logging
import time
from dataclasses import dataclass, field

import httpx

from app.services.paper_parser import PaperParseResult

logger = logging.getLogger(__name__)


@dataclass
class EnhancedMetadata:
    """Enhanced metadata from external sources."""
    title: str | None = None
    authors: list[dict] = field(default_factory=list)
    journal: str | None = None
    publication_date: str | None = None
    abstract: str | None = None
    mesh_terms: list[str] = field(default_factory=list)
    diseases: list[str] = field(default_factory=list)
    drugs: list[str] = field(default_factory=list)
    source: str = "none"  # crossref, pubmed, mesh, none


def enhance_via_crossref(doi: str) -> EnhancedMetadata | None:
    """Look up paper metadata via CrossRef API."""
    t0 = time.monotonic()
    logger.info("CrossRef lookup start doi=%s", doi)
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(f"https://api.crossref.org/works/{doi}")
            if resp.status_code != 200:
                logger.warning(
                    "CrossRef lookup returned non-200 doi=%s status_code=%s duration_ms=%.2f",
                    doi,
                    resp.status_code,
                    (time.monotonic() - t0) * 1000,
                )
                return None
            data = resp.json()["message"]
            metadata = EnhancedMetadata(source="crossref")

            metadata.title = data.get("title", [""])[0] if data.get("title") else None

            # Authors
            authors_data = data.get("author", [])
            metadata.authors = [
                {
                    "name": f"{a.get('given', '')} {a.get('family', '')}".strip(),
                    "affiliation": (a.get("affiliation", [{}])[0].get("name")
                                    if a.get("affiliation") else None),
                }
                for a in authors_data[:20]
            ]

            metadata.journal = data.get("container-title", [""])[0] if data.get("container-title") else None

            # Publication date
            if data.get("published") and data["published"].get("date-parts"):
                parts = data["published"]["date-parts"][0]
                if len(parts) >= 3:
                    metadata.publication_date = f"{parts[0]:04d}-{parts[1]:02d}-{parts[2]:02d}"
                elif len(parts) >= 2:
                    metadata.publication_date = f"{parts[0]:04d}-{parts[1]:02d}"
                else:
                    metadata.publication_date = f"{parts[0]:04d}"

            metadata.abstract = data.get("abstract")
            logger.info(
                "CrossRef lookup complete doi=%s has_title=%s author_count=%s has_abstract=%s duration_ms=%.2f",
                doi,
                bool(metadata.title),
                len(metadata.authors),
                bool(metadata.abstract),
                (time.monotonic() - t0) * 1000,
            )
            return metadata

    except (httpx.RequestError, KeyError, IndexError, json.JSONDecodeError) as e:
        logger.warning("CrossRef lookup failed for DOI %s: %s", doi, e)
        return None


def enhance_via_pubmed(pmid: str) -> EnhancedMetadata | None:
    """Look up paper metadata via PubMed E-utilities."""
    t0 = time.monotonic()
    logger.info("PubMed lookup start pmid=%s", pmid)
    try:
        with httpx.Client(timeout=15) as client:
            # Fetch summary
            resp = client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                params={"db": "pubmed", "id": pmid, "retmode": "json"},
            )
            if resp.status_code != 200:
                logger.warning(
                    "PubMed summary lookup returned non-200 pmid=%s status_code=%s duration_ms=%.2f",
                    pmid,
                    resp.status_code,
                    (time.monotonic() - t0) * 1000,
                )
                return None
            result = resp.json()
            docsum = result.get("result", {}).get(pmid)
            if not docsum:
                logger.warning("PubMed lookup returned no document pmid=%s duration_ms=%.2f", pmid, (time.monotonic() - t0) * 1000)
                return None

            metadata = EnhancedMetadata(source="pubmed")
            metadata.title = docsum.get("title")
            metadata.authors = [
                {"name": name.strip()} for name in docsum.get("authors", [])[:20]
            ]
            metadata.journal = docsum.get("source")

            pub_date = docsum.get("pubdate")
            if pub_date:
                metadata.publication_date = pub_date

            # Fetch abstract
            abs_resp = client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params={"db": "pubmed", "id": pmid, "retmode": "xml"},
            )
            if abs_resp.status_code == 200:
                abstract_text = _extract_abstract_from_xml(abs_resp.text)
                if abstract_text:
                    metadata.abstract = abstract_text

            # Fetch MeSH terms
            mesh_text = _extract_mesh_from_xml(abs_resp.text)
            if mesh_text:
                metadata.mesh_terms = mesh_text

            logger.info(
                "PubMed lookup complete pmid=%s has_title=%s author_count=%s mesh_count=%s has_abstract=%s duration_ms=%.2f",
                pmid,
                bool(metadata.title),
                len(metadata.authors),
                len(metadata.mesh_terms),
                bool(metadata.abstract),
                (time.monotonic() - t0) * 1000,
            )
            return metadata

    except (httpx.RequestError, KeyError, json.JSONDecodeError) as e:
        logger.warning("PubMed lookup failed for PMID %s: %s", pmid, e)
        return None


def _extract_abstract_from_xml(xml_text: str) -> str | None:
    """Extract abstract text from PubMed XML."""
    try:
        import re
        # Simple regex-based extraction for AbstractText elements
        abstract_parts = re.findall(r'<AbstractText[^>]*>(.*?)</AbstractText>', xml_text, re.DOTALL)
        if abstract_parts:
            return " ".join(p.strip() for p in abstract_parts)
    except Exception:
        pass
    return None


def _extract_mesh_from_xml(xml_text: str) -> list[str]:
    """Extract MeSH terms from PubMed XML."""
    try:
        import re
        descriptors = re.findall(r'<MeshHeading>.*?<DescriptorName[^>]*>(.*?)</DescriptorName>.*?</MeshHeading>', xml_text, re.DOTALL)
        return [d.strip() for d in descriptors if d.strip()]
    except Exception:
        return []


def extract_medical_entities(text: str) -> dict:
    """Extract medical entities from text using keyword matching.

    This is a lightweight local extraction. For production, consider
    a dedicated NER model (scispaCy, BioBERT, etc.).
    """
    import re

    # Common disease patterns
    disease_keywords = [
        r'\b(?:cancer|tumor|carcinoma|adenoma|melanoma|lymphoma|leukemia|sarcoma|glioma|neoplasm)\b',
        r'\b(?:diabetes|hypertension|obesity|asthma|arthritis|pneumonia|hepatitis|cirrhosis)\b',
        r'\b(?:alzheimer|parkinson|epilepsy|migraine|sclerosis|depression|anxiety)\b',
        r'\b(?:syndrome|disease|disorder|infection|inflammation)\b',
    ]

    # Common drug patterns (generic names with common suffixes)
    drug_keywords = [
        r'\b\w+(?:mab|statin|pril|sartan|oxetine|oxetine|triptan|cillin|mycin|cycline|vir)\b',
        r'\b(?:aspirin|ibuprofen|metformin|atorvastatin|lisinopril|omeprazole|amlodipine)\b',
        r'\b(?:pembrolizumab|nivolumab|trastuzumab|bevacizumab|rituximab)\b',
    ]

    # Common target/biomarker patterns
    target_keywords = [
        r'\b(?:EGFR|HER2|PD-1|PD-L1|VEGF|BRAF|KRAS|PIK3CA|ALK|ROS1|MET|RET)\b',
        r'\b(?:BRCA1|BRCA2|TP53|MYC|RB1|PTEN|AKT|mTOR|ERK|JAK|STAT)\b',
    ]

    text_upper = text.upper()
    text_lower = text.lower()

    diseases = set()
    for pattern in disease_keywords:
        for m in re.finditer(pattern, text_lower):
            diseases.add(m.group().title())

    drugs = set()
    for pattern in drug_keywords:
        for m in re.finditer(pattern, text_lower, re.IGNORECASE):
            drugs.add(m.group().lower())

    targets = set()
    for pattern in target_keywords:
        for m in re.finditer(pattern, text_upper):
            targets.add(m.group())

    entities = {
        "diseases": sorted(diseases),
        "drugs": sorted(drugs),
        "targets": sorted(targets),
    }
    logger.info(
        "Extract medical entities complete text_length=%s disease_count=%s drug_count=%s target_count=%s",
        len(text or ""),
        len(entities["diseases"]),
        len(entities["drugs"]),
        len(entities["targets"]),
    )
    return entities


def enhance_paper(result: PaperParseResult, doi: str | None = None, pmid: str | None = None) -> PaperParseResult:
    """Enhance a parsed paper with external metadata from CrossRef, PubMed, MeSH."""
    logger.info(
        "Enhance paper start doi=%s pmid=%s title_present=%s section_count=%s",
        doi,
        pmid,
        bool(result.title),
        len(result.sections),
    )
    if doi:
        crossref_meta = enhance_via_crossref(doi)
        if crossref_meta:
            if not result.title and crossref_meta.title:
                result.title = crossref_meta.title
            if not result.authors and crossref_meta.authors:
                result.authors = crossref_meta.authors
            if not result.journal and crossref_meta.journal:
                result.journal = crossref_meta.journal
            if not result.publication_date and crossref_meta.publication_date:
                result.publication_date = crossref_meta.publication_date
            if not result.abstract and crossref_meta.abstract:
                result.abstract = crossref_meta.abstract

    if pmid:
        pubmed_meta = enhance_via_pubmed(pmid)
        if pubmed_meta:
            if not result.title and pubmed_meta.title:
                result.title = pubmed_meta.title
            if not result.authors and pubmed_meta.authors:
                result.authors = pubmed_meta.authors
            if not result.journal and pubmed_meta.journal:
                result.journal = pubmed_meta.journal
            if not result.publication_date and pubmed_meta.publication_date:
                result.publication_date = pubmed_meta.publication_date
            if not result.abstract and pubmed_meta.abstract:
                result.abstract = pubmed_meta.abstract
            if pubmed_meta.mesh_terms:
                pass  # Mesh terms will be stored on the Paper model

    # Extract medical entities from the full text
    full_text = result.abstract + " " + " ".join(s.content for s in result.sections)
    entities = extract_medical_entities(full_text)

    logger.info(
        "Enhance paper complete doi=%s pmid=%s title_present=%s entity_counts=%s",
        doi,
        pmid,
        bool(result.title),
        {k: len(v) for k, v in entities.items()},
    )
    return result, entities
