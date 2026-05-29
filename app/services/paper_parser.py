"""SCI PDF paper parser with GROBID integration and local fallback."""
import json
import re
import logging
from dataclasses import dataclass, field

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class PaperSection:
    """A structured section from a parsed paper."""
    section_type: str  # abstract, introduction, methods, results, discussion, conclusion, references, other
    heading: str
    content: str
    page_start: int | None = None
    page_end: int | None = None


@dataclass
class PaperParseResult:
    """Complete result from parsing a SCI PDF."""
    title: str = ""
    authors: list[dict] = field(default_factory=list)
    abstract: str = ""
    journal: str | None = None
    doi: str | None = None
    publication_date: str | None = None
    sections: list[PaperSection] = field(default_factory=list)
    references: list[dict] = field(default_factory=list)
    raw_text: str = ""
    parser: str = "local_fallback"  # grobid or local_fallback
    grobid_confidence: float | None = None


# Section type mapping keywords
SECTION_TYPE_MAP = {
    "abstract": ["abstract", "summary"],
    "introduction": ["introduction", "background", "intro"],
    "methods": ["methods", "methodology", "materials and methods", "experimental", "study design", "materials"],
    "results": ["results", "findings", "outcomes", "results and analysis"],
    "discussion": ["discussion", "interpretation"],
    "conclusion": ["conclusion", "conclusions", "summary and conclusion"],
    "references": ["references", "bibliography", "literature cited"],
    "acknowledgements": ["acknowledgements", "acknowledgments", "funding"],
}


def _classify_section(heading: str) -> str:
    """Classify a section heading into a standard section type."""
    heading_lower = heading.lower().strip().rstrip(".")
    for section_type, keywords in SECTION_TYPE_MAP.items():
        for kw in keywords:
            if kw in heading_lower:
                return section_type
    return "other"


def _parse_grobid_tei(xml_text: str) -> PaperParseResult:
    """Parse GROBID TEI XML output into structured result."""
    try:
        from xml.etree.ElementTree import fromstring
    except Exception:
        raise RuntimeError("XML parsing not available")

    root = fromstring(xml_text)
    result = PaperParseResult(parser="grobid")

    # Extract title
    title_elem = root.find(".//{http://www.tei-c.org/ns/1.0}titleStmt/{http://www.tei-c.org/ns/1.0}title[@level='a']")
    if title_elem is not None and title_elem.text:
        result.title = title_elem.text.strip()

    # Extract abstract
    abstract_elem = root.find(".//{http://www.tei-c.org/ns/1.0}abstract")
    if abstract_elem is not None:
        result.abstract = "".join(abstract_elem.itertext()).strip()

    # Extract sections from body
    body = root.find(".//{http://www.tei-c.org/ns/1.0}body")
    if body is not None:
        for div in body.findall(".//{http://www.tei-c.org/ns/1.0}div"):
            heading_elem = div.find("{http://www.tei-c.org/ns/1.0}head")
            heading = heading_elem.text.strip() if heading_elem is not None and heading_elem.text else "Untitled"
            section_type = _classify_section(heading)
            content = "".join(p.text or "" for p in div.findall(".//{http://www.tei-c.org/ns/1.0}p") if p.text)
            if content.strip():
                result.sections.append(PaperSection(
                    section_type=section_type,
                    heading=heading,
                    content=content.strip(),
                ))

    # Extract references
    back = root.find(".//{http://www.tei-c.org/ns/1.0}back")
    if back is not None:
        for bibl in back.findall(".//{http://www.tei-c.org/ns/1.0}biblStruct"):
            ref = {}
            title_elem = bibl.find(".//{http://www.tei-c.org/ns/1.0}title[@level='a']")
            if title_elem is not None and title_elem.text:
                ref["title"] = title_elem.text.strip()
            author_elems = bibl.findall(".//{http://www.tei-c.org/ns/1.0}author")
            if author_elems:
                ref["authors"] = ", ".join(
                    a.findtext(".//{http://www.tei-c.org/ns/1.0}surname", "")
                    for a in author_elems
                    if a.findtext(".//{http://www.tei-c.org/ns/1.0}surname")
                )
            date_elem = bibl.find(".//{http://www.tei-c.org/ns/1.0}date")
            if date_elem is not None and date_elem.get("when"):
                ref["year"] = date_elem.get("when")[:4]
            if ref:
                result.references.append(ref)

    return result


def parse_paper_grobid(pdf_path: str, grobid_url: str | None = None) -> PaperParseResult:
    """Parse a SCI PDF using GROBID service."""
    grobid_url = grobid_url or getattr(settings, "grobid_url", "http://localhost:8070")

    try:
        with open(pdf_path, "rb") as f:
            files = {"input": ("paper.pdf", f, "application/pdf")}
            with httpx.Client(timeout=120) as client:
                resp = client.post(
                    f"{grobid_url}/api/processFulltextDocument",
                    files=files,
                    data={"generateIDs": "true", "consolidateHeader": "1", "consolidateCitations": "1"},
                )
                resp.raise_for_status()
                return _parse_grobid_tei(resp.text)
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
        logger.warning("GROBID unavailable (%s), falling back to local parser: %s", grobid_url, e)
        return parse_paper_local(pdf_path)


def parse_paper_local(pdf_path: str) -> PaperParseResult:
    """Parse a SCI PDF using local PyMuPDF as fallback."""
    import fitz

    doc = fitz.open(pdf_path)
    result = PaperParseResult(parser="local_fallback")

    all_text = ""
    page_texts = []
    for page in doc:
        text = page.get_text("text")
        page_texts.append(text)
        all_text += text + "\n"

    doc.close()
    result.raw_text = all_text

    lines = all_text.split("\n")

    # Extract title (first non-empty line, typically)
    for line in lines[:10]:
        stripped = line.strip()
        if stripped and len(stripped) > 10:
            result.title = stripped
            break

    # Extract abstract
    abstract_start = -1
    abstract_end = -1
    for i, line in enumerate(lines):
        lower = line.lower().strip()
        if lower.startswith("abstract") or lower == "abstract":
            abstract_start = i + 1
        elif abstract_start >= 0 and (
            lower.startswith("introduction")
            or lower.startswith("keywords")
            or lower.startswith("1.")
            or lower.startswith("1 ")
        ):
            abstract_end = i
            break

    if abstract_start >= 0:
        if abstract_end < 0:
            abstract_end = min(abstract_start + 30, len(lines))
        result.abstract = "\n".join(lines[abstract_start:abstract_end]).strip()

    # Extract sections using heading patterns
    heading_re = re.compile(
        r"^(#{1,3}\s*)?"
        r"(\d+[\.\s]+)?"
        r"([A-Z][A-Za-z\s&]+)"
        r"$"
    )

    sections_text = "\n".join(lines[abstract_end + 1:]) if abstract_end >= 0 else all_text
    section_lines = sections_text.split("\n")

    current_heading = ""
    current_content = []
    for line in section_lines:
        match = heading_re.match(line.strip())
        if match and len(line.strip()) < 100 and line.strip().isupper() or (
            match and len(line.strip()) < 100 and line.strip()[0].isupper()
            and line.strip().istitle()
        ):
            if current_content:
                content = "\n".join(current_content).strip()
                if content:
                    result.sections.append(PaperSection(
                        section_type=_classify_section(current_heading),
                        heading=current_heading,
                        content=content,
                    ))
            current_heading = line.strip()
            current_content = []
        else:
            current_content.append(line)

    if current_content:
        content = "\n".join(current_content).strip()
        if content:
            result.sections.append(PaperSection(
                section_type=_classify_section(current_heading),
                heading=current_heading,
                content=content,
            ))

    return result


def paper_to_chunkable_text(result: PaperParseResult) -> str:
    """Convert a PaperParseResult into text suitable for chunking."""
    parts = []

    if result.title:
        parts.append(f"# {result.title}")

    if result.authors:
        author_str = ", ".join(
            a.get("name", a) if isinstance(a, dict) else str(a)
            for a in result.authors[:10]
        )
        parts.append(f"Authors: {author_str}")

    if result.journal:
        parts.append(f"Journal: {result.journal}")

    if result.publication_date:
        parts.append(f"Publication Date: {result.publication_date}")

    if result.doi:
        parts.append(f"DOI: {result.doi}")

    if result.abstract:
        parts.append(f"\n## Abstract\n{result.abstract}")

    for section in result.sections:
        if section.section_type == "references":
            continue  # References handled separately
        parts.append(f"\n## {section.heading}\n{section.content}")

    return "\n".join(parts)


def paper_references_to_text(result: PaperParseResult) -> str:
    """Convert references to chunkable text."""
    if not result.references:
        return ""
    parts = ["## References"]
    for i, ref in enumerate(result.references, 1):
        ref_parts = []
        if ref.get("authors"):
            ref_parts.append(str(ref["authors"]))
        if ref.get("title"):
            ref_parts.append(f"\"{ref['title']}\"")
        if ref.get("year"):
            ref_parts.append(f"({ref['year']})")
        parts.append(f"[{i}] {'. '.join(ref_parts)}")
    return "\n".join(parts)
