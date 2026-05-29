"""Section-aware chunking for scientific papers."""
from app.config import settings
from app.services.chunker import count_tokens
from app.services.paper_parser import PaperParseResult, PaperSection


# Section importance for retrieval boosting
SECTION_BOOST = {
    "abstract": 1.5,
    "results": 1.3,
    "conclusion": 1.2,
    "discussion": 1.1,
    "methods": 1.0,
    "introduction": 0.9,
    "references": 0.5,
    "other": 0.8,
}


def chunk_paper(
    result: PaperParseResult,
    title: str = "",
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[dict]:
    """Chunk a parsed paper using section-aware strategy.

    Different from general text chunking:
    - Abstract is kept as a single chunk (usually < 300 words)
    - Methods/Results/Discussion are split by token boundaries
    - References are chunked individually (each reference is a chunk)
    - Section type and page info are preserved in metadata
    """
    chunk_size = chunk_size or settings.rag_chunk_size
    overlap = overlap or settings.rag_chunk_overlap

    doc_title = title or result.title
    chunks = []

    # 1. Abstract as a single chunk (high importance)
    if result.abstract.strip():
        chunks.append({
            "content": f"# Abstract\n{result.abstract.strip()}",
            "section_path": f"{doc_title}/Abstract",
            "section_type": "abstract",
            "page_start": None,
            "page_end": None,
            "boost": SECTION_BOOST["abstract"],
        })

    # 2. Process each section
    for section in result.sections:
        if section.section_type == "references":
            continue  # References handled separately
        if section.section_type == "acknowledgements":
            continue  # Skip acknowledgements (low value)

        section_path = f"{doc_title}/{section.heading}".strip("/")
        content = f"# {section.heading}\n{section.content.strip()}"

        tokens = count_tokens(content)
        if tokens <= chunk_size:
            chunks.append({
                "content": content,
                "section_path": section_path,
                "section_type": section.section_type,
                "page_start": section.page_start,
                "page_end": section.page_end,
                "boost": SECTION_BOOST.get(section.section_type, 0.8),
            })
        else:
            # Split large sections
            sub_chunks = _split_text_with_context(content, section.heading, chunk_size, overlap)
            for i, sub in enumerate(sub_chunks):
                chunks.append({
                    "content": sub,
                    "section_path": section_path,
                    "section_type": section.section_type,
                    "page_start": section.page_start,
                    "page_end": section.page_end,
                    "boost": SECTION_BOOST.get(section.section_type, 0.8),
                })

    # 3. References as individual chunks (each ref is a chunk)
    ref_chunks = _chunk_references(result.references)
    chunks.extend(ref_chunks)

    return chunks


def _split_text_with_context(text: str, section_heading: str, chunk_size: int, overlap: int) -> list[str]:
    """Split a large section into chunks with overlap and heading prefix."""
    lines = text.split("\n")
    chunks = []
    current_lines = []
    current_tokens = 0

    for line in lines:
        line_tokens = count_tokens(line + "\n")
        if current_tokens + line_tokens > chunk_size and current_lines:
            # Save current chunk
            chunks.append("\n".join(current_lines))
            # Keep overlap lines
            overlap_lines = []
            overlap_tokens = 0
            for ol in reversed(current_lines):
                ot = count_tokens(ol + "\n")
                if overlap_tokens + ot > overlap:
                    break
                overlap_lines.insert(0, ol)
                overlap_tokens += ot
            current_lines = overlap_lines + [line]
            current_tokens = overlap_tokens + line_tokens
        else:
            current_lines.append(line)
            current_tokens += line_tokens

    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks


def _chunk_references(references: list[dict]) -> list[dict]:
    """Chunk references - each reference is a chunk."""
    chunks = []
    for i, ref in enumerate(references, 1):
        ref_parts = []
        if ref.get("authors"):
            ref_parts.append(str(ref["authors"]))
        if ref.get("title"):
            ref_parts.append(f"\"{ref['title']}\"")
        if ref.get("year"):
            ref_parts.append(f"({ref['year']})")
        if ref.get("doi"):
            ref_parts.append(f"DOI: {ref['doi']}")
        if ref.get("pmid"):
            ref_parts.append(f"PMID: {ref['pmid']}")

        content = f"[{i}] {'. '.join(ref_parts)}"
        chunks.append({
            "content": content,
            "section_path": "References",
            "section_type": "references",
            "page_start": None,
            "page_end": None,
            "boost": SECTION_BOOST["references"],
        })

    return chunks


def get_paper_evidence_summary(result: PaperParseResult) -> dict:
    """Extract a structured evidence summary from a parsed paper."""
    summary = {
        "abstract": result.abstract[:500] if result.abstract else "",
        "study_design": None,
        "key_findings": [],
        "conclusion": "",
        "limitations": "",
    }

    for section in result.sections:
        if section.section_type == "methods":
            summary["study_design"] = section.content[:300]
        elif section.section_type == "results":
            # Take first few sentences as key findings
            sentences = section.content.split(".")[:5]
            summary["key_findings"] = [s.strip() + "." for s in sentences if s.strip()]
        elif section.section_type == "conclusion":
            summary["conclusion"] = section.content[:500]
        elif section.section_type == "discussion":
            # Look for limitation mentions
            lower = section.content.lower()
            for kw in ["limitation", "caveat", "caution", "further research"]:
                idx = lower.find(kw)
                if idx >= 0:
                    summary["limitations"] = section.content[idx:idx + 300]
                    break

    return summary
