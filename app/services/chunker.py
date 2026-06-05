import logging
import re
import uuid
import tiktoken

from app.config import settings

logger = logging.getLogger(__name__)
enc = tiktoken.get_encoding("cl100k_base")
_HEADING_RE = re.compile(
    r"^(#{1,6}\s+|(\d+|[一二三四五六七八九十]+)[.、\s]+|[A-Z][a-zA-Z\s]+:\s*)"
)
_TEXT_UNIT_RE = re.compile(
    r"\s+|[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]|[^\s\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+"
)


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


def _split_text_units(text: str, chunk_size: int) -> list[str]:
    units = _TEXT_UNIT_RE.findall(text)
    split_units = []
    for unit in units:
        if not unit.isspace() and count_tokens(unit) > chunk_size:
            split_units.extend(unit)
        else:
            split_units.append(unit)
    return split_units


def _split_by_token_window(text: str, chunk_size: int, overlap: int) -> list[str]:
    if count_tokens(text) <= chunk_size:
        chunk = text.strip()
        return [chunk] if chunk else []

    if chunk_size <= 0:
        raise ValueError("rag_chunk_size must be greater than 0")

    overlap = max(0, min(overlap, chunk_size - 1))
    units = _split_text_units(text, chunk_size)
    chunks = []

    start = 0
    while start < len(units):
        end = start
        last_good = start

        while end < len(units):
            candidate = "".join(units[start:end + 1]).strip()
            if not candidate:
                end += 1
                last_good = end
                continue
            if count_tokens(candidate) > chunk_size:
                break
            end += 1
            last_good = end

        if last_good == start:
            last_good = start + 1

        chunk = "".join(units[start:last_good]).strip()
        if chunk:
            chunks.append(chunk)
        if last_good >= len(units):
            break

        overlap_start = last_good
        for idx in range(last_good - 1, start - 1, -1):
            candidate = "".join(units[idx:last_good]).strip()
            if not candidate:
                continue
            if count_tokens(candidate) > overlap:
                break
            overlap_start = idx

        start = overlap_start if overlap_start > start else last_good

    return chunks


def chunk_text(text: str, title: str = "") -> list[dict]:
    """
    Split text into chunks based on headings and token boundaries.
    Each chunk: {"content": str, "section_path": str, "page_start": int | None}
    """
    chunk_size = settings.rag_chunk_size
    overlap = settings.rag_chunk_overlap
    logger.info(
        "Chunk text start title=%s text_length=%s chunk_size=%s overlap=%s",
        title,
        len(text or ""),
        chunk_size,
        overlap,
    )

    lines = text.split("\n")
    sections: list[tuple[str, str]] = []
    current_section_title = title
    current_lines: list[str] = []

    for line in lines:
        if _HEADING_RE.match(line.strip()):
            if current_lines:
                sections.append((current_section_title, "\n".join(current_lines)))
            current_section_title = line.strip().lstrip("#").strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_section_title, "\n".join(current_lines)))
    logger.debug("Chunk text sections identified title=%s section_count=%s", title, len(sections))

    chunks = []
    current_path = title

    for section_title, section_text in sections:
        current_path = f"{title}/{section_title}".strip("/") if title else section_title

        tokens = count_tokens(section_text)
        if tokens <= chunk_size:
            if section_text.strip():
                chunks.append({
                    "content": section_text.strip(),
                    "section_path": current_path,
                })
        else:
            for chunk_text_piece in _split_by_token_window(section_text, chunk_size, overlap):
                chunks.append({
                    "content": chunk_text_piece,
                    "section_path": current_path,
                })

    parent_chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{title}:{len(text or '')}:{count_tokens(text or '')}"))
    for idx, chunk in enumerate(chunks):
        chunk["chunk_index"] = idx
        chunk["parent_chunk_id"] = parent_chunk_id
        chunk["child_chunk_ids"] = []

    logger.info(
        "Chunk text complete title=%s section_count=%s chunk_count=%s",
        title,
        len(sections),
        len(chunks),
    )
    return chunks
