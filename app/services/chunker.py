import logging
import re
import tiktoken

from app.config import settings

logger = logging.getLogger(__name__)
enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


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

    heading_re = re.compile(r"^(#{1,6}\s+|[\d一二三四五六七八九十]+[.、]\s*|[A-Z][a-zA-Z\s]+:\s*)")

    for line in lines:
        if heading_re.match(line.strip()):
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
            words = section_text.split()
            start = 0
            while start < len(words):
                end = start
                chunk_tokens = 0
                while end < len(words) and chunk_tokens < chunk_size:
                    chunk_tokens += len(enc.encode(words[end] + " "))
                    end += 1

                chunk_text_piece = " ".join(words[start:end]).strip()
                if chunk_text_piece:
                    chunks.append({
                        "content": chunk_text_piece,
                        "section_path": current_path,
                    })

                start = end - (overlap // 5) if end < len(words) else end

    logger.info(
        "Chunk text complete title=%s section_count=%s chunk_count=%s",
        title,
        len(sections),
        len(chunks),
    )
    return chunks
