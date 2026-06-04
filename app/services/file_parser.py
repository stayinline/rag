import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def compute_file_hash(path: str) -> str:
    logger.debug("Compute file hash start path=%s", path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    digest = h.hexdigest()
    logger.debug("Compute file hash complete path=%s sha256=%s", path, digest)
    return digest


def parse_pdf(path: str) -> str:
    import fitz

    logger.info("Parse PDF start path=%s", path)
    doc = fitz.open(path)
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text("text"))
    page_count = doc.page_count
    doc.close()
    text = "\n".join(text_parts)
    logger.info("Parse PDF complete path=%s pages=%s text_length=%s", path, page_count, len(text))
    return text


def parse_docx(path: str) -> str:
    from docx import Document

    logger.info("Parse DOCX start path=%s", path)
    doc = Document(path)
    text_parts = []
    for para in doc.paragraphs:
        if para.text.strip():
            text_parts.append(para.text)
    text = "\n".join(text_parts)
    logger.info("Parse DOCX complete path=%s paragraph_count=%s text_length=%s", path, len(text_parts), len(text))
    return text


def parse_txt(path: str) -> str:
    logger.info("Parse text file start path=%s", path)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    logger.info("Parse text file complete path=%s text_length=%s", path, len(text))
    return text


def parse_markdown(path: str) -> str:
    return parse_txt(path)


def parse_file(path: str) -> str:
    ext = Path(path).suffix.lower()
    parsers = {
        ".pdf": parse_pdf,
        ".docx": parse_docx,
        ".doc": parse_docx,
        ".txt": parse_txt,
        ".md": parse_markdown,
        ".markdown": parse_markdown,
    }
    parser = parsers.get(ext)
    if not parser:
        logger.error("Parse file failed path=%s extension=%s reason=unsupported_type", path, ext)
        raise ValueError(f"Unsupported file type: {ext}")
    logger.info("Parse file dispatch path=%s extension=%s parser=%s", path, ext, parser.__name__)
    return parser(path)
