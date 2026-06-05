"""Tests for chunker service."""
from unittest.mock import patch

from app.services.chunker import chunk_text, count_tokens


def test_count_tokens():
    text = "Hello world"
    tokens = count_tokens(text)
    assert tokens > 0
    assert isinstance(tokens, int)


def test_count_tokens_empty():
    tokens = count_tokens("")
    assert tokens == 0


def test_chunk_simple_text():
    text = "This is a simple test paragraph."
    chunks = chunk_text(text, title="Test Doc")
    assert len(chunks) >= 1
    assert "content" in chunks[0]
    assert "section_path" in chunks[0]


def test_chunk_with_headings():
    text = """# Introduction
This is the introduction section with some content.

# Methods
Here we describe the methods used in this study.

# Results
The results show significant findings.
"""
    chunks = chunk_text(text, title="Research Paper")
    assert len(chunks) >= 3
    # Check section paths contain headings
    paths = [c["section_path"] for c in chunks]
    assert any("Introduction" in p for p in paths)
    assert any("Methods" in p for p in paths)
    assert any("Results" in p for p in paths)


def test_chunk_preserves_content():
    text = """# Summary
The quick brown fox jumps over the lazy dog.
"""
    chunks = chunk_text(text, title="Test")
    all_content = " ".join(c["content"] for c in chunks)
    assert "quick brown fox" in all_content


def test_chunk_empty_text():
    chunks = chunk_text("", title="Empty")
    assert len(chunks) == 0


def test_chunk_with_markdown_headings():
    text = """## Background
Background information here.

## Analysis
Analysis details follow.

### Sub-section
More details.
"""
    chunks = chunk_text(text, title="Report")
    assert len(chunks) >= 3


def test_chunk_numbered_headings():
    text = """一、概述
这是第一部分的内容。

二、方法
这是第二部分的内容。
"""
    chunks = chunk_text(text, title="文档")
    assert len(chunks) >= 2


def test_chunk_chinese_text_without_spaces_uses_token_boundaries():
    text = "这是一个没有空格的中文段落，用于验证分块器不会依赖英文空格切词。" * 30

    with patch("app.services.chunker.settings") as mock_settings:
        mock_settings.rag_chunk_size = 80
        mock_settings.rag_chunk_overlap = 20
        chunks = chunk_text(text, title="中文文档")

    assert len(chunks) > 1
    assert all(count_tokens(chunk["content"]) <= 80 for chunk in chunks)
    assert all(" " not in chunk["content"] for chunk in chunks)


def test_chunk_overlap_is_measured_with_tokens():
    text = " ".join(f"term{i}" for i in range(60))

    with patch("app.services.chunker.settings") as mock_settings:
        mock_settings.rag_chunk_size = 20
        mock_settings.rag_chunk_overlap = 6
        chunks = chunk_text(text, title="Overlap")

    assert len(chunks) > 1
    assert count_tokens(chunks[0]["content"]) <= 20
    assert count_tokens(chunks[1]["content"]) <= 20
    first_words = chunks[0]["content"].split()
    second_words = chunks[1]["content"].split()
    common_overlap = []
    for size in range(min(len(first_words), len(second_words)), 0, -1):
        if first_words[-size:] == second_words[:size]:
            common_overlap = first_words[-size:]
            break
    assert common_overlap
    assert count_tokens(" ".join(common_overlap)) <= 6
