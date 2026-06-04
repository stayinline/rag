"""Tests for chunker service."""

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
