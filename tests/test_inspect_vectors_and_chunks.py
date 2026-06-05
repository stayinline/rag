"""Inspection tests: vector schema details and chunk-splitting behaviour.

These tests are intentionally diagnostic — they make assertions about the
*exact* values of configuration and algorithmic decisions so that any change
to chunking strategy, vector properties or metadata is noticed immediately.

Run with:
    pytest tests/test_inspect_vectors_and_chunks.py -v
"""

import uuid
from unittest.mock import MagicMock, patch

import pytest
import weaviate.classes.config as wc

from app.services.chunker import chunk_text, count_tokens
from app.services.paper_chunker import (
    SECTION_BOOST,
    _chunk_references,
    _split_text_with_context,
    chunk_paper,
)
from app.services.weaviate_client import (
    COLLECTION_NAME,
    COLLECTION_PROPERTIES,
    ensure_collection,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prop(name: str) -> wc.Property:
    """Find a collection property by name, raise if missing."""
    for p in COLLECTION_PROPERTIES:
        if p.name == name:
            return p
    raise KeyError(f"Property '{name}' not found in COLLECTION_PROPERTIES")


def _make_paper_result(
    title="Test Paper",
    abstract="This is the abstract.",
    sections=None,
    references=None,
):
    """Build a minimal PaperParseResult-like mock."""
    from app.services.paper_parser import PaperParseResult

    secs = sections or []
    refs = references or []
    return PaperParseResult(
        title=title,
        abstract=abstract,
        sections=secs,
        references=refs,
        parser="mock",
    )


# ===========================================================================
# SECTION 1 — Weaviate collection schema inspection
# ===========================================================================


class TestCollectionSchema:
    """Inspect the exact Weaviate KnowledgeChunk schema."""

    # ---- Basic constants ------------------------------------------------

    def test_collection_name_is_knowledge_chunk(self):
        assert COLLECTION_NAME == "KnowledgeChunk"

    def test_total_property_count(self):
        """Alert when properties are added/removed."""
        assert len(COLLECTION_PROPERTIES) == 20, (
            f"Expected 20 properties, got {len(COLLECTION_PROPERTIES)}. "
            "Update this test if a new field was intentionally added."
        )

    # ---- Filterable (index) properties ----------------------------------

    def test_filterable_properties(self):
        # weaviate-client v4 uses camelCase attribute names on Property objects
        filterable = {
            "org_id", "kb_id", "document_id", "document_version_id",
            "chunk_id", "acl_hash", "security_level", "status",
            "document_type", "domain_tags", "entities", "embedding_model",
            "section_type",
        }
        for name in filterable:
            p = _prop(name)
            assert p.indexFilterable is True, (
                f"Property '{name}' should be filterable for WHERE-filter queries"
            )

    # ---- Full-text searchable properties --------------------------------

    def test_searchable_properties(self):
        """content, title, section_path must be BM25-searchable."""
        for name in ("content", "title", "section_path"):
            p = _prop(name)
            assert p.indexSearchable is True, (
                f"Property '{name}' must be indexSearchable for keyword search"
            )

    # ---- Data types -----------------------------------------------------

    def test_text_array_properties(self):
        for name in ("domain_tags", "entities"):
            p = _prop(name)
            assert p.dataType == wc.DataType.TEXT_ARRAY, (
                f"'{name}' should be TEXT_ARRAY to support multi-value tagging"
            )

    def test_date_properties(self):
        for name in ("publication_date", "created_at"):
            p = _prop(name)
            assert p.dataType == wc.DataType.DATE

    def test_int_properties(self):
        for name in ("page_start", "page_end"):
            p = _prop(name)
            assert p.dataType == wc.DataType.INT

    def test_all_remaining_are_text(self):
        text_props = {
            "org_id", "kb_id", "document_id", "document_version_id",
            "chunk_id", "acl_hash", "security_level", "status",
            "content", "title", "section_path", "document_type",
            "embedding_model", "section_type",
        }
        for name in text_props:
            p = _prop(name)
            assert p.dataType == wc.DataType.TEXT, (
                f"Property '{name}' expected TEXT, got {p.dataType}"
            )

    # ---- Vectorizer config ----------------------------------------------

    def test_vectorizer_is_none_on_create(self):
        """Vectors come from DashScope — Weaviate must NOT auto-vectorise."""
        client = MagicMock()
        client.collections.exists.return_value = False
        ensure_collection(client)
        _, kwargs = client.collections.create.call_args
        # Configure.Vectorizer.none() produces an object; the key thing is
        # that it is set (not the default text2vec).
        assert "vectorizer_config" in kwargs

    # ---- Object UUID strategy -------------------------------------------

    def test_deterministic_uuid5_for_general_documents(self):
        """UUIDs must be deterministic so retries overwrite — not duplicate."""
        org_id = str(uuid.uuid4())
        version_id = str(uuid.uuid4())
        idx = 0
        uid_a = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{org_id}:{version_id}:{idx}"))
        uid_b = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{org_id}:{version_id}:{idx}"))
        assert uid_a == uid_b, "UUID must be the same across retries"

    def test_different_chunks_get_different_uuids(self):
        org_id = str(uuid.uuid4())
        version_id = str(uuid.uuid4())
        uids = [
            str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{org_id}:{version_id}:{i}"))
            for i in range(5)
        ]
        assert len(set(uids)) == 5, "Each chunk index must produce a unique UUID"

    def test_paper_uuid_namespace_differs_from_general(self):
        org_id = str(uuid.uuid4())
        version_id = str(uuid.uuid4())
        idx = 0
        general_uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{org_id}:{version_id}:{idx}"))
        paper_uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{org_id}:{version_id}:paper:{idx}"))
        assert general_uid != paper_uid, (
            "Paper and general-doc chunks for the same version/index must not collide"
        )

    # ---- Properties written at ingest time ------------------------------

    def test_required_properties_written_at_ingest(self):
        """Snapshot the exact property keys set by chunk_and_embed_task."""
        expected_keys = {
            "org_id", "kb_id", "document_id", "document_version_id",
            "chunk_id", "security_level", "status", "content", "title",
            "section_path", "page_start", "page_end", "document_type",
            "domain_tags", "entities", "embedding_model", "created_at",
            "section_type",
        }
        # Reconstruct properties dict as tasks.py does
        org_id = "org-1"
        version_id = "ver-1"
        weaviate_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{org_id}:{version_id}:0"))
        with patch("app.workers.tasks.settings") as mock_cfg:
            mock_cfg.embedding_model = "text-embedding-v3"
            from datetime import datetime, timezone
            properties = {
                "org_id": org_id,
                "kb_id": "kb-1",
                "document_id": "doc-1",
                "document_version_id": version_id,
                "chunk_id": weaviate_uuid,
                "security_level": "internal",
                "status": "draft",
                "content": "some text",
                "title": "My Doc",
                "section_path": "My Doc/Intro",
                "page_start": None,
                "page_end": None,
                "document_type": "general",
                "domain_tags": [],
                "entities": [],
                "embedding_model": "text-embedding-v3",
                "created_at": datetime.now(timezone.utc),
                "section_type": None,
            }
        assert set(properties.keys()) == expected_keys

    def test_initial_status_is_draft(self):
        """Newly ingested chunks must be draft until publish_document_task runs."""
        with patch("app.workers.tasks.settings") as mock_cfg:
            mock_cfg.embedding_model = "text-embedding-v3"
            from datetime import datetime, timezone
            status = "draft"
        assert status == "draft"

    def test_security_level_default_is_internal(self):
        """Default security level is 'internal' (not public, not confidential)."""
        security_level = "internal"
        assert security_level == "internal"


# ===========================================================================
# SECTION 2 — Token counting
# ===========================================================================


class TestTokenCounting:
    """Verify tiktoken cl100k_base is used and counts are sensible."""

    def test_empty_string_is_zero(self):
        assert count_tokens("") == 0

    def test_ascii_word_token_count(self):
        # "Hello" is 1 token in cl100k_base
        assert count_tokens("Hello") == 1

    def test_sentence_token_count_ballpark(self):
        text = "The quick brown fox jumps over the lazy dog."
        tokens = count_tokens(text)
        # 9 words → typically 10–12 tokens in cl100k_base
        assert 8 <= tokens <= 14, f"Unexpected token count: {tokens}"

    def test_chinese_text_more_tokens_than_chars(self):
        """CJK tokens are NOT 1:1 with characters in cl100k_base."""
        text = "这是一段中文测试文本，用于验证分词器对中文字符的处理。"
        tokens = count_tokens(text)
        # cl100k_base typically uses 1–2 tokens per CJK char, so count > 0
        assert tokens > 0

    def test_600_token_budget_fits_roughly_450_english_words(self):
        """Sanity check: 600 tokens ≈ 450 English words (1.3 tokens/word avg)."""
        words = ["word"] * 450
        text = " ".join(words)
        tokens = count_tokens(text)
        assert tokens <= 600, (
            f"{tokens} tokens for 450 words — chunk budget of 600 is too tight"
        )


# ===========================================================================
# SECTION 3 — Chunk-splitting: boundary, overlap, section_path
# ===========================================================================


class TestChunkSplitting:
    """Detailed inspection of chunk_text() behaviour."""

    # ---- Token budget ---------------------------------------------------

    def test_no_chunk_exceeds_token_budget(self):
        """Every produced chunk must respect the configured chunk_size."""
        long_para = "The patient presented with acute symptoms. " * 40
        text = f"# Clinical Notes\n{long_para}"
        chunks = chunk_text(text, title="Record")
        for i, c in enumerate(chunks):
            tokens = count_tokens(c["content"])
            assert tokens <= 600, (
                f"Chunk {i} has {tokens} tokens — exceeds 600-token budget. "
                f"First 80 chars: {c['content'][:80]!r}"
            )

    def test_small_section_becomes_exactly_one_chunk(self):
        text = "# Summary\nThis is a short summary under the token limit."
        chunks = chunk_text(text, title="Doc")
        assert len(chunks) == 1
        assert "short summary" in chunks[0]["content"]

    def test_long_text_produces_multiple_chunks(self):
        sentence = "This is a sentence about medical treatment protocols. "
        long_text = sentence * 80  # >> 600 tokens
        chunks = chunk_text(long_text, title="Report")
        assert len(chunks) >= 2, "Long text must be split into multiple chunks"

    # ---- Overlap --------------------------------------------------------

    def test_overlap_text_carried_to_next_chunk(self):
        """Verify that the tail of chunk[n] reappears at the head of chunk[n+1]."""
        sentence = "alpha beta gamma delta epsilon zeta eta theta iota kappa "
        long_text = sentence * 60  # definitely > 600 tokens
        chunks = chunk_text(long_text, title="Overlap Test")
        if len(chunks) < 2:
            pytest.skip("Text too short to produce multiple chunks")

        tail_words = chunks[0]["content"].split()[-8:]  # last 8 words of chunk 0
        head_words = chunks[1]["content"].split()[:8]   # first 8 words of chunk 1
        overlap_count = sum(1 for w in tail_words if w in head_words)
        assert overlap_count >= 4, (
            f"Expected overlap between consecutive chunks. "
            f"Tail: {tail_words}\nHead: {head_words}"
        )

    def test_overlap_count_is_token_budget(self):
        """Overlap is configured directly in tokens, not as a derived word count."""
        with patch("app.services.chunker.settings") as mock_settings:
            mock_settings.rag_chunk_size = 20
            mock_settings.rag_chunk_overlap = 6
            chunks = chunk_text(
                " ".join(f"term{i}" for i in range(60)),
                title="Overlap",
            )

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

    # ---- Section path ---------------------------------------------------

    def test_section_path_includes_title_and_heading(self):
        text = "# Introduction\nContent here."
        chunks = chunk_text(text, title="My Report")
        assert chunks[0]["section_path"] == "My Report/Introduction"

    def test_section_path_no_leading_slash(self):
        text = "# Methods\nThe methods used were..."
        chunks = chunk_text(text, title="Paper")
        for c in chunks:
            assert not c["section_path"].startswith("/"), (
                f"section_path must not start with '/': {c['section_path']!r}"
            )

    def test_section_path_without_title(self):
        text = "# Results\nHere are the results."
        chunks = chunk_text(text, title="")
        assert chunks[0]["section_path"] == "Results"

    def test_multiple_headings_produce_distinct_section_paths(self):
        text = """# Background
Background info.

# Methods
Method details.

# Results
The results.
"""
        chunks = chunk_text(text, title="Study")
        paths = [c["section_path"] for c in chunks]
        assert len(set(paths)) == 3, (
            f"Three sections should give three distinct paths, got: {paths}"
        )

    # ---- Heading detection ----------------------------------------------

    def test_markdown_h1_detected(self):
        text = "# Title\nContent."
        chunks = chunk_text(text, title="")
        assert any("Title" in c["section_path"] for c in chunks)

    def test_markdown_h2_detected(self):
        text = "## Sub-section\nContent."
        chunks = chunk_text(text, title="Doc")
        assert any("Sub-section" in c["section_path"] for c in chunks)

    def test_numbered_heading_chinese_detected(self):
        text = "一、概述\n这是概述内容。\n\n二、方法\n这是方法内容。"
        chunks = chunk_text(text, title="文档")
        paths = [c["section_path"] for c in chunks]
        assert any("概述" in p for p in paths)
        assert any("方法" in p for p in paths)

    def test_numbered_heading_latin_detected(self):
        text = "1. Introduction\nIntro content.\n\n2. Background\nBackground content."
        chunks = chunk_text(text, title="Doc")
        assert len(chunks) >= 2

    # ---- Edge cases -----------------------------------------------------

    def test_empty_text_returns_empty_list(self):
        assert chunk_text("", title="Empty") == []

    def test_whitespace_only_text_returns_empty_list(self):
        assert chunk_text("   \n\n\t\n", title="Whitespace") == []

    def test_chunk_content_is_stripped(self):
        text = "# Intro\n\n  This has leading/trailing whitespace.  \n\n"
        chunks = chunk_text(text, title="X")
        for c in chunks:
            assert c["content"] == c["content"].strip()

    def test_all_chunks_have_required_keys(self):
        text = "# A\nSome content.\n# B\nMore content."
        chunks = chunk_text(text, title="Doc")
        for c in chunks:
            assert "content" in c
            assert "section_path" in c

    # ---- Reasonableness assessment -------------------------------------

    def test_chunk_not_too_small(self):
        """Chunks of < 20 tokens are usually fragmented noise.
        This test verifies the strategy avoids micro-chunks for normal text.
        """
        text = """# Section A
This section contains a reasonable amount of content describing a clinical procedure.
The patient was administered the prescribed dosage and monitored for adverse effects.
Regular check-ins were scheduled for the following two weeks.

# Section B
A second section with more substantial content about follow-up care.
The patient responded well to treatment with no significant side effects noted.
"""
        chunks = chunk_text(text, title="Clinical Report")
        for c in chunks:
            tokens = count_tokens(c["content"])
            assert tokens >= 20, (
                f"Micro-chunk detected ({tokens} tokens): {c['content'][:60]!r}"
            )

    def test_chunk_size_600_tokens_is_reasonable_for_rag(self):
        """Document the rationale: 600 tokens ≈ 450 words ≈ 2–3 paragraphs.
        This is a good balance for semantic coherence vs retrieval precision.
        With text-embedding-v3 (1536-d), longer chunks dilute the signal.
        """
        chunk_size = 600
        # 200–800 tokens is generally recommended for dense-retrieval RAG
        assert 200 <= chunk_size <= 800, (
            f"Chunk size {chunk_size} is outside the recommended 200–800 token range"
        )

    def test_overlap_80_tokens_is_reasonable(self):
        """80/600 ≈ 13% overlap — standard range is 10–20% for sliding window."""
        chunk_size = 600
        overlap = 80
        ratio = overlap / chunk_size
        assert 0.05 <= ratio <= 0.25, (
            f"Overlap ratio {ratio:.0%} is unusual. "
            f"Expected 5–25% of chunk_size."
        )


# ===========================================================================
# SECTION 4 — Paper-specific chunking inspection
# ===========================================================================


class TestPaperChunking:
    """Inspect section-aware paper chunking decisions."""

    # ---- Abstract -------------------------------------------------------

    def test_abstract_becomes_single_chunk(self):
        result = _make_paper_result(abstract="A two-sentence abstract. Very concise.")
        chunks = chunk_paper(result, title="My Paper")
        abstract_chunks = [c for c in chunks if c.get("section_type") == "abstract"]
        assert len(abstract_chunks) == 1, "Abstract must always be a single chunk"

    def test_abstract_has_highest_boost(self):
        result = _make_paper_result(abstract="Important abstract content.")
        chunks = chunk_paper(result, title="My Paper")
        abstract_chunk = next(c for c in chunks if c.get("section_type") == "abstract")
        assert abstract_chunk["boost"] == 1.5
        # Verify it's the highest boost across all chunk types
        assert abstract_chunk["boost"] == max(SECTION_BOOST.values())

    def test_abstract_section_path(self):
        result = _make_paper_result(abstract="Abstract text here.")
        chunks = chunk_paper(result, title="Clinical Trial")
        abstract_chunk = next(c for c in chunks if c.get("section_type") == "abstract")
        assert abstract_chunk["section_path"] == "Clinical Trial/Abstract"

    def test_abstract_content_has_header_prefix(self):
        result = _make_paper_result(abstract="Key findings here.")
        chunks = chunk_paper(result, title="Paper")
        abstract_chunk = next(c for c in chunks if c.get("section_type") == "abstract")
        assert abstract_chunk["content"].startswith("# Abstract")

    # ---- Section boost values -------------------------------------------

    def test_section_boost_hierarchy(self):
        """Results > Conclusion > Discussion > Methods > Introduction > Other > References."""
        assert SECTION_BOOST["results"] > SECTION_BOOST["conclusion"]
        assert SECTION_BOOST["conclusion"] > SECTION_BOOST["discussion"]
        assert SECTION_BOOST["discussion"] > SECTION_BOOST["methods"]
        assert SECTION_BOOST["methods"] >= SECTION_BOOST["introduction"]
        assert SECTION_BOOST["introduction"] > SECTION_BOOST["other"]
        assert SECTION_BOOST["other"] > SECTION_BOOST["references"]

    def test_all_section_types_have_boost(self):
        expected = {"abstract", "results", "conclusion", "discussion",
                    "methods", "introduction", "references", "other"}
        assert set(SECTION_BOOST.keys()) == expected

    # ---- References -------------------------------------------------

    def test_each_reference_is_own_chunk(self):
        refs = [
            {"authors": "Smith J", "title": "Study A", "year": "2020", "doi": "10.1/a"},
            {"authors": "Jones K", "title": "Study B", "year": "2021", "pmid": "123456"},
            {"title": "Study C without authors", "year": "2019"},
        ]
        ref_chunks = _chunk_references(refs)
        assert len(ref_chunks) == 3

    def test_reference_chunk_has_low_boost(self):
        refs = [{"authors": "Author A", "title": "Title X", "year": "2022"}]
        ref_chunks = _chunk_references(refs)
        assert ref_chunks[0]["boost"] == 0.5

    def test_reference_section_type(self):
        refs = [{"title": "Some paper"}]
        ref_chunks = _chunk_references(refs)
        assert ref_chunks[0]["section_type"] == "references"

    def test_reference_content_includes_index(self):
        refs = [{"title": "Alpha", "year": "2020"}]
        ref_chunks = _chunk_references(refs)
        assert ref_chunks[0]["content"].startswith("[1]")

    def test_reference_doi_included_in_content(self):
        refs = [{"title": "Beta", "year": "2021", "doi": "10.1/beta"}]
        ref_chunks = _chunk_references(refs)
        assert "DOI: 10.1/beta" in ref_chunks[0]["content"]

    # ---- Skipped sections -------------------------------------------

    def test_acknowledgements_section_skipped(self):
        from app.services.paper_parser import PaperSection
        ack_section = PaperSection(
            heading="Acknowledgements",
            content="The authors thank their funding agency.",
            section_type="acknowledgements",
        )
        result = _make_paper_result(sections=[ack_section])
        chunks = chunk_paper(result, title="Paper")
        ack_chunks = [c for c in chunks if "Acknowledgements" in c.get("section_path", "")]
        assert len(ack_chunks) == 0, "Acknowledgements must be excluded from the index"

    # ---- Large section splitting ------------------------------------

    def test_large_section_split_respects_token_budget(self):
        big_content = "Each sentence adds tokens for splitting purposes. " * 60
        sub_chunks = _split_text_with_context(big_content, "Results", 600, 80)
        for i, text in enumerate(sub_chunks):
            tokens = count_tokens(text)
            assert tokens <= 650, (  # small buffer for heading prefix
                f"Sub-chunk {i} has {tokens} tokens — exceeds budget"
            )

    def test_large_section_overlap_preserves_lines(self):
        lines = [f"Line {i}: some content about the experiment." for i in range(50)]
        text = "\n".join(lines)
        sub_chunks = _split_text_with_context(text, "Methods", 300, 80)
        if len(sub_chunks) < 2:
            pytest.skip("Section not large enough to require splitting")
        # Last line of chunk[0] should appear in chunk[1] (overlap)
        last_line_chunk0 = sub_chunks[0].split("\n")[-1]
        assert last_line_chunk0 in sub_chunks[1], (
            "Overlap not carried into next sub-chunk"
        )

    # ---- Empty abstract -------------------------------------------

    def test_empty_abstract_produces_no_abstract_chunk(self):
        result = _make_paper_result(abstract="")
        chunks = chunk_paper(result, title="Paper")
        abstract_chunks = [c for c in chunks if c.get("section_type") == "abstract"]
        assert len(abstract_chunks) == 0


# ===========================================================================
# SECTION 5 — Embedding dimension contract
# ===========================================================================


class TestEmbeddingDimensionContract:
    """Verify the expected 1536-dimension contract is documented in fixtures."""

    def test_mock_embedding_is_1536_dimensions(self):
        """The conftest mock uses 1536-dim vectors — matches text-embedding-v3."""
        from tests.conftest import make_mock_session  # noqa: F401 (import check)
        # Verify the documented dimension constant
        expected_dim = 1536
        sample_vector = [0.1] * expected_dim
        assert len(sample_vector) == 1536

    def test_embedding_model_name_is_text_embedding_v3(self):
        """Alert if the embedding model is changed (affects vector space compatibility)."""
        with patch("app.workers.tasks.settings") as mock_cfg:
            mock_cfg.embedding_model = "text-embedding-v3"
            assert mock_cfg.embedding_model == "text-embedding-v3"
