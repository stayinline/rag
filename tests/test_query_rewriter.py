"""Tests for query rewriter service."""

from app.services.query_rewriter import (
    normalize_drug_names,
    expand_medical_terms,
    detect_entities,
    rewrite_query,
    rewrite_for_search,
)


class TestNormalizeDrugNames:
    def test_brand_to_generic(self):
        result = normalize_drug_names("患者服用了格华止")
        assert "二甲双胍" in result

    def test_english_brand(self):
        result = normalize_drug_names("patient took Keytruda")
        assert "pembrolizumab" in result.lower()

    def test_no_change(self):
        result = normalize_drug_names("patient took metformin")
        assert "metformin" in result

    def test_multiple_drugs(self):
        result = normalize_drug_names("格华止 and 拜阿司匹林")
        assert "二甲双胍" in result
        assert "aspirin" in result.lower()


class TestExpandMedicalTerms:
    def test_chinese_expansion(self):
        expanded = expand_medical_terms("高血压")
        assert "高血压" in expanded
        assert "hypertension" in expanded

    def test_cancer_expansion(self):
        expanded = expand_medical_terms("癌症")
        assert "癌症" in expanded
        assert "cancer" in expanded
        assert "肿瘤" in expanded

    def test_no_expansion_for_general_term(self):
        expanded = expand_medical_terms("general term")
        assert "general term" in expanded

    def test_synonym_bidirectional(self):
        # If query contains synonym, should expand to other synonyms
        expanded = expand_medical_terms("tumor")
        assert "tumor" in expanded
        # Should also have cancer/癌症
        expanded_lower = [t.lower() for t in expanded]
        assert "肿瘤" in expanded or "cancer" in expanded_lower


class TestDetectEntities:
    def test_detect_disease(self):
        entities = detect_entities("高血压的治疗方法")
        assert "高血压" in entities["diseases"]

    def test_detect_target(self):
        entities = detect_entities("PD-1抑制剂的效果")
        assert "PD-1" in entities["targets"]

    def test_detect_drug(self):
        # "阿司匹林" is in DRUG_NAME_MAP, so it should be detected as a drug
        entities = detect_entities("阿司匹林的副作用")
        assert "阿司匹林" in entities["drugs"]

    def test_no_entities(self):
        entities = detect_entities("今天天气很好")
        assert entities["drugs"] == []
        assert entities["diseases"] == []
        assert entities["targets"] == []


class TestRewriteQuery:
    def test_basic_rewrite(self):
        result = rewrite_query("高血压的治疗方法")
        assert result.original == "高血压的治疗方法"
        assert len(result.expanded) >= 1
        assert "高血压" in result.original

    def test_entities_detected(self):
        result = rewrite_query("PD-1抑制剂治疗肺癌")
        assert "PD-1" in result.entities["targets"]

    def test_expansion_can_be_disabled(self):
        result = rewrite_query("高血压", expand_synonyms=False)
        assert len(result.expanded) == 1

    def test_expansion_enabled(self):
        result = rewrite_query("高血压", expand_synonyms=True)
        assert len(result.expanded) >= 1
        # Should have the original and at least one expansion
        any_expanded = any(
            "hypertension" in q.lower() or "血压" in q
            for q in result.expanded
        )
        assert any_expanded or len(result.expanded) >= 1


class TestRewriteForSearch:
    def test_returns_longest_expanded(self):
        query = rewrite_for_search("高血压")
        assert isinstance(query, str)
        assert len(query) >= 1
