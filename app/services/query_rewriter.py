"""Query rewriter for medical terminology expansion and multi-language support."""
import re
import logging

logger = logging.getLogger(__name__)


# Medical term synonym dictionary (simplified - for production, use UMLS/MedDRA)
MEDICAL_SYNONYMS = {
    "高血压": ["hypertension", "high blood pressure", "动脉压升高"],
    "糖尿病": ["diabetes", "diabetes mellitus", "血糖异常"],
    "癌症": ["cancer", "肿瘤", "neoplasm", "malignancy", "carcinoma"],
    "肿瘤": ["tumor", "癌症", "neoplasm", "mass"],
    "心脏病": ["heart disease", "cardiovascular disease", "心血管疾病"],
    "肺炎": ["pneumonia", "肺部感染", "lung infection"],
    "肝癌": ["liver cancer", "hepatocellular carcinoma", "HCC"],
    "肺癌": ["lung cancer", "pulmonary carcinoma", "NSCLC", "SCLC"],
    "乳腺癌": ["breast cancer", "mammary carcinoma"],
    "白血病": ["leukemia", "血癌", "blood cancer"],
    "抑郁": ["depression", " depressive disorder", "抑郁症"],
    "阿司匹林": ["aspirin", "乙酰水杨酸", "acetylsalicylic acid"],
    "二甲双胍": ["metformin", "格华止", "glucophage"],
    "PD-1": ["programmed cell death 1", "程序性死亡受体1"],
    "PD-L1": ["programmed death ligand 1", "程序性死亡配体1"],
    "EGFR": ["表皮生长因子受体", "epidermal growth factor receptor"],
    "VEGF": ["血管内皮生长因子", "vascular endothelial growth factor"],
}

# Drug name normalization (brand -> generic)
DRUG_NAME_MAP = {
    "格华止": "二甲双胍",
    "glucophage": "metformin",
    "泰瑞沙": "奥希替尼",
    "tagrisso": "osimertinib",
    "可瑞达": "帕博利珠单抗",
    "keytruda": "pembrolizumab",
    "欧狄沃": "纳武利尤单抗",
    "opdivo": "nivolumab",
    "赫赛汀": "曲妥珠单抗",
    "herceptin": "trastuzumab",
    "阿司匹林": "aspirin",
    "拜阿司匹林": "aspirin",
}


class QueryRewriteResult:
    """Result from query rewriting."""
    def __init__(self, original: str, expanded: list[str], entities: dict):
        self.original = original
        self.expanded = expanded  # List of expanded query variants
        self.entities = entities  # Detected entities (drugs, diseases, targets)


def normalize_drug_names(text: str) -> str:
    """Normalize drug brand names to generic names."""
    result = text
    replacements = 0
    for brand, generic in DRUG_NAME_MAP.items():
        if brand.lower() in result.lower():
            result = re.sub(re.escape(brand), generic, result, flags=re.IGNORECASE)
            replacements += 1
    logger.debug("Normalize drug names complete input_length=%s replacements=%s", len(text or ""), replacements)
    return result


def expand_medical_terms(query: str) -> list[str]:
    """Expand a query with medical term synonyms."""
    expanded = {query}

    query_lower = query.lower()
    for term, synonyms in MEDICAL_SYNONYMS.items():
        if term.lower() in query_lower:
            for synonym in synonyms:
                expanded.add(query_lower.replace(term.lower(), synonym))
        # Also check if synonym appears in query and expand to other synonyms
        for synonym in synonyms:
            if synonym.lower() in query_lower:
                expanded.add(query_lower.replace(synonym.lower(), term.lower()))
                for other_syn in synonyms:
                    if other_syn != synonym:
                        expanded.add(query_lower.replace(synonym.lower(), other_syn.lower()))

    expanded_list = list(expanded)
    logger.debug("Expand medical terms complete query_length=%s expanded_count=%s", len(query or ""), len(expanded_list))
    return expanded_list


def detect_entities(query: str) -> dict:
    """Detect medical entities in query."""
    query_lower = query.lower()
    entities = {"drugs": [], "diseases": [], "targets": []}

    for term in MEDICAL_SYNONYMS:
        if term.lower() in query_lower:
            # Check if it's a drug
            if term in DRUG_NAME_MAP or term.lower() in DRUG_NAME_MAP:
                entities["drugs"].append(term)
            # Check if it's a target
            if term in ("PD-1", "PD-L1", "EGFR", "VEGF"):
                entities["targets"].append(term)
            else:
                entities["diseases"].append(term)

    for drug, generic in DRUG_NAME_MAP.items():
        if drug.lower() in query_lower:
            entities["drugs"].append(f"{drug}->{generic}")

    logger.debug("Detect query entities complete query_length=%s entity_counts=%s", len(query or ""), {k: len(v) for k, v in entities.items()})
    return entities


def rewrite_query(query: str, expand_synonyms: bool = True) -> QueryRewriteResult:
    """Rewrite a query for better retrieval.

    Steps:
    1. Detect medical entities
    2. Normalize drug names
    3. Expand with medical term synonyms (for multi-language support)
    """
    logger.info("Rewrite query start query_length=%s expand_synonyms=%s", len(query or ""), expand_synonyms)
    # Step 1: Detect entities
    entities = detect_entities(query)

    # Step 2: Normalize drug names
    normalized = normalize_drug_names(query)

    # Step 3: Expand queries
    expanded = [normalized]
    if expand_synonyms:
        synonyms = expand_medical_terms(normalized)
        expanded = [normalized] + synonyms

    result = QueryRewriteResult(
        original=query,
        expanded=expanded,
        entities=entities,
    )
    logger.info(
        "Rewrite query complete query_length=%s expanded_count=%s entity_counts=%s",
        len(query or ""),
        len(result.expanded),
        {k: len(v) for k, v in result.entities.items()},
    )
    return result


def rewrite_for_search(query: str) -> str:
    """Get the best single query for search (normalized + most important expansion)."""
    result = rewrite_query(query)
    # Return the longest expanded query (usually the most comprehensive)
    return max(result.expanded, key=len)
