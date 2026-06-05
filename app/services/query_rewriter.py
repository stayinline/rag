"""Query rewriter for medical terminology expansion and multi-language support."""
import re
import logging
from http import HTTPStatus

from app.config import settings

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


class ConversationalQueryRewriteResult:
    """Result from resolving a conversational query into a standalone retrieval query."""

    def __init__(self, original: str, rewritten: str, used_llm: bool, reason: str = ""):
        self.original = original
        self.rewritten = rewritten
        self.used_llm = used_llm
        self.reason = reason


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


def rewrite_conversational_query(query: str, messages: list[dict] | None = None) -> ConversationalQueryRewriteResult:
    """Resolve references in a follow-up query into a standalone retrieval query."""
    messages = messages or []
    if not messages:
        return ConversationalQueryRewriteResult(original=query, rewritten=query, used_llm=False, reason="no_history")

    if getattr(settings, "llm_query_rewrite", False):
        try:
            rewritten = _rewrite_conversational_query_with_llm(query, messages)
            if rewritten:
                return ConversationalQueryRewriteResult(
                    original=query,
                    rewritten=rewritten,
                    used_llm=True,
                    reason="llm",
                )
        except Exception as exc:
            logger.warning("LLM query rewrite failed; falling back to deterministic rewrite: %s", exc, exc_info=True)

    rewritten = _fallback_conversational_query_rewrite(query, messages)
    return ConversationalQueryRewriteResult(original=query, rewritten=rewritten, used_llm=False, reason="fallback")


def _rewrite_conversational_query_with_llm(query: str, messages: list[dict]) -> str:
    from dashscope import Generation

    history = _format_recent_history(messages)
    prompt = (
        "将用户当前问题改写为适合知识库检索的独立问题。要求：\n"
        "1. 只输出改写后的检索问题，不要解释。\n"
        "2. 保留医学实体、药品名、疾病名、指标名。\n"
        "3. 如果当前问题已经独立，原样输出。\n\n"
        f"对话历史：\n{history}\n\n当前问题：{query}"
    )
    response = Generation.call(
        model=getattr(settings, "llm_query_rewrite_model", "") or getattr(settings, "summary_model", ""),
        messages=[
            {"role": "system", "content": "你是 RAG 检索查询改写器。"},
            {"role": "user", "content": prompt},
        ],
        api_key=getattr(settings, "dashscope_api_key", "") or None,
        result_format="message",
        stream=False,
        request_timeout=getattr(settings, "llm_timeout", 90),
    )
    if getattr(response, "status_code", None) != HTTPStatus.OK:
        message = getattr(response, "message", "") or getattr(response, "code", "") or "unknown error"
        raise RuntimeError(f"query rewrite failed status={getattr(response, 'status_code', None)}: {message}")

    output = getattr(response, "output", None)
    choices = getattr(output, "choices", None) if output is not None else None
    if not choices:
        return ""
    rewritten = choices[0].get("message", {}).get("content", "").strip()
    return _sanitize_rewritten_query(rewritten)


def _fallback_conversational_query_rewrite(query: str, messages: list[dict]) -> str:
    history = _format_recent_history(messages, max_chars=1200)
    if not history:
        return query
    return f"{history}\n当前问题：{query}"


def _format_recent_history(messages: list[dict], max_chars: int = 1800) -> str:
    lines = []
    for message in messages[-6:]:
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            continue
        content = str(message.get("content", "") or "").strip()
        if not content:
            continue
        if len(content) > 500:
            content = f"{content[:500]}..."
        lines.append(f"{role}: {content}")
    text = "\n".join(lines)
    return text[-max_chars:] if len(text) > max_chars else text


def _sanitize_rewritten_query(value: str) -> str:
    rewritten = value.strip().strip('"').strip("'").strip()
    prefixes = ("改写后：", "检索问题：", "查询：", "Rewritten query:", "Query:")
    for prefix in prefixes:
        if rewritten.lower().startswith(prefix.lower()):
            rewritten = rewritten[len(prefix):].strip()
    return rewritten


def rewrite_for_search(query: str) -> str:
    """Get the best single query for search (normalized + most important expansion)."""
    result = rewrite_query(query)
    # Return the longest expanded query (usually the most comprehensive)
    return max(result.expanded, key=len)
