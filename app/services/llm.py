import logging
import time

import dashscope
from dashscope import Generation

from app.config import settings

logger = logging.getLogger(__name__)
dashscope.api_key = settings.dashscope_api_key

SYSTEM_PROMPT = """你是生命健康领域企业知识助手。请基于提供的参考资料回答问题。

要求：
1. 只基于提供的资料回答，不要编造信息。
2. 每个事实性结论必须标注来源编号，格式为 [1]、[2] 等。
3. 如果资料不足以回答问题，请明确说明。
4. 涉及诊疗、用药、患者相关问题时，必须提醒：本系统不替代专业医疗建议。
5. 回答使用中文。
"""


def generate_stream(
    query: str,
    context: str,
    messages: list[dict] | None = None,
):
    """Stream generation with context. Returns a Response object for streaming."""
    t0 = time.monotonic()
    system_msg = SYSTEM_PROMPT + "\n\n参考资料：\n" + context

    msg_list = messages or []
    full_messages = [
        {"role": "system", "content": system_msg},
        *msg_list,
        {"role": "user", "content": query},
    ]

    logger.info(
        "LLM generation request start model=%s query_length=%s context_length=%s message_count=%s stream=%s",
        settings.llm_model,
        len(query or ""),
        len(context or ""),
        len(full_messages),
        True,
    )
    response = Generation.call(
        model=settings.llm_model,
        messages=full_messages,
        api_key=settings.dashscope_api_key or None,
        result_format="message",
        stream=True,
        incremental_output=True,
        request_timeout=settings.llm_timeout,
    )
    logger.info(
        "LLM generation request created model=%s duration_ms=%.2f",
        settings.llm_model,
        (time.monotonic() - t0) * 1000,
    )
    return response
