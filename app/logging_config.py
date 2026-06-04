import logging
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.config import settings

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get()
        return True


def setup_logging() -> None:
    """Configure a single console logging format for the app and workers."""
    level_name = getattr(settings, "log_level", "DEBUG" if settings.debug else "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(request_id)s] %(name)s:%(lineno)d - %(message)s"
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.addFilter(RequestIdFilter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access", "celery"):
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(level)


def get_request_id() -> str:
    return request_id_ctx.get()


def safe_extra(data: dict[str, Any] | None) -> dict[str, Any]:
    """Remove sensitive or overly large values before logging structured details."""
    if not data:
        return {}

    sensitive_keys = {
        "password",
        "plain",
        "hashed_password",
        "authorization",
        "token",
        "access_token",
        "secret",
        "secret_key",
        "api_key",
        "dashscope_api_key",
    }
    cleaned: dict[str, Any] = {}
    for key, value in data.items():
        key_lower = key.lower()
        if any(part in key_lower for part in sensitive_keys):
            cleaned[key] = "<redacted>"
        elif isinstance(value, str) and len(value) > 300:
            cleaned[key] = f"{value[:300]}...<truncated {len(value)} chars>"
        elif isinstance(value, (list, tuple)) and len(value) > 20:
            cleaned[key] = list(value[:20]) + [f"...<truncated {len(value)} items>"]
        elif isinstance(value, dict) and len(value) > 20:
            cleaned[key] = {"_summary": f"<truncated {len(value)} keys>"}
        else:
            cleaned[key] = value
    return cleaned


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        token = request_id_ctx.set(request_id)
        logger = logging.getLogger("app.request")
        start = time.monotonic()
        client_host = request.client.host if request.client else "-"

        logger.info(
            "HTTP request start method=%s path=%s query=%s client=%s",
            request.method,
            request.url.path,
            request.url.query or "-",
            client_host,
        )
        try:
            response = await call_next(request)
            duration_ms = (time.monotonic() - start) * 1000
            response.headers["X-Request-ID"] = request_id
            logger.info(
                "HTTP request complete method=%s path=%s status=%s duration_ms=%.2f",
                request.method,
                request.url.path,
                response.status_code,
                duration_ms,
            )
            return response
        except Exception:
            duration_ms = (time.monotonic() - start) * 1000
            logger.exception(
                "HTTP request failed method=%s path=%s duration_ms=%.2f",
                request.method,
                request.url.path,
                duration_ms,
            )
            raise
        finally:
            request_id_ctx.reset(token)
