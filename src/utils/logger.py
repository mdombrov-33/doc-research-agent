import logging
import os
import socket
import sys

import structlog

from src.config import get_settings

_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "openrouter_api_key",
    "openai_api_key",
    "qdrant_api_key",
}


def _add_service_context(
    logger: structlog.types.WrappedLogger,
    method: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    settings = get_settings()
    event_dict.setdefault("service", "doc-research-agent")
    event_dict.setdefault("env", settings.APP_ENV)
    event_dict.setdefault("version", os.getenv("APP_VERSION", "unknown"))
    event_dict.setdefault("git_sha", os.getenv("GIT_SHA", "unknown"))
    event_dict.setdefault("hostname", socket.gethostname())
    return event_dict


def _redact_sensitive(
    logger: structlog.types.WrappedLogger,
    method: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    for key in list(event_dict.keys()):
        if key.lower() in _SENSITIVE_KEYS:
            event_dict[key] = "[REDACTED]"
    return event_dict


def configure_logging() -> None:
    settings = get_settings()
    is_dev = settings.APP_ENV == "development"

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        _add_service_context,
        _redact_sensitive,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    structlog.configure(
        processors=shared_processors + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    renderer: structlog.types.Processor = (
        structlog.dev.ConsoleRenderer(colors=True)
        if is_dev
        else structlog.processors.JSONRenderer()
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.LOG_LEVEL)

    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


configure_logging()

logger: structlog.stdlib.BoundLogger = structlog.get_logger("doc-research-agent")
