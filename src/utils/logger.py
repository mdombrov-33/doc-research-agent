import logging
import sys

import structlog

from src.config import get_settings


def _add_service_context(logger: object, method: str, event_dict: dict) -> dict:
    settings = get_settings()
    event_dict.setdefault("service", "doc-research-agent")
    event_dict.setdefault("env", settings.APP_ENV)
    return event_dict


def configure_logging() -> None:
    settings = get_settings()
    is_dev = settings.APP_ENV == "development"

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        _add_service_context,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    structlog.configure(
        processors=shared_processors
        + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
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
