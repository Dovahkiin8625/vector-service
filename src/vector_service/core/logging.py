"""Structured logging via structlog."""
from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import IO, Any

import structlog

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def setup_logging(
    format: str = "json",
    level: str = "INFO",
    stream: IO[Any] | None = None,
) -> None:
    """Configure structlog + stdlib logging.

    Args:
        format: "json" or "console"
        level: stdlib log level name
        stream: optional output stream (defaults to stderr)
    """
    out = stream or sys.stderr
    log_level = getattr(logging, level.upper(), logging.INFO)

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True, key="ts")

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        timestamper,
        _add_request_id,
    ]

    if format == "json":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=False)

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(out)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)

    _quiet_chatty_loggers(handler, log_level, out=out, formatter=formatter)


# Third-party libraries that flood INFO with pipeline/plugin/download
# chatter. WARNING/ERROR from these still propagates to our handler
# (Docling/HF download- and provider-failure messages are WARNING+),
# so problems stay locatable; set VS_LOG_LEVEL=DEBUG to see every line.
_CHATTY_STDLIB_LOGGERS: tuple[str, ...] = (
    "docling",
    "huggingface_hub",
    "httpx",
    "httpcore",
    "urllib3",
    "filelock",
    "matplotlib",
    "onnxruntime",
    "pdfminer",
)


def _quiet_chatty_loggers(
    handler: logging.Handler,
    log_level: int,
    *,
    out: IO[Any],
    formatter: logging.Formatter,
) -> None:
    """Reduce third-party INFO chatter without hiding real problems.

    - Propagating loggers (Docling, huggingface_hub, ...) are dropped
      to WARNING unless the service itself runs at DEBUG.
    - RapidOCR ships its own ``logging`` logger (``"RapidOCR"``) with
      ``propagate=False`` and a verbose color handler installed at
      import time; root-level filtering cannot reach it. We pre-seed
      that logger with the service handler at WARNING before the
      module is imported — RapidOCR's Logger keeps an existing
      handler — and also normalise a logger RapidOCR already created.
    - The ONNX Runtime *native* (C++) logger is controlled separately
      from Python logging; ``set_default_logger_severity`` silences
      its multi-line yellow "No registered plugin EP" notes that fire
      even on the happy CUDA path. Real C++ errors remain visible.
    """
    third_party_level = logging.DEBUG if log_level <= logging.DEBUG else logging.WARNING

    for name in _CHATTY_STDLIB_LOGGERS:
        logging.getLogger(name).setLevel(third_party_level)

    rapid = logging.getLogger("RapidOCR")
    rapid.propagate = False
    # Give RapidOCR its own handler with our formatter but an
    # independent level — it must NOT share the root handler, or
    # raising its level would silence application INFO logs too.
    # Replace one an earlier setup_logging attached (tagged), and
    # raise the level of RapidOCR's own colored handler if the
    # module was imported before setup_logging ran.
    for existing in list(rapid.handlers):
        if getattr(existing, "_vs_handler", False):
            rapid.removeHandler(existing)
        else:
            existing.setLevel(third_party_level)
    rapid_handler = logging.StreamHandler(out)
    rapid_handler.setFormatter(formatter)
    rapid_handler.setLevel(third_party_level)
    rapid_handler._vs_handler = True  # type: ignore[attr-defined]
    rapid.addHandler(rapid_handler)
    # RapidOCR's import resets its logger level to INFO; the handler
    # level above is what actually filters.
    rapid.setLevel(logging.NOTSET)

    try:
        import onnxruntime as ort

        # 0 VERBOSE, 1 INFO, 2 WARNING, 3 ERROR, 4 FATAL — ORT's
        # native log-severity scale is inverted vs Python's.
        ort.set_default_logger_severity(1 if log_level <= logging.DEBUG else 3)
    except ImportError:
        pass


def _add_request_id(_logger: Any, _name: str, event_dict: dict) -> dict:
    rid = request_id_var.get()
    if rid is not None:
        event_dict.setdefault("request_id", rid)
    return event_dict


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
