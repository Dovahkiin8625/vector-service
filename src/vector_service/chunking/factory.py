"""Chunker factory — the one place the API layer talks to."""
from __future__ import annotations

from typing import Any

from vector_service.chunking.base import (
    Chunker,
    get_strategy_class,
    list_strategies,
)
from vector_service.core.logging import get_logger

log = get_logger(__name__)


def build_chunker(
    strategy: str,
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 75,
    options: dict[str, Any] | None = None,
    token_counter=None,
    embed_fn=None,
    chat_fn=None,
) -> Chunker:
    """Construct the chunker registered under ``strategy``.

    ``options`` carries strategy-specific parameters (unknown keys
    are dropped with a warning, so a typo can never 500 a request):

    - fixed: ``protect_code`` (bool, default true);
    - semantic: ``breakpoint_percentile`` (float 0–100, default 95);
    - llm: ``input_token_budget`` (int tokens, default 12000);
    - any strategy: ``min_chunk_size`` (int, default 10% of
      chunk_size) — smaller chunks get merged into neighbours.

    ``embed_fn`` is required by semantic and ``chat_fn`` by llm;
    the API layer supplies them from the loaded embedder /
    configured LLM client.
    """
    if strategy not in list_strategies():
        raise ValueError(
            f"unknown chunking strategy {strategy!r}; "
            f"expected one of {', '.join(list_strategies())}"
        )

    opts = dict(options or {})
    cls = get_strategy_class(strategy)
    kwargs: dict[str, Any] = {"token_counter": token_counter}

    # Strategy-agnostic: every chunker accepts an explicit minimum
    # chunk size (default = 10% of chunk_size, resolved in BaseChunker).
    min_size = _pop_int(opts, "min_chunk_size", 0)
    if min_size > 0:
        kwargs["min_chunk_size"] = min_size

    if strategy == "fixed":
        kwargs["protect_code"] = _pop_bool(opts, "protect_code", True)
    elif strategy == "semantic":
        kwargs["embed_fn"] = embed_fn
        kwargs["breakpoint_percentile"] = _pop_float(
            opts, "breakpoint_percentile", 95.0
        )
    elif strategy == "llm":
        kwargs["chat_fn"] = chat_fn
        kwargs["input_token_budget"] = _pop_int(
            opts, "input_token_budget", 12_000
        )
    # "paragraph" / "recursive" take no strategy-specific options.

    if opts:
        log.warning("chunk_unknown_options", strategy=strategy, keys=sorted(opts))

    return cls(chunk_size, chunk_overlap, **kwargs)


def _pop_bool(opts: dict[str, Any], key: str, default: bool) -> bool:
    if key not in opts:
        return default
    value = opts.pop(key)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in ("0", "false", "no", "off", "")


def _pop_float(opts: dict[str, Any], key: str, default: float) -> float:
    if key not in opts:
        return default
    try:
        return float(opts.pop(key))
    except (TypeError, ValueError):
        return default


def _pop_int(opts: dict[str, Any], key: str, default: int) -> int:
    if key not in opts:
        return default
    try:
        return int(opts.pop(key))
    except (TypeError, ValueError):
        return default
