"""Contract test for the real CrossEncoderReranker.

Skipped by default on Windows (matches project convention from b86e077).
Run explicitly with::

    pytest -m contract tests/contract/test_cross_encoder_reranker.py
"""
from __future__ import annotations

import sys

import pytest

from vector_service.core.config import RerankerSettings, Settings
from vector_service.rerankers.cross_encoder import CrossEncoderReranker

pytestmark = [pytest.mark.contract, pytest.mark.slow]

# Match project convention: skip contract tests on Windows by default.
_SKIP_ON_WINDOWS = pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="contract tests require model weights; skip on Windows by default",
)


@_SKIP_ON_WINDOWS
def test_bge_reranker_v2_m3_orders_relevant_first():
    settings = Settings(
        reranker=RerankerSettings(
            backend="bge-reranker-v2-m3",
            model_dir="./models/bge-reranker-v2-m3",
        )
    )
    r = CrossEncoderReranker(settings=settings)
    r.load()
    docs = [
        "巴黎是法国首都。",
        "苹果是一种水果。",
        "北京是中华人民共和国的首都。",
    ]
    hits = r.rerank("中国首都", docs, top_n=3)
    assert hits[0].index == 2
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
