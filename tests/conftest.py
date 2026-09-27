"""Shared pytest setup.

Several tests ``monkeypatch.chdir`` into a temporary directory (or run
without the repo root on the cwd). The process settings are then built
without the project ``.env`` file, and the one field with no default —
``RerankerSettings.backend`` — would make ``get_settings()`` raise.
Seed it (and anything else required later) via env vars, but never
override a value the operator already exported.
"""
from __future__ import annotations

import os

os.environ.setdefault("VS_RERANKER__BACKEND", "bge-reranker-v2-m3")
