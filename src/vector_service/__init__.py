"""Vector service: OpenAI-compatible embeddings + extensible vector store."""

# Single source of truth for the service version. ``pyproject.toml`` reads
# this attribute dynamically (``[tool.setuptools.dynamic]``), the Prometheus
# ``vs_info`` gauge labels itself with it, ``GET /v1/system/status`` serves it
# to the dashboard, and the dashboard statusbar renders it from that payload.
# Do not copy the number anywhere else — see
# tests/unit/test_version_single_source.py, which fails on a second literal.
__version__ = "0.2.0"
