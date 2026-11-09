"""Launch the vector service.

Usage:
    python run.py

Reads host/port/workers/log level from VS_* settings (see .env / .env.example).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Support running from a source checkout without installing the package.
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))


def main() -> None:
    from vector_service.main import run

    run()


if __name__ == "__main__":
    main()
