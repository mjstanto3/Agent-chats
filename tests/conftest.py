"""pytest configuration – ensures the repo root is on sys.path."""

from __future__ import annotations

import sys
from pathlib import Path

# Add the repository root to sys.path so that ``multi_agent_debate`` is
# importable regardless of how pytest is invoked (e.g. from any working
# directory, via ``python -m pytest``, or via a bare ``pytest`` call).
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
