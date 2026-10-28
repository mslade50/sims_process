"""Public CLI for the discovery-only complete base-skill market evaluator."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from archive.evaluate_complete_base_skill_market_candidates import build_parser, run


if __name__ == "__main__":
    run(build_parser().parse_args())
