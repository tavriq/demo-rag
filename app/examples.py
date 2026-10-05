"""Questions shown on the page (example buttons, placeholder) and spot checks for evals.

Single source: data/spot_checks.jsonl. A test keeps app/static/index.html in sync
with it, and ``python3 -m app.evals`` reports where the hybrid ranks the expected
article for each line, so a bad example is visible in the eval report.
"""

from __future__ import annotations

import json
from pathlib import Path

SPOT_CHECKS_PATH = Path(__file__).resolve().parents[1] / "data" / "spot_checks.jsonl"


def load_spot_checks(path: str | Path = SPOT_CHECKS_PATH) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ui_examples(path: str | Path = SPOT_CHECKS_PATH) -> list[str]:
    return [row["q"] for row in load_spot_checks(path) if row["type"] == "ui_example"]
