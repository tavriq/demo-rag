"""UI examples come from data/spot_checks.jsonl and must rank the expected article first.

The ranking test needs the real index and embedding model (var/index, var/models)
and is skipped without them; the official check is the spot-check table in the
eval report, made on the demo server (python3 -m app.evals).
"""

import re
from pathlib import Path

import pytest

from app.examples import load_spot_checks, ui_examples

ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = ROOT / "app" / "static" / "index.html"


def test_page_examples_match_spot_checks():
    html = INDEX_HTML.read_text(encoding="utf-8")
    chips = re.findall(r'<button type="button" class="chip">([^<]+)</button>', html)
    assert chips == ui_examples()
    placeholder = [r["q"] for r in load_spot_checks() if r["type"] == "ui_placeholder"]
    assert placeholder and f'placeholder="Например: {placeholder[0]}"' in html


def test_readme_curl_example_is_a_spot_check():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    example = [r["q"] for r in load_spot_checks() if r["type"] == "readme_example"]
    assert example and f'{{"question": "{example[0]}"}}' in readme


def _real_retriever():
    from app.config import Settings
    from app.main import _load_retriever

    settings = Settings.from_env({"VAR_DIR": str(ROOT / "var")})
    if not (settings.index_dir / "meta.json").exists() or not any(settings.models_dir.glob("**/*.onnx")):
        pytest.skip("real index or embedding model not built locally")
    try:
        return _load_retriever(settings)
    except Exception as exc:  # e.g. the index was built with the hash embedder
        pytest.skip(f"real index not loadable: {exc}")


def test_examples_rank_expected_article_first_on_real_index():
    retriever = _real_retriever()
    for row in load_spot_checks():
        if row["type"] not in ("ui_example", "ui_placeholder", "readme_example"):
            continue
        top = retriever.search(row["q"], top_k=1)[0].chunk.doc_id
        assert top == row["expected_id"], row["q"]
