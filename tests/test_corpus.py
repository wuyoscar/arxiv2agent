"""Golden-corpus regression tests over real arXiv papers.

Excluded from the default run (see addopts in pyproject.toml). Usage:

    pytest -m corpus                    # assert against tests/golden/*.json
    pytest -m corpus --update-golden    # regenerate snapshots after an
                                        # intentional extraction change

First run downloads each paper's LaTeX source (politely spaced ≥3s per
arXiv guidance); afterwards everything is served from the local cache and
the suite is offline and deterministic. Sources are never committed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from arxiv2agent import digest
from arxiv2agent._tex import _is_valid_cache_dir, get_default_cache_dir

_TESTS_DIR = Path(__file__).parent
_GOLDEN_DIR = _TESTS_DIR / "golden"
_LOCK = json.loads((_TESTS_DIR / "corpus_lock.json").read_text())

_ARXIV_POLITENESS_SECONDS = 3


def summarize(paper: dict) -> dict:
    """Reduce a full digest to the stable fingerprint we snapshot.

    Counts + section skeleton + quality metrics: enough to catch any
    extraction regression, small enough to review in a diff.
    """
    citations = paper["citations"]
    resolved = sum(1 for c in citations if c.get("title"))
    return {
        "paper_id": paper["paper_id"],
        "title": paper["metadata"]["title"],
        "abstract_source": paper["metadata"]["abstract_source"],
        "n_authors": len(paper["metadata"]["authors"]),
        "n_sections": len(paper["sections"]),
        "n_appendix_sections": sum(1 for s in paper["sections"] if s["is_appendix"]),
        "n_figures": len(paper["figures"]),
        "n_figure_src_refs": sum(len(f["src_refs"]) for f in paper["figures"]),
        "n_figures_with_body": sum(
            1 for f in paper["figures"] if (f.get("body_tex") or "").strip()
        ),
        "n_tables": len(paper["tables"]),
        "n_equations": len(paper["equations"]),
        "n_algorithms": len(paper["algorithms"]),
        "n_listings": len(paper["listings"]),
        "n_footnotes": len(paper["footnotes"]),
        "n_citations": len(citations),
        "citation_resolution_rate": (
            round(resolved / len(citations), 3) if citations else None
        ),
        "residue_section_count": paper["warnings"]["residue_section_count"],
        "section_ids": [s["id"] for s in paper["sections"]],
        "section_titles": [s["title"] for s in paper["sections"]],
    }


def _digest_politely(arxiv_id: str) -> dict:
    if not _is_valid_cache_dir(get_default_cache_dir() / arxiv_id):
        time.sleep(_ARXIV_POLITENESS_SECONDS)
    return digest(arxiv_id=arxiv_id)


@pytest.mark.corpus
@pytest.mark.parametrize("entry", _LOCK["papers"], ids=lambda e: e["arxiv_id"])
def test_corpus_paper(entry: dict, request: pytest.FixtureRequest) -> None:
    got = summarize(_digest_politely(entry["arxiv_id"]))

    # Sanity gates independent of the snapshot — a paper must always yield
    # a title and real structure, even right after --update-golden.
    assert got["title"], f"{entry['arxiv_id']}: no title extracted"
    assert got["n_sections"] >= 3, f"{entry['arxiv_id']}: suspiciously few sections"
    assert got["n_citations"] > 0, f"{entry['arxiv_id']}: no citations found"

    golden_path = _GOLDEN_DIR / f"{entry['arxiv_id'].replace('/', '_')}.json"
    if request.config.getoption("--update-golden"):
        _GOLDEN_DIR.mkdir(exist_ok=True)
        golden_path.write_text(json.dumps(got, indent=2, ensure_ascii=False) + "\n")
        pytest.skip(f"golden updated: {golden_path.name}")

    assert golden_path.exists(), (
        f"No golden snapshot for {entry['arxiv_id']}. "
        f"Run: pytest -m corpus --update-golden"
    )
    want = json.loads(golden_path.read_text())
    assert got == want, (
        f"{entry['arxiv_id']} digest fingerprint changed. If intentional "
        f"(or the paper has a new arXiv version), re-run with --update-golden."
    )
