"""End-to-end smoke test: CLI entry point → digest folder on disk.

Excluded from the default run; enable with `pytest -m e2e`. Uses one small,
stable paper; the download is cached after the first run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arxiv2agent.cli import main

_ARXIV_ID = "1706.03762"


@pytest.mark.e2e
def test_cli_writes_complete_digest_folder(tmp_path: Path) -> None:
    rc = main([_ARXIV_ID, "-o", str(tmp_path)])
    assert rc == 0

    root = tmp_path / _ARXIV_ID
    assert root.is_dir()

    # Canonical record loads and self-identifies.
    paper = json.loads((root / "paper.json").read_text())
    assert paper["paper_id"] == _ARXIV_ID
    assert paper["metadata"]["title"]

    # Human entry point + aggregations exist.
    assert (root / "README.md").is_file()
    assert (root / "references.json").is_file()

    # Every section in paper.json has exactly one markdown file with
    # YAML frontmatter.
    section_files = sorted((root / "sections").glob("*.md"))
    assert len(section_files) == len(paper["sections"])
    for f in section_files:
        assert f.read_text().startswith("---\n"), f"{f.name}: missing frontmatter"

    # --include-source is off by default: no raw LaTeX mirror.
    assert not (root / "source").exists()
