"""Download an arXiv paper's LaTeX source as-is, plus one flattened .tex."""

from __future__ import annotations

import shutil
from pathlib import Path

from arxiv2agent._tex import (
    _CACHE_COMPLETE_MARKER,
    download_arxiv_source,
    find_main_tex,
    flatten_tex,
    get_default_cache_dir,
)
from arxiv2agent.arxiv_api import _CACHE_FILENAME


def fetch(arxiv_id: str, output_dir: str | Path = ".") -> Path:
    """Write ``<output_dir>/<id>/source/`` (untouched tree) and
    ``<output_dir>/<id>/<id>.tex`` (\\input-expanded, comments kept).
    Returns the paper directory."""
    if not download_arxiv_source(arxiv_id, use_cache=True):
        raise RuntimeError(f"no LaTeX source available for {arxiv_id}")
    cached = get_default_cache_dir() / arxiv_id
    safe_id = arxiv_id.replace("/", "_")
    root = Path(output_dir) / safe_id
    src = root / "source"
    if src.exists():
        shutil.rmtree(src)
    shutil.copytree(cached, src, ignore=shutil.ignore_patterns(
        _CACHE_COMPLETE_MARKER, _CACHE_FILENAME,
    ))
    main = find_main_tex(str(src))
    if main is None:
        raise RuntimeError(f"no main .tex (\\begin{{document}}) found for {arxiv_id}")
    (root / f"{safe_id}.tex").write_text(flatten_tex(str(src), main), encoding="utf-8")
    return root
