"""Unit test for fetch (offline — download stubbed with a fake cache)."""

import importlib

fetch_mod = importlib.import_module("arxiv2agent.fetch")


def test_fetch_copies_tree_and_flattens(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    paper = cache / "2401.00001"
    (paper / "sec").mkdir(parents=True)
    (paper / "main.tex").write_text(
        "\\documentclass{article}\n\\begin{document}\n\\input{sec/intro}\n\\end{document}\n"
    )
    (paper / "sec" / "intro.tex").write_text("Hello. % author note\n")
    (paper / ".arxiv_cache_complete").write_text("ok\n")
    monkeypatch.setattr(fetch_mod, "get_default_cache_dir", lambda: cache)
    monkeypatch.setattr(fetch_mod, "download_arxiv_source", lambda aid, use_cache: True)

    root = fetch_mod.fetch("2401.00001", tmp_path / "out")
    assert (root / "source" / "sec" / "intro.tex").is_file()
    assert not (root / "source" / ".arxiv_cache_complete").exists()
    flat = (root / "2401.00001.tex").read_text()
    assert "Hello. % author note" in flat  # inlined, comments kept
