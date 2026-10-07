"""Unit tests for the CLI (offline — network-facing functions stubbed)."""

import json

import pytest

from arxiv2agent import cli


def test_digest_batch_processed_sequentially(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cli, "_digest_one", lambda aid, lf, args: calls.append(aid))
    rc = cli.main(["digest", "1111.1111", "2222.2222", "3333.3333", "-o", str(tmp_path)])
    assert rc == 0
    assert calls == ["1111.1111", "2222.2222", "3333.3333"]


def test_batch_continues_after_failure(monkeypatch, tmp_path, capsys):
    def fake(arxiv_id, local_folder, args):
        if arxiv_id == "2222.2222":
            raise RuntimeError("boom")
    monkeypatch.setattr(cli, "_fetch_one", fake)
    rc = cli.main(["fetch", "1111.1111", "2222.2222", "3333.3333", "-o", str(tmp_path)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "FAILED: 2222.2222" in err
    assert "2/3 papers" in err


def test_digest_ids_and_local_folder_are_exclusive(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["digest", "1111.1111", "--local-folder", str(tmp_path)])


def test_find_prints_candidates(monkeypatch, capsys):
    hits = [{"id": "1706.03762", "title": "Attention Is All You Need",
             "authors": ["Ashish Vaswani", "Noam Shazeer"], "year": "2017"}]
    monkeypatch.setattr(cli, "find", lambda title, max_results: hits)
    assert cli.main(["find", "attention", "is", "all", "you", "need"]) == 0
    assert capsys.readouterr().out.startswith("1706.03762\t2017\tAttention Is All You Need")
    assert cli.main(["find", "--json", "attention"]) == 0
    assert json.loads(capsys.readouterr().out) == hits


def test_find_no_match_exits_nonzero(monkeypatch):
    monkeypatch.setattr(cli, "find", lambda title, max_results: [])
    assert cli.main(["find", "nonexistent paper"]) == 1
