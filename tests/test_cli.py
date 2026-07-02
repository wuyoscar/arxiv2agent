"""Unit tests for the batch CLI (offline — digest/write_digest stubbed)."""

from arxiv2agent import cli


def test_batch_ids_processed_sequentially(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cli, "digest", lambda arxiv_id, local_folder: {"paper_id": arxiv_id})
    monkeypatch.setattr(
        cli, "write_digest",
        lambda paper, output_dir, source_folder, include_source: tmp_path / paper["paper_id"],
    )
    monkeypatch.setattr(cli, "_run_one", cli._run_one)  # keep real orchestration

    def fake_run_one(arxiv_id, local_folder, args):
        calls.append(arxiv_id)
        return 0
    monkeypatch.setattr(cli, "_run_one", fake_run_one)

    rc = cli.main(["1111.1111", "2222.2222", "3333.3333", "-o", str(tmp_path)])
    assert rc == 0
    assert calls == ["1111.1111", "2222.2222", "3333.3333"]


def test_batch_continues_after_failure(monkeypatch, tmp_path, capsys):
    def fake_run_one(arxiv_id, local_folder, args):
        if arxiv_id == "2222.2222":
            raise RuntimeError("boom")
        return 0
    monkeypatch.setattr(cli, "_run_one", fake_run_one)

    rc = cli.main(["1111.1111", "2222.2222", "3333.3333", "-o", str(tmp_path)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "FAILED: 2222.2222" in err
    assert "2/3 papers digested" in err


def test_ids_and_local_folder_are_exclusive(tmp_path):
    import pytest
    with pytest.raises(SystemExit):
        cli.main(["1111.1111", "--local-folder", str(tmp_path)])
