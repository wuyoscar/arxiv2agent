"""Unit tests for the dependency-free BibTeX parser."""

from pathlib import Path

from arxiv2agent.bib import parse_bib, resolve


def test_parse_basic_entry():
    text = r"""
@inproceedings{zimmermann2019small,
  title={Small world with high risks},
  author={Zimmermann, Markus and Pradel, Michael},
  booktitle={USENIX Security},
  year={2019}
}
"""
    e = parse_bib(text)
    assert "zimmermann2019small" in e
    rec = e["zimmermann2019small"]
    assert rec["type"] == "inproceedings"
    assert rec["fields"]["title"] == "Small world with high risks"
    assert rec["fields"]["year"] == "2019"


def test_parse_nested_braces_in_title():
    """A title with nested {…} groups must be captured whole, not truncated."""
    text = r"@article{k, title={A {Nested} Title}, year={2020}}"
    e = parse_bib(text)
    assert e["k"]["fields"]["title"] == "A {Nested} Title"


def test_parse_quote_delimited_value():
    text = r'@misc{k, title = "Quoted Title", year = "2021"}'
    e = parse_bib(text)
    assert e["k"]["fields"]["title"] == "Quoted Title"
    assert e["k"]["fields"]["year"] == "2021"


def test_skip_comment_and_string():
    text = r"""
@comment{ignored}
@string{x = "y"}
@article{real, title={Kept}, year={2022}}
"""
    e = parse_bib(text)
    assert set(e.keys()) == {"real"}


def test_resolve_filters_to_cited_keys(tmp_path: Path):
    bib = tmp_path / "main.bib"
    bib.write_text(
        r"""
@inproceedings{cited2020, title={Used Paper}, author={Doe, Jane and Roe, Rich}, booktitle={NeurIPS}, year={2020}}
@article{uncited2019, title={Unused}, author={Nobody, A}, journal={J}, year={2019}}
""",
        encoding="utf-8",
    )
    out = resolve(tmp_path, cited_keys={"cited2020"})
    assert set(out.keys()) == {"cited2020"}          # uncited2019 filtered out
    rec = out["cited2020"]
    assert rec["title"] == "Used Paper"
    assert rec["authors"] == ["Jane Doe", "Rich Roe"]  # "Last, First" → "First Last"
    assert rec["year"] == "2020"
    assert rec["venue"] == "NeurIPS"                  # booktitle
    assert rec["bib_raw"].startswith("@inproceedings{cited2020")


def test_resolve_venue_priority(tmp_path: Path):
    """booktitle wins over journal when both present (unlikely but deterministic)."""
    bib = tmp_path / "x.bib"
    bib.write_text(
        r"@article{k, title={T}, booktitle={BookV}, journal={JourV}, year={2020}}",
        encoding="utf-8",
    )
    out = resolve(tmp_path, cited_keys={"k"})
    assert out["k"]["venue"] == "BookV"


def test_resolve_missing_key_absent(tmp_path: Path):
    bib = tmp_path / "x.bib"
    bib.write_text(r"@article{a, title={T}, year={2020}}", encoding="utf-8")
    out = resolve(tmp_path, cited_keys={"a", "ghost"})
    assert "a" in out
    assert "ghost" not in out      # not in .bib → simply absent, caller fills None
