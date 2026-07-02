"""Unit tests for marker conversion."""

from arxiv2agent.markers import (
    apply_cite_markers,
    apply_ref_markers,
    apply_footnote_markers,
)


def test_cite_single():
    text, keys = apply_cite_markers(r"As shown by \cite{xu2021}.")
    assert text == "As shown by [@xu2021]."
    assert keys == ["xu2021"]


def test_cite_multi_key():
    text, keys = apply_cite_markers(r"\citep{a,b,c}")
    assert text == "[@a; @b; @c]"
    assert keys == ["a", "b", "c"]


def test_cite_variant_commands():
    text, keys = apply_cite_markers(r"\citet{foo} and \citeauthor{bar}")
    assert "[@foo]" in text
    assert "[@bar]" in text
    assert keys == ["foo", "bar"]


def test_ref_variants():
    text, refs = apply_ref_markers(r"See \ref{fig:1} and \eqref{eq:2} and \autoref{tab:3}.")
    assert "[#fig:1]" in text
    assert "[#eq:2]" in text
    assert "[#tab:3]" in text
    assert refs == ["fig:1", "eq:2", "tab:3"]


def test_footnote_collected():
    text, fns = apply_footnote_markers(r"Hello\footnote{the note}.")
    assert "[^fn:1]" in text
    assert fns == [("fn:1", "the note")]


def test_footnote_numbering_resumes():
    _, fns = apply_footnote_markers(r"\footnote{a}\footnote{b}", start_id=5)
    assert fns == [("fn:5", "a"), ("fn:6", "b")]
