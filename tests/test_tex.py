"""Unit tests for the vendored LaTeX flatten/macro layer (_tex.py)."""

from arxiv2agent.extract import extract_affiliations, _clean_affiliation
from arxiv2agent._tex import expand_macros


def test_protected_section_redefinition_not_expanded():
    """A paper that redefines \\section via \\def (common in journal styles)
    must keep `\\section{X}` literal — else the section parser finds nothing
    (regression: 34 papers redefine a structural command; some had 0 sections)."""
    tex = (
        r"\def\section{\@startsection{section}{1}{\z@}{-3ex}{2ex}{\large\bf}}"
        "\n"
        r"\section{Introduction}"
        "\n"
        r"Body text."
    )
    out = expand_macros(tex)
    assert r"\section{Introduction}" in out          # usage stays literal
    assert r"\@startsection" not in out              # redefinition not applied


def test_protected_renewcommand_title_not_expanded():
    tex = r"\renewcommand{\title}[1]{\Large #1}" "\n" r"\title{My Paper}"
    out = expand_macros(tex)
    assert r"\title{My Paper}" in out


def test_normal_macro_still_expands():
    tex = r"\newcommand{\foo}{bar}" "\n" r"a \foo b"
    out = expand_macros(tex)
    assert "bar" in out
    assert r"\foo" not in out


def test_find_main_tex_prefers_begin_document_over_documentclass(tmp_path):
    """When main.tex holds the body (\\begin{document}) but \\input's a preamble
    file that holds \\documentclass, the master must be main.tex — not the
    subdir preamble (regression: body in Sections/ , tex/ came out 0-section)."""
    from arxiv2agent._tex import find_main_tex
    (tmp_path / "Sections").mkdir()
    # preamble (in a subdir) carries \documentclass but NOT the body
    (tmp_path / "Sections" / "0-preamble.tex").write_text(
        r"\documentclass{article}\usepackage{amsmath}", encoding="utf-8")
    # the real master: body + \begin{document}, no \documentclass of its own
    (tmp_path / "main.tex").write_text(
        r"\input{Sections/0-preamble}" "\n" r"\begin{document}\section{Intro}\end{document}",
        encoding="utf-8")
    assert find_main_tex(str(tmp_path)) == "main.tex"


# ── affiliation extraction (institution set, structured templates only) ──


def test_affiliation_acmart_institution():
    tex = r"""
    \author{Jingyu Tang$^{\dagger}$}
    \affiliation{%
      \institution{University of Notre Dame}
      \city{Notre Dame}\country{USA}}
    \author{Bob}
    \affiliation{\institution{University of Michigan}}
    """
    affs, src = extract_affiliations(tex)
    assert affs == ["University of Notre Dame", "University of Michigan"]
    assert src == "acmart_institution"


def test_affiliation_dedupes_repeats():
    tex = r"""
    \affiliation{\institution{National ChengChi University}}
    \affiliation{\institution{National ChengChi University}}
    """
    affs, src = extract_affiliations(tex)
    assert affs == ["National ChengChi University"]


def test_affiliation_icml():
    tex = r"\icmlaffiliation{goo}{Google DeepMind, London, UK}"
    affs, src = extract_affiliations(tex)
    assert affs == ["Google DeepMind, London, UK"]
    assert src == "icml"


def test_affiliation_llncs_and_split():
    tex = r"\institute{University of Pavia \and University of Padua}"
    affs, src = extract_affiliations(tex)
    assert affs == ["University of Pavia", "University of Padua"]
    assert src == "llncs_institute"


def test_affiliation_individual_when_truly_bare():
    # no institution keyword, no email anywhere → individual, not a wrong guess
    tex = r"\author{Marco Arazzi, Vinod P.}"
    affs, src = extract_affiliations(tex)
    assert affs == ["individual"]
    assert src == "individual"


def test_affiliation_authorblock_freetext_after_linebreak():
    # TUM-style: institution written inside \author{} after \\
    tex = (
        r"\author{Yuxiao Li, Alina Fastowski, Gjergji Kasneci \\"
        "\n\\\\\nTechnical University of Munich \\\\\n"
        r"Munich Center for Machine Learning \\"
        "\n\\texttt{\\{name.surname\\}@tum.de}\n}"
    )
    affs, src = extract_affiliations(tex)
    assert "Technical University of Munich" in affs
    assert "Munich Center for Machine Learning" in affs
    assert "authorblock" in src
    assert not any("Fastowski" in a for a in affs)  # name line must not leak in


def test_affiliation_superscript_numbered_block():
    tex = (
        r"\author{Haoran Gao$^{1}$, Yang Liu$^3$ \\"
        "\n$^1$China Mobile Research Institute. "
        r"$^3$Nanyang Technological University. \\"
        "\n\\texttt{x@ntu.edu.sg}\n}"
    )
    affs, src = extract_affiliations(tex)
    assert any("China Mobile" in a for a in affs)
    assert any("Nanyang Technological University" in a for a in affs)


def test_affiliation_corporate_email_domain():
    # only superscripts in the block, but a corporate email reveals the org
    tex = r"\author{Jane Doe$^{1}$} \texttt{jane@anthropic.com}"
    affs, src = extract_affiliations(tex)
    assert "Anthropic" in affs
    assert "email" in src


def test_clean_affiliation_strips_markers_and_emails():
    assert _clean_affiliation(r"University of X$^{\dagger}$ \\ \texttt{a@x.edu}") == "University of X"
