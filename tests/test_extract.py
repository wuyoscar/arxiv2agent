"""Unit tests for extract.py."""

from arxiv2agent.extract import (
    extract_figures,
    extract_tables,
    extract_title,
    extract_abstract,
    extract_listings,
    find_appendix_start,
)


def test_extract_one_figure():
    tex = r"""
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{images/foo.pdf}
\caption{An overview of our system.}
\label{fig:overview}
\end{figure}
"""
    figs = extract_figures(tex)
    assert len(figs) == 1
    assert figs[0]["id"] == "fig:overview"
    assert figs[0]["src_refs"] == ["images/foo.pdf"]
    assert figs[0]["caption"].startswith("An overview")


def test_extract_table_with_starred_env():
    tex = r"""
\begin{table*}[t]
\caption{Main results.}
\label{tab:results}
\begin{tabular}{ll}
a & b \\
\end{tabular}
\end{table*}
"""
    tables = extract_tables(tex)
    assert len(tables) == 1
    assert tables[0]["env"] == "table*"
    assert tables[0]["id"] == "tab:results"
    assert tables[0]["caption"].startswith("Main results")
    assert "tabular" in tables[0]["raw_tex"]


def test_extract_table_cites_inside():
    tex = r"""
\begin{table}[t]
\caption{Comparison}
\label{tab:cmp}
\begin{tabular}{ll}
Llama \cite{dubey2024} & 80 \\
\end{tabular}
\end{table}
"""
    tables = extract_tables(tex)
    assert tables[0]["cites_inside"] == ["dubey2024"]


def test_extract_title_and_abstract():
    tex = r"""
\title{Hello World}
\begin{document}
\begin{abstract}
We introduce \emph{a method} that does X.
\end{abstract}
"""
    assert extract_title(tex) == "Hello World"
    abs_, source = extract_abstract(tex)
    assert source == "abstract_env"
    assert "a method" in abs_
    assert "\\emph" not in abs_


def test_extract_abstract_command_form():
    """FAR AI / qpaper templates use \\abstract{...} / \\setabstract{...}."""
    tex = r"""
\begin{document}
\setabstract{This is the abstract content describing the work in over fifty characters of meaningful prose.}
\maketitle
\section{Introduction}
"""
    abs_, source = extract_abstract(tex)
    assert source == "abstract_cmd"
    assert "abstract content" in abs_


def test_extract_abstract_no_source():
    tex = r"\title{X}\begin{document}\section{Body}\end{document}"
    abs_, source = extract_abstract(tex)
    assert abs_ == ""
    assert source == "none"


def test_find_appendix():
    # Real \appendix must be at line start (not buried inside another command).
    assert find_appendix_start("Body content here.\n\\appendix\n\\section{A}") > 0
    assert find_appendix_start("No appendix here.") == -1
    # False positives we MUST reject:
    assert find_appendix_start(r"\appto\appendix{do something}") == -1   # etoolbox
    assert find_appendix_start(r"\let\appendix=\section") == -1          # \let redef


def test_listing_json_body_not_mistaken_for_language():
    """A lstlisting whose body starts with a JSON brace must NOT have that
    brace consumed as minted's `{language}` arg (regression: 28 listings across
    10 papers had their JSON content captured into the language field)."""
    tex = (
        r"\begin{lstlisting}[style=htmlstyle, caption={x}]"
        "\n"
        r'{"is_phishing": "woof", "rationale": "grr"}'
        "\n"
        r"\end{lstlisting}"
    )
    lst = extract_listings(tex)[0]
    assert lst["language"] == ""                       # no language= opt
    assert lst["code"].lstrip().startswith("{")        # body keeps its brace


def test_minted_language_arg_still_parsed():
    tex = "\\begin{minted}{python}\nprint(1)\n\\end{minted}"
    lst = extract_listings(tex)[0]
    assert lst["language"] == "python"


def test_title_with_linebreak_escape():
    """A title containing \\texorpdfstring{\\\\}{} (a `\\\\` line break inside
    braces) must not break brace matching (regression: ICML \\icmltitle papers
    returned empty title because `\\\\}` was misread as an escaped brace)."""
    from arxiv2agent.extract import extract_title
    tex = r"\icmltitle{Agent Smith: One \texorpdfstring{\\}{} Million Agents}" "\n\\begin{document}"
    assert extract_title(tex) == "Agent Smith: One Million Agents"


def test_extract_multi_image_figure_keeps_all_src_refs():
    tex = r"""
\begin{figure}[t]
\centering
\begin{subfigure}{0.48\linewidth}
  \includegraphics[width=\linewidth]{images/a.pdf}
  \caption{Left panel.}
\end{subfigure}
\begin{subfigure}{0.48\linewidth}
  \includegraphics[width=\linewidth]{images/b.png}
  \caption{Right panel.}
\end{subfigure}
\caption{Two panels.}
\label{fig:panels}
\end{figure}
"""
    figs = extract_figures(tex)
    assert len(figs) == 1
    assert figs[0]["src_refs"] == ["images/a.pdf", "images/b.png"]
    # subfigure scaffolding alone is layout, not content
    assert figs[0]["body_tex"] == ""


def test_extract_text_body_figure_preserves_body():
    tex = r"""
\begin{figure}[t]
\centering
\fbox{\parbox{0.9\linewidth}{Prompt: You are a helpful assistant.
Please answer the following question truthfully and cite your sources.}}
\caption{The system prompt used in all experiments.}
\label{fig:prompt}
\end{figure}
"""
    figs = extract_figures(tex)
    assert len(figs) == 1
    assert figs[0]["src_refs"] == []
    body = figs[0]["body_tex"]
    assert "You are a helpful assistant" in body
    assert "\\caption" not in body
    assert "\\label" not in body


def test_extract_image_only_figure_has_empty_body():
    tex = r"""
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{images/foo.pdf}
\caption{Image only.}
\label{fig:img}
\end{figure}
"""
    figs = extract_figures(tex)
    assert figs[0]["body_tex"] == ""
