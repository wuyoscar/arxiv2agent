"""Unit tests for the denoise PIPELINE."""

from arxiv2agent.denoise import PIPELINE, denoise


def test_pipeline_is_an_ordered_list_of_rules():
    """PIPELINE is the canonical denoise sequence — visible for review."""
    names = [r.name for r in PIPELINE]
    # A few canonical rule names must exist (renaming requires test update)
    assert {'block-strip', 'markup', 'noise', 'whitespace', 'hyperlinks'} <= set(names)
    # math-mask/restore must bracket the whole pipeline (BUG-1: protect $…$)
    assert names[0] == 'math-mask'
    assert names[-1] == 'math-restore'
    assert names.index('whitespace') < names.index('math-restore')


# ─── block stripping ───────────────────────────────────────────────────────


def test_strip_figure_block():
    text = r"""Before figure.
\begin{figure}[t]
\centering
\includegraphics{foo.pdf}
\caption{Foo}
\label{fig:foo}
\end{figure}
After figure."""
    out = denoise(text)
    assert "Before figure." in out
    assert "After figure." in out
    assert "includegraphics" not in out
    assert "\\caption" not in out


def test_strip_table_block_with_backreference():
    text = r"""Pre.
\begin{table*}[t]
\caption{T}
\begin{tabular}{ll} a & b \\ \end{tabular}
\end{table*}
Post."""
    out = denoise(text)
    assert "Pre." in out
    assert "Post." in out
    assert "tabular" not in out


# ─── markup: markdown (default) ────────────────────────────────────────────


def test_markup_default_is_markdown():
    """Default style preserves author emphasis via markdown wrappers."""
    text = r"This is \textit{emphasized} and \textbf{bold} and \texttt{code}."
    out = denoise(text)
    assert out == "This is *emphasized* and **bold** and `code`."


def test_markup_nested_double_class():
    text = r"\textbf{\textit{nested}} text"
    out = denoise(text)
    # Inner italic then outer bold; markdown allows ***foo***
    assert "*nested*" in out and "**" in out


def test_markup_dropped_wrappers():
    """Wrappers with no markdown equivalent are silently unwrapped."""
    text = r"Some \textsc{SmallCaps} and \underline{stuff}."
    out = denoise(text)
    assert "textsc" not in out
    assert "underline" not in out
    assert "SmallCaps" in out
    assert "stuff" in out


# ─── balanced-brace markup (regression: real-paper nested cases) ──────────


def test_markup_balanced_braces():
    """Real papers contain ``\\textbf{Finding \\refstepcounter{c}\\thec:}``
    where the textbf arg itself has nested ``{...}`` subgroups. The previous
    ``[^{}]*`` regex silently failed and left ``\\textbf`` as residue.
    """
    text = r"\textbf{Finding \refstepcounter{c}\thec:} body"
    out = denoise(text)
    assert "\\textbf" not in out, f"textbf leaked: {out!r}"
    assert "Finding" in out
    assert "**" in out


def test_markup_nested_other_markup():
    """``\\textbf{A \\textit{B} C}`` → ``**A *B* C**``."""
    text = r"\textbf{A \textit{B} C}"
    out = denoise(text)
    assert "\\textbf" not in out
    assert "\\textit" not in out
    assert "*B*" in out
    assert "**" in out


# ─── residual zero-arg commands ────────────────────────────────────────────


def test_strip_residual_zero_arg_macros():
    text = r"We propose \mymethod{} for this task."
    out = denoise(text)
    assert out == "We propose for this task."


# ─── whitespace + tilde ────────────────────────────────────────────────────


def test_collapse_whitespace_and_tilde():
    text = "Hello~world  with    extra\n\n\n\nbreaks."
    out = denoise(text)
    assert "Hello world" in out
    assert "    " not in out
    assert "\n\n\n" not in out


# ─── markers passthrough ───────────────────────────────────────────────────


def test_does_not_touch_cite_or_ref_markers():
    """[@key] and [#ref] markers placed by markers.py must survive denoise."""
    text = r"See [@xu2021] and [#fig:1]."
    out = denoise(text)
    assert "[@xu2021]" in out
    assert "[#fig:1]" in out


# ─── section heading ───────────────────────────────────────────────────────


def test_strip_section_command():
    text = r"\section{Introduction} The body starts here."
    out = denoise(text)
    assert "Introduction" not in out
    assert "The body starts here." in out


def test_strip_section_command_with_nested_braces():
    """Section command stripping must handle nested groups from macro expansion."""
    text = r"\section{{ExpandedName}{}} The body."
    out = denoise(text)
    assert "ExpandedName" not in out
    assert "The body." in out


# ─── escape normalisation ──────────────────────────────────────────────────


def test_escape_chars_normalised():
    text = r"We saw 97.44\% accuracy and 5\& improvement."
    out = denoise(text)
    assert "97.44%" in out
    assert "5&" in out
    assert "\\%" not in out
    assert "\\&" not in out


# ─── paragraph heading → markdown bold ────────────────────────────────────


def test_paragraph_heading_becomes_markdown():
    text = r"\paragraph{Setup.} We use 4 GPUs."
    out = denoise(text)
    assert "**Setup.**" in out
    assert "We use 4 GPUs." in out


# ─── lossless preservation (new in v0.3) ───────────────────────────────────


def test_lstlisting_stripped_from_text():
    """v0.3+: code listings are extracted to listings[] side table — the
    section text should be tight without the code dumped inline."""
    text = r"""See:
\begin{lstlisting}[language=Python]
def attack(prompt):
    return query(prompt)
\end{lstlisting}
Done."""
    out = denoise(text)
    assert "def attack" not in out
    assert "lstlisting" not in out
    # Surrounding prose should survive
    assert "See:" in out
    assert "Done." in out


def test_theorem_env_becomes_blockquote():
    text = r"""\begin{theorem}[Universality]
For any distribution $p$, the bound holds.
\end{theorem}"""
    out = denoise(text)
    assert "**Theorem (Universality).**" in out
    assert "the bound holds" in out
    # Blockquote-formatted: every content line should begin with '>'
    body_lines = [ln for ln in out.split('\n') if ln.strip()]
    # The headline + body are all blockquote-prefixed
    assert any(ln.startswith('> ') for ln in body_lines)


def test_mdframed_dialog_preserved():
    """Multi-turn dialog wrapped in mdframed inside figure must survive."""
    text = r"""\begin{figure}
\begin{mdframed}
\item[A:] Question one.
\item[B:] Answer one.
\end{mdframed}
\caption{Dialog example}
\end{figure}"""
    out = denoise(text)
    assert "Question one" in out
    assert "Answer one" in out


def test_image_figure_still_stripped():
    """Figure with \\includegraphics is captured in figures[]; drop from body."""
    text = r"""Pre.
\begin{figure}
\includegraphics{x.pdf}
\caption{An image}
\label{fig:x}
\end{figure}
Post."""
    out = denoise(text)
    assert "Pre." in out
    assert "Post." in out
    assert "includegraphics" not in out
    assert "An image" not in out   # caption lives in figures[].caption


# ─── 2026-06 audit fixes (BUG-1..7): regression guards ──────────────────────

def test_bug1_math_mode_not_markdownified():
    """\\mathbf inside $…$ must stay LaTeX, never become **x** (BUG-1)."""
    out = denoise(r"The vector $\mathbf{x}$ and display $$\mathbf{A}_t = \theta$$ hold.")
    assert r"$\mathbf{x}$" in out
    assert "**" not in out          # no markdown bold injected into math
    assert r"\mathbf{A}_t" in out   # display math preserved verbatim
    assert r"\theta" in out         # \to/\theta-style tokens not stripped


def test_bug1_textbf_outside_math_still_bolds():
    """Masking math must not disable normal \\textbf conversion."""
    out = denoise(r"\textbf{Bold} and $\mathbf{m}$ together.")
    assert "**Bold**" in out
    assert r"$\mathbf{m}$" in out


def test_bug2_paragraph_with_nested_command():
    """\\paragraph{\\textbf{X}} must not leak an orphan \\paragraph (BUG-2)."""
    out = denoise(r"\paragraph{\textbf{Agents.}} Body text.")
    assert r"\paragraph" not in out
    assert "**Agents.**" in out


def test_bug4_hyperlinks_collapse_in_body():
    """\\href/\\url/\\hyperref collapse to visible text in section body (BUG-4)."""
    out = denoise(r"See \href{https://x.com}{our code}, \url{http://y.org}, \hyperref[s2]{Section 2}.")
    assert "our code" in out and r"\href" not in out
    assert "http://y.org" in out and r"\url" not in out
    assert "Section 2" in out and r"\hyperref" not in out


def test_bug5_nested_bold_collapses():
    """\\textbf{\\textbf{X}} → **X**, not ****X**** (BUG-5)."""
    out = denoise(r"\textbf{\textbf{Dataset.}} Numbers.")
    assert "****" not in out
    assert "**Dataset.**" in out


def test_bug7_layout_commands_stripped():
    """\\balance/\\clearpage/\\enddocument etc. don't leak as residue (BUG-7)."""
    out = denoise(r"End. \balance \clearpage \onecolumn More.")
    for cmd in (r"\balance", r"\clearpage", r"\onecolumn"):
        assert cmd not in out
    assert "End." in out and "More." in out


def test_denoise_unwraps_boxes_and_color():
    src = (r"\fbox{ \tt \parbox{0.9\linewidth}{ {\color{gray}Title: A \\ Article:} "
           r"\textbf{Body text} } }")
    out = denoise(src)
    assert "\\fbox" not in out and "\\parbox" not in out and "\\color" not in out
    assert "Title: A" in out
    assert "**Body text**" in out


def test_denoise_converts_forced_linebreaks():
    out = denoise(r"line one \\ line two \\[2mm] line three")
    assert out.splitlines() == ["line one", "line two", "line three"]


def test_denoise_textcolor_keeps_inner_text():
    assert denoise(r"\textcolor{red}{warning text}") == "warning text"
