"""Denoise per-section text after extraction + marker conversion.

PIPELINE is a fixed ordered list of `Rule(name, description, apply)`. Each
rule is a pure ``str -> str`` transform. `denoise(text)` runs them in order;
no runtime configuration. The list IS the documentation — read it top to
bottom to know exactly what happens to a section's text after `markers.py`
has done its job.

To add a new transform: append a `Rule` to `PIPELINE`. Don't add config
parameters — keep this simple.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

# ─────────────────────────────────────────────────────────────────────────────
# Block environments (content stored in side tables — strip from body)
# ─────────────────────────────────────────────────────────────────────────────

# Blocks we always strip (content lives in side tables — equations/algorithms —
# or in dedicated extracted records). For figure/table we use smart strip
# below that preserves text-only floats (dialog examples, prompt boxes).
_ALWAYS_STRIP_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r'\\if0\b.*?\\fi\b', flags=re.DOTALL),
    re.compile(
        r'\\begin\{(equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?)\}.*?\\end\{\1\}',
        flags=re.DOTALL,
    ),
    re.compile(
        r'\\begin\{(algorithm\*?|algorithmic\*?|algorithm2e)\}.*?\\end\{\1\}',
        flags=re.DOTALL,
    ),
]

# Float-wrapper environments. We don't want to lose text/code/dialog inside
# them just because the LaTeX author wrapped a content block in a float
# (common pattern: \begin{figure}\begin{mdframed}…dialog…\end{mdframed}\end{figure}).
#
# Strategy: if the float contains \includegraphics (a real image), drop the
# whole thing (caption was already captured into figures[]/tables[]); if not,
# keep the body — only strip the surrounding \begin{float}, \caption, \label,
# \centering, \begin{table}, \end{...} wrapper.
_FLOAT_RE = re.compile(
    r'\\begin\{(?P<env>figure\*?|table\*?|longtable|sidewaystable\*?|'
    r'wraptable\*?|wrapfigure\*?|listing|figure\*)\}'
    r'\s*(?:\[[^\]]*\])?'
    r'(?P<body>.*?)'
    r'\\end\{(?P=env)\}',
    flags=re.DOTALL,
)
_INCLUDEGRAPHICS_RE = re.compile(r'\\includegraphics\b')
_TABULAR_RE = re.compile(r'\\begin\{tabular\*?\}|\\begin\{tabularx\}|\\begin\{longtable\}')
_FLOAT_WRAPPER_CMDS_RE = re.compile(
    r'\\(?:caption|centering|label|begin\{(?:figure|table|listing)\*?\}|'
    r'end\{(?:figure|table|listing)\*?\})(?:\[[^\]]*\])?(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})?'
)


def _smart_strip_floats(text: str) -> str:
    """Strip image floats but PRESERVE text-only floats (dialog frames, prompts).

    A float with \\includegraphics → drop whole block (caption is in figures[]).
    A float with \\begin{tabular} → drop whole block (raw_tex is in tables[]).
    Otherwise → keep inner body, drop wrapper commands.
    """
    def repl(m: re.Match[str]) -> str:
        body = m.group('body')
        if _INCLUDEGRAPHICS_RE.search(body) or _TABULAR_RE.search(body):
            return ''   # drop — content is already in side table
        # Preserve body, just strip wrapper commands inside it
        kept = _FLOAT_WRAPPER_CMDS_RE.sub('', body).strip()
        return f'\n\n{kept}\n\n' if kept else ''
    return _FLOAT_RE.sub(repl, text)


def _strip_blocks(text: str) -> str:
    for pat in _ALWAYS_STRIP_PATTERNS:
        text = pat.sub('', text)
    text = _smart_strip_floats(text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Blocks we CONVERT to markdown (lossless inline preservation).
#
# These environments carry first-class paper content that an agent reading the
# section sequentially still wants to see:
#   - Code listings → fenced code blocks ``` ... ```
#   - Theorem-likes  → blockquote with bold type heading: "> **Theorem 1.** ..."
#   - Callout boxes → blockquote                          "> ..."
#   - Dialog frames (mdframed wrapping multi-turn examples)
#
# These don't move into a side table — they live inline in `section.text` so
# the narrative flows. Stripping them entirely lost too much content
# (Crescendo's multi-turn dialog, papers' Remark boxes, proof contents).
# ─────────────────────────────────────────────────────────────────────────────

# lstlisting / verbatim / minted are now captured to a dedicated listings[]
# side table (see extract.extract_listings). We just need to drop them from
# section text since content lives elsewhere.
_CODE_BLOCK_PATTERNS = [
    re.compile(
        r'\\begin\{(lstlisting|minted)\*?\}.*?\\end\{\1\*?\}',
        flags=re.DOTALL,
    ),
    re.compile(
        r'\\begin\{verbatim\*?\}.*?\\end\{verbatim\*?\}',
        flags=re.DOTALL,
    ),
]


def _convert_code_listings(text: str) -> str:
    """Code listings live in listings[]; remove from text to keep narrative tight."""
    for pat in _CODE_BLOCK_PATTERNS:
        text = pat.sub('', text)
    return text


# Theorem-like envs → "> **Type [name].** body"
_THEOREM_ENVS = (
    'definition', 'theorem', 'lemma', 'proof', 'proposition',
    'corollary', 'claim', 'remark', 'example', 'fact', 'observation',
    'conjecture', 'assumption',
)
_THEOREM_RE = re.compile(
    rf'\\begin\{{(?P<env>{"|".join(_THEOREM_ENVS)})\*?\}}'
    r'\s*(?:\[(?P<title>[^\]]*)\])?'
    r'(?P<body>.*?)'
    rf'\\end\{{(?P=env)\*?\}}',
    flags=re.DOTALL,
)


def _convert_theorem_envs(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        env = m.group('env').capitalize()
        name = m.group('title')
        head = f'**{env} ({name}).**' if name else f'**{env}.**'
        body = m.group('body').strip()
        # Indent each line with "> " to form a markdown blockquote
        quoted = '\n'.join('> ' + ln if ln else '>' for ln in body.split('\n'))
        return f'\n\n> {head}\n{quoted}\n'
    return _THEOREM_RE.sub(repl, text)


# Callout boxes → blockquote (preserve interior text losslessly).
# Includes both standard package envs AND common custom names defined via
# \newtcolorbox / \newmdenv / \newenvironment in papers' preambles. Add new
# names as they appear (residue warnings will tell you).
_CALLOUT_ENVS = (
    'mdframed', 'tcolorbox', 'tcb', 'framed', 'shaded', 'colorbox',
    # Custom names commonly used for "Finding N:" / "Remark:" / dialog boxes:
    'takeaway', 'monoquote', 'SQLVerbatim', 'verbatimcode',
    'finding', 'remark', 'note', 'highlight', 'insight', 'observation',
    'recipe', 'callout', 'infobox', 'warningbox', 'errorbox', 'examplebox',
    'promptbox', 'attackbox', 'defensebox',
)
_CALLOUT_RE = re.compile(
    rf'\\begin\{{(?P<env>{"|".join(_CALLOUT_ENVS)})\*?\}}'
    r'\s*(?:\[(?:[^\[\]]|\[[^\]]*\])*\])?'    # optional [opts]
    r'\s*(?:\{[^{}]*\})?'                      # optional {color}
    r'(?P<body>.*?)'
    rf'\\end\{{(?P=env)\*?\}}',
    flags=re.DOTALL,
)


def _convert_callout_envs(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        body = m.group('body').strip()
        quoted = '\n'.join('> ' + ln if ln else '>' for ln in body.split('\n'))
        return f'\n\n{quoted}\n'
    return _CALLOUT_RE.sub(repl, text)


# \fcolorbox{frame-color}{bg-color}{...body...} — Latin Inline form used for
# "Remark" callouts in some papers. Wraps body in a colored frame.
_FCOLORBOX_RE = re.compile(
    r'\\fcolorbox\{[^}]*\}\{[^}]*\}\{(?P<body>(?:[^{}]|\{[^{}]*\})*)\}',
    flags=re.DOTALL,
)


def _convert_fcolorbox(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        body = m.group('body').strip()
        quoted = '\n'.join('> ' + ln if ln else '>' for ln in body.split('\n'))
        return f'\n\n{quoted}\n'
    return _FCOLORBOX_RE.sub(repl, text)


# ─────────────────────────────────────────────────────────────────────────────
# Labels
# ─────────────────────────────────────────────────────────────────────────────

_LABEL_RE = re.compile(r'\\label\{[^}]+\}')


def _strip_labels(text: str) -> str:
    return _LABEL_RE.sub('', text)


# ─────────────────────────────────────────────────────────────────────────────
# Section heading commands (brace-aware, so \section{{macro_expanded}} works)
# ─────────────────────────────────────────────────────────────────────────────

_SECTION_CMD_START_RE = re.compile(r'\\(?:sub){0,2}section\*?\s*\{')


# ─────────────────────────────────────────────────────────────────────────────
# Math masking (BUG-1): the markup/noise/group rules are NOT math-aware. Inside
# `$…$` a `\mathbf{x}` must NOT become `**x**`, `\to` must not be stripped, etc.
# Mask every math span to an opaque placeholder before the pipeline runs, then
# restore verbatim at the end. Display math first ($$…$$, \[…\]) so the inline
# `$…$` pass can't mis-pair the doubled dollars.
# ─────────────────────────────────────────────────────────────────────────────

_MATH_SENTINEL = "\x00MATH{}\x00"
_MATH_SPAN_RES = [
    re.compile(r'\$\$.*?\$\$', re.DOTALL),          # $$ display $$
    re.compile(r'\\\[.*?\\\]', re.DOTALL),          # \[ display \]
    re.compile(r'\\\(.*?\\\)', re.DOTALL),          # \( inline \)
    re.compile(r'(?<!\\)\$(?:\\.|[^$\\])+?\$'),      # $ inline $ (skip \$)
]


def _mask_math(text: str) -> tuple[str, list[str]]:
    spans: list[str] = []

    def grab(m: "re.Match[str]") -> str:
        spans.append(m.group(0))
        return _MATH_SENTINEL.format(len(spans) - 1)

    for rx in _MATH_SPAN_RES:
        text = rx.sub(grab, text)
    return text, spans


def _restore_math(text: str, spans: list[str]) -> str:
    for idx, span in enumerate(spans):
        text = text.replace(_MATH_SENTINEL.format(idx), span)
    return text


def _find_matching_brace(text: str, pos: int) -> int:
    """Position of brace that matches '{' at ``pos``. Returns -1 on miss."""
    if pos >= len(text) or text[pos] != '{':
        return -1
    depth = 1
    i = pos + 1
    while i < len(text):
        c = text[i]
        if c == '\\' and i + 1 < len(text) and text[i + 1] in ('{', '}'):
            i += 2
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _strip_section_commands(text: str) -> str:
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        m = _SECTION_CMD_START_RE.match(text, i)
        if not m:
            out.append(text[i])
            i += 1
            continue
        brace_open = m.end() - 1
        close = _find_matching_brace(text, brace_open)
        if close == -1:
            out.append(text[i])
            i += 1
            continue
        i = close + 1
    return ''.join(out)


# ─────────────────────────────────────────────────────────────────────────────
# \paragraph{X} → markdown bold heading
# ─────────────────────────────────────────────────────────────────────────────

# Brace-aware (BUG-2): the old `[^{}]*` regex could not match an argument that
# itself contains a command with braces — `\paragraph{\textbf{Agents.}}` — so it
# skipped the heading and left an orphan `\paragraph` prefix once `markup` ran.
_PARAGRAPH_START_RE = re.compile(r'\\(sub)?paragraph\*?\s*\{')


def _convert_paragraph_headings(text: str) -> str:
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        m = _PARAGRAPH_START_RE.match(text, i)
        if not m:
            out.append(text[i])
            i += 1
            continue
        brace_open = m.end() - 1
        close = _find_matching_brace(text, brace_open)
        if close == -1:
            out.append(text[i])
            i += 1
            continue
        inner = _convert_markup(text[brace_open + 1:close])  # recurse: \textbf{} inside
        if m.group(1):  # \subparagraph
            out.append(f'\n*{inner}*\n')
        else:
            out.append(f'\n\n**{inner}**\n')
        i = close + 1
    return ''.join(out)


# ─────────────────────────────────────────────────────────────────────────────
# itemize / enumerate / description → markdown bullets
# ─────────────────────────────────────────────────────────────────────────────

_ITEMIZE_BEGIN_RE = re.compile(r'\\begin\{(itemize|enumerate|description)\}\s*(?:\[[^\]]*\])?')
_ITEMIZE_END_RE = re.compile(r'\\end\{(itemize|enumerate|description)\}')
_ITEM_RE = re.compile(r'\\item\b\s*(?:\[[^\]]*\])?\s*')


def _convert_lists(text: str) -> str:
    text = _ITEMIZE_BEGIN_RE.sub('\n', text)
    text = _ITEMIZE_END_RE.sub('\n', text)
    text = _ITEM_RE.sub('\n- ', text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Markup — configurable: strip OR convert to markdown.
#
# Default: markdown. Preserves the author's emphasis signal for downstream
# agents. The "strip" mode is left as an opt-in for cases where pure prose
# is preferred (e.g. embedding training).
# ─────────────────────────────────────────────────────────────────────────────

# Map LaTeX command → markdown wrappers (open, close).
# An empty tuple means "drop wrapper entirely" (keep argument text only).
_MARKUP_MARKDOWN_MAP: dict[str, tuple[str, str]] = {
    'textbf':    ('**', '**'),
    'textit':    ('*',  '*'),
    'textsl':    ('*',  '*'),
    'emph':      ('*',  '*'),
    'texttt':    ('`',  '`'),
    'underline': ('',   ''),     # markdown has no underline — drop
    'uline':     ('',   ''),
    'sout':      ('~~', '~~'),
    'textsc':    ('',   ''),     # small-caps — visual, no markdown equiv
    'textsf':    ('',   ''),
    'textrm':    ('',   ''),
    'text':      ('',   ''),     # \text{} inside math — drop wrapper
    'mathbf':    ('**', '**'),
    'mathit':    ('*',  '*'),
    'mathrm':    ('',   ''),
    'mbox':      ('',   ''),
    'hbox':      ('',   ''),
    'fbox':      ('',   ''),     # framed box — the frame is visual only
}

# A balanced-brace start pattern: matches `\cmd<spaces>{`. We then walk the
# matching close brace ourselves so nested groups like
#   \textbf{Finding \refstepcounter{counter}\thecounter:}
# survive cleanly. The previous regex used `[^{}]*` which silently failed
# on any nested `{...}` and left `\textbf` residue in the output.
_MARKUP_NAMES = '|'.join(_MARKUP_MARKDOWN_MAP.keys())
_MARKUP_START_RE = re.compile(rf'\\({_MARKUP_NAMES})\s*\{{')


def _convert_markup(text: str) -> str:
    """Convert inline markup commands to markdown wrappers.

    Walks the text linearly. For each ``\\cmd{...}`` match, uses balanced-
    brace scanning (handles nested groups) and recurses into the inner
    argument so ``\\textbf{outer \\textit{inner} mix}`` lands as
    ``**outer *inner* mix**``.

    Commands with no markdown equivalent (``\\textsc``, ``\\underline``,
    ``\\mathrm``, …) are unwrapped to the inner text — see
    `_MARKUP_MARKDOWN_MAP` for the policy per command.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        m = _MARKUP_START_RE.match(text, i)
        if not m:
            out.append(text[i])
            i += 1
            continue
        brace_open = m.end() - 1
        close = _find_matching_brace(text, brace_open)
        if close == -1:
            out.append(text[i])
            i += 1
            continue
        cmd_name = m.group(1)
        inner = text[brace_open + 1:close]
        # Recurse so nested markup of any depth gets converted too.
        inner = _convert_markup(inner)
        open_md, close_md = _MARKUP_MARKDOWN_MAP[cmd_name]
        out.append(f'{open_md}{inner}{close_md}')
        i = close + 1
    result = ''.join(out)
    # BUG-5: nested bold/italic (\textbf{\textbf{X}} or \mypar{\textbf{X}}) yields
    # redundant adjacent markers (****X****). Collapse runs to a single pair.
    result = re.sub(r'\*{4,}', '**', result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Hyperlinks (BUG-4): the section-body PIPELINE had NO handler for \href / \url /
# \hyperref (only the entity-level _denoise_inline did), so links leaked the raw
# command + URL into clean section text. Collapse to the visible text.
# ─────────────────────────────────────────────────────────────────────────────

_HYPERLINK_START_RE = re.compile(
    r'\\(href\s*\{[^{}]*\}|hyperref\s*\[[^\]]*\])\s*\{'  # \href{url}{ or \hyperref[lbl]{
)


# ─────────────────────────────────────────────────────────────────────────────
# Boxes with setup arguments — \parbox{width}{BODY}, \textcolor{color}{BODY},
# etc. Keep BODY, drop the wrapper + its setup args. Needed for text-body
# figures (fbox'd prompt/completion boxes) which are preserved as Figure.body_tex
# and denoised through this pipeline.
# ─────────────────────────────────────────────────────────────────────────────

# cmd → number of mandatory {…} setup args BEFORE the body argument.
_BOX_SPECS: dict[str, int] = {
    'parbox': 1,        # \parbox[pos]{width}{BODY}
    'makebox': 0,       # \makebox[w][p]{BODY}
    'framebox': 0,      # \framebox[w][p]{BODY}
    'raisebox': 1,      # \raisebox{lift}[h][d]{BODY}
    'colorbox': 1,      # \colorbox{color}{BODY}
    'textcolor': 1,     # \textcolor{color}{BODY}
    'scalebox': 1,      # \scalebox{factor}{BODY}
    'rotatebox': 1,     # \rotatebox{angle}{BODY}
}
_BOX_START_RE = re.compile(r'\\(' + '|'.join(_BOX_SPECS) + r')\b')


def _unwrap_boxes(text: str) -> str:
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        m = _BOX_START_RE.match(text, i)
        if not m:
            out.append(text[i])
            i += 1
            continue
        j = m.end()
        # Skip whitespace and optional [..] args interleaved anywhere.
        def _skip_opts(k: int) -> int:
            while k < n:
                while k < n and text[k] in ' \t\n':
                    k += 1
                if k < n and text[k] == '[':
                    close = text.find(']', k)
                    if close == -1:
                        return -1
                    k = close + 1
                else:
                    return k
            return k
        j = _skip_opts(j)
        ok = j != -1
        # Skip the mandatory setup args (balanced braces — widths like
        # {0.9\linewidth} contain commands).
        if ok:
            for _ in range(_BOX_SPECS[m.group(1)]):
                j = _skip_opts(j)
                if j == -1 or j >= n or text[j] != '{':
                    ok = False
                    break
                close = _find_matching_brace(text, j)
                if close == -1:
                    ok = False
                    break
                j = _skip_opts(close + 1)
                if j == -1:
                    ok = False
                    break
        if ok and j < n and text[j] == '{':
            close = _find_matching_brace(text, j)
            if close != -1:
                # Recurse so nested boxes unwrap too.
                out.append(_unwrap_boxes(text[j + 1:close]))
                i = close + 1
                continue
        # Malformed → leave the char and move on (noise pass will report it).
        out.append(text[i])
        i += 1
    return ''.join(out)


_LINEBREAK_RE = re.compile(r'\\\\\*?(?:\[[^\]]*\])?')


def _convert_linebreaks(text: str) -> str:
    """Forced line break ``\\\\`` / ``\\\\[2mm]`` → newline. Runs after math-mask
    (align separators are protected) and after block-strip (tabular row
    separators live in Table.raw_tex, never here)."""
    return _LINEBREAK_RE.sub('\n', text)


def _convert_hyperlinks(text: str) -> str:
    # \url{url} -> url (single arg, rarely nested)
    text = re.sub(r'\\url\s*\{([^{}]*)\}', r'\1', text)
    # \href{url}{text} / \hyperref[label]{text} -> text, brace-aware so the
    # visible text may contain nested groups / span lines (BUG-4 follow-up).
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        m = _HYPERLINK_START_RE.match(text, i)
        if not m:
            out.append(text[i])
            i += 1
            continue
        brace_open = m.end() - 1
        close = _find_matching_brace(text, brace_open)
        if close == -1:
            out.append(text[i])
            i += 1
            continue
        out.append(text[brace_open + 1:close])  # keep the visible text only
        i = close + 1
    return ''.join(out)


# ─────────────────────────────────────────────────────────────────────────────
# Structural noise — commands that produce no content. Replaced with space
# so neighbouring tokens stay separated.
# ─────────────────────────────────────────────────────────────────────────────

# `_END` blocks the regex from matching a longer custom command name (e.g.,
# `\noindention` should be left alone — only `\noindent` should match).
_END = r'(?![a-z])'

_NOISE_PATS = [
    r'\\maketitle' + _END,
    r'\\tableofcontents' + _END,
    r'\\bibliographystyle\{[^}]*\}',
    r'\\bibliography\{[^}]*\}',
    r'\\printbibliography' + _END,
    r'\\appendix' + _END,
    r'\\input\{[^}]*\}',
    r'\\include\{[^}]*\}',
    r'\\graphicspath\{(?:\{[^}]*\})+\}',
    r'\\centering' + _END,
    r'\\noindent' + _END,
    r'\\linebreak' + _END,
    r'\\newline' + _END,
    r'\\newpage' + _END,
    r'\\clearpage' + _END,
    r'\\smallskip' + _END,
    r'\\medskip' + _END,
    r'\\bigskip' + _END,
    r'\\par' + _END,
    r'\\quad' + _END,
    r'\\qquad' + _END,
    r'\\hfill' + _END,
    r'\\vfill' + _END,
    r'\\hspace\*?\{[^}]*\}',
    r'\\vspace\*?\{[^}]*\}',
    r'\\FloatBarrier' + _END,
    r'\\afterpage\{[^}]*\}',
    r'\\ignorespaces' + _END,
    r'\\protect' + _END,
    r'\\thispagestyle\{[^}]*\}',
    r'\\pagestyle\{[^}]*\}',
    r'\\setcounter\{[^}]*\}\{[^}]*\}',
    r'\\addtocounter\{[^}]*\}\{[^}]*\}',
    r'\\addcontentsline\{[^}]*\}\{[^}]*\}\{[^}]*\}',
    r'\\part\{[^}]*\}',
    r'\\parttoc' + _END,
    r'\\hypersetup\{(?:[^{}]|\{[^{}]*\})*\}',
    # Residue surfaced by Codex review (round 1) — common LaTeX commands that
    # don't produce useful prose content for an agent:
    r'\\color\{[^}]*\}',                      # {\color{gray} …} declaration form
    r'\\xspace' + _END,                       # xspace package — context macro
    r'\\ding\{[^}]*\}',                       # pifont symbols (★ ✓ etc.)
    r'\\setlength\{[^}]*\}\{[^}]*\}',         # layout commands
    r'\\renewcommand\{[^}]*\}\{[^}]*\}',
    r'\\providecommand\{[^}]*\}\{[^}]*\}',
    r'\\arraystretch' + _END,
    r'\\tabcolsep' + _END,
    r'\\itemsep' + _END,
    r'\\topsep' + _END,
    r'\\parsep' + _END,
    r'\\parskip' + _END,
    r'\\columnwidth' + _END,
    r'\\textwidth' + _END,
    r'\\linewidth' + _END,
    r'\\rule\{[^}]*\}\{[^}]*\}',
    # Custom callout-frame `\begin*`/`\end*` macros that authors define for
    # boxes ("takeaway", "monoquote", "SQLVerbatim", "Remark", etc.). They are
    # purely wrapper delimiters with no content; safe to drop.
    r'\\(?:begin|end)'
    r'(?:minipage|monoquote|takeaway|mybox|comment|highlight|'
    r'SQLVerbatim|verbatimcode|prompt|attack|defense|finding|remark|note|'
    r'callout|infobox|warningbox|errorbox|examplebox|insight|tcb|recipe)'
    + _END,
    r'\\gdef' + _END,
    r'\\global' + _END,
    r'\\let' + _END,
    # BUG-7: layout/structural commands that leaked into clean text as residue.
    r'\\balance' + _END,
    r'\\onecolumn' + _END,
    r'\\twocolumn' + _END,
    r'\\enddocument' + _END,
    r'\\endinput' + _END,
    r'\\clearpage' + _END,
    r'\\cleardoublepage' + _END,
    r'\\newpage' + _END,
    r'\\dotfill' + _END,
    r'\\hrulefill' + _END,
]

# Markup wrappers that pass through their inner text (so denoise unwraps them).
_INNER_MARKUP_RE = re.compile(
    r'\\(?:bm|operatorname|operatorname\*|mathcal|mathbb|mathfrak|mathrm|mathit|mathbf|'
    r'mathring|widetilde|widehat|overline|underline|ensuremath)\{([^{}]*)\}'
)
_NOISE_RE = re.compile('|'.join(_NOISE_PATS))


def _strip_noise(text: str) -> str:
    text = _NOISE_RE.sub(' ', text)
    # Unwrap math-mode markup like \bm{X} → X and \mathcal{F} → F so the agent
    # sees plain symbols. Iterate to catch nested cases.
    for _ in range(3):
        new = _INNER_MARKUP_RE.sub(r'\1', text)
        if new == text:
            break
        text = new
    return text


# Section sign before a reference marker: "\S[#sec:foo]" → "§ [#sec:foo]"
_SECTION_SIGN_RE = re.compile(r'\\S\s*(?=\[#)')


def _convert_section_sign(text: str) -> str:
    return _SECTION_SIGN_RE.sub('§ ', text)


# ─────────────────────────────────────────────────────────────────────────────
# Escape-char normalisation
# ─────────────────────────────────────────────────────────────────────────────

_ESCAPE_RES = [
    (re.compile(r'\\%'),                     '%'),
    (re.compile(r'\\&'),                     '&'),
    (re.compile(r'\\\$'),                    '$'),
    (re.compile(r'\\_'),                     '_'),
    (re.compile(r'\\#'),                     '#'),
    (re.compile(r'\\textquotesingle\b'),     "'"),
    (re.compile(r'\\textquotedblleft\b'),    '"'),
    (re.compile(r'\\textquotedblright\b'),   '"'),
    (re.compile(r'\\ldots\b'),               '…'),
    (re.compile(r'\\dots\b'),                '…'),
    (re.compile(r'\\textbackslash\b'),       r'\\\\'),
]


def _normalize_escapes(text: str) -> str:
    for pat, repl in _ESCAPE_RES:
        text = pat.sub(repl, text)
    # HTML entities sometimes leak in via Markdown-flavoured input
    return text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')


# ─────────────────────────────────────────────────────────────────────────────
# Other passes
# ─────────────────────────────────────────────────────────────────────────────

def _strip_tilde(text: str) -> str:
    return text.replace('~', ' ')


_RESIDUAL_EMPTY_RE = re.compile(r'\\[a-zA-Z]+\*?\{\}')


def _strip_residual_empty(text: str) -> str:
    return _RESIDUAL_EMPTY_RE.sub(' ', text)


_FONT_SIZE_RE = re.compile(
    r'\\(?:Large|large|huge|Huge|small|footnotesize|scriptsize|tiny|normalsize|'
    r'bf|bfseries|it|itshape|sf|sffamily|tt|ttfamily|'
    r'rm|rmfamily|em|sl|slshape|sc|scshape)\b'
)


def _strip_font_size(text: str) -> str:
    return _FONT_SIZE_RE.sub(' ', text)


def _unwrap_groups(text: str) -> str:
    """Unwrap single-element TeX groups left from macro expansion.

    ``{Crescendomation}{}`` → ``Crescendomation``. Conservative: only groups
    whose contents have no commands. Iterate to handle 2-3 levels of nesting.
    """
    for _ in range(3):
        new = re.sub(r'\{([^{}\\]+)\}', r'\1', text)
        new = re.sub(r'\{\s*\}', '', new)
        if new == text:
            break
        text = new
    return text


def _collapse_whitespace(text: str) -> str:
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' *\n *', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE — declarative list of denoise steps
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Rule:
    name: str
    description: str
    apply: Callable[[str], str]


_MATH_CARRIER: list[str] = []


def _mask_math_rule(text: str) -> str:
    # Stash math spans on a module-level carrier so the paired restore rule can
    # splice them back. denoise() is single-threaded per call, so this is safe.
    masked, spans = _mask_math(text)
    _MATH_CARRIER.clear()
    _MATH_CARRIER.extend(spans)
    return masked


def _restore_math_rule(text: str) -> str:
    return _restore_math(text, _MATH_CARRIER)


PIPELINE: list[Rule] = [
    # ── BUG-1: make math opaque to every downstream rule, restore at the end.
    Rule('math-mask',       'Stash $…$/$$…$$/\\[…\\]/\\(…\\) so markup/noise/group '
                            'rules never corrupt math (\\mathbf{x}→**x** etc.).',  _mask_math_rule),
    # ── Lossless content preservation: convert env to markdown, don't drop.
    Rule('code-listings',   'Drop lstlisting/verbatim/minted blocks from section '
                            'text (the literal code is preserved in listings[] '
                            'side table with stable id "lst:N").',                  _convert_code_listings),
    Rule('theorem-envs',    'Convert definition/theorem/lemma/proof → blockquote '
                            'with bold heading "> **Theorem 1.** body".',          _convert_theorem_envs),
    Rule('callout-envs',    'Convert mdframed/tcolorbox/framed/shaded → blockquote. '
                            'Preserves dialog frames, Remark boxes, etc.',         _convert_callout_envs),
    Rule('fcolorbox',       'Convert \\fcolorbox{}{}{body} inline callout → '
                            'blockquote. Used for inline Remark/Finding boxes.',   _convert_fcolorbox),
    # ── Lossy block strip: only for content captured in side tables already
    Rule('block-strip',     'Drop figure/table/equation/algorithm blocks '
                            '(content already captured in dedicated side tables).', _strip_blocks),
    Rule('strip-labels',    'Drop \\label{} anchors (not visible to readers).',     _strip_labels),
    Rule('section-cmd',     'Drop \\section/\\subsection commands with brace-aware '
                            'matching (titles already captured in Section.title).', _strip_section_commands),
    Rule('paragraph-head',  'Convert \\paragraph{X} → markdown **X** heading; '
                            '\\subparagraph{X} → *X*.',                             _convert_paragraph_headings),
    Rule('itemize-bullets', 'Convert itemize/enumerate/description → markdown - bullets.',
                            _convert_lists),
    Rule('hyperlinks',      'Collapse \\href{u}{t}→t, \\hyperref[l]{t}→t, \\url{u}→u '
                            '(BUG-4: section body had no link handler).',           _convert_hyperlinks),
    Rule('box-unwrap',      'Unwrap setup-arg boxes: \\parbox{w}{X}/\\textcolor{c}{X}/'
                            '\\makebox/\\raisebox/\\colorbox/\\scalebox → X. Needed '
                            'for text-body figures (Figure.body_tex).',              _unwrap_boxes),
    Rule('linebreaks',      'Convert forced line breaks \\\\ (and \\\\[2mm]) → '
                            'newline so boxed prompt text keeps its line structure.', _convert_linebreaks),
    Rule('markup',          'Convert \\textbf/\\textit/\\emph/\\texttt → markdown '
                            '(**X** / *X* / `X`) with BALANCED-brace handling so '
                            'nested groups (\\textbf{X \\refstepcounter{c}:}) survive. '
                            'See _MARKUP_MARKDOWN_MAP.',                              _convert_markup),
    Rule('noise',           'Strip structural commands with no content '
                            '(\\maketitle, \\noindent, \\par, layout commands).',   _strip_noise),
    Rule('section-sign',    'Convert \\S before a [#ref] marker → "§ [#ref]".',     _convert_section_sign),
    Rule('escape-chars',    'Normalise \\%, \\&, \\$, \\_, \\#, \\textquotesingle, '
                            '\\ldots, HTML entities to plain chars.',                _normalize_escapes),
    Rule('tilde',           'Replace LaTeX non-breaking space ~ with regular space.',
                            _strip_tilde),
    Rule('residual-empty',  'Drop residual zero-arg commands (\\chatgpt{} / '
                            '\\frameworkname{}) left by undefined macros.',         _strip_residual_empty),
    Rule('font-size',       'Strip old-style font/size declarations '
                            '(\\Large \\bf \\itshape …).',                           _strip_font_size),
    Rule('group-unwrap',    'Unwrap simple TeX groups {X} left after macro '
                            'expansion. Conservative — only no-command content.',   _unwrap_groups),
    Rule('whitespace',      'Collapse runs of spaces and blank lines.',             _collapse_whitespace),
    # ── BUG-1 pair: restore math verbatim AFTER all rules have run.
    Rule('math-restore',    'Splice masked math spans back in verbatim.',           _restore_math_rule),
]


def denoise(text: str) -> str:
    """Run every rule in PIPELINE order. Markdown markup is preserved by default."""
    for rule in PIPELINE:
        text = rule.apply(text)
    return text
