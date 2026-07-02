"""Extract figures / tables / cite keys / titles from cleaned LaTeX.

These operate on the full document text. Section-level association (which
section a figure was defined in) is done in core.py by position matching
against the section tree.
"""

from __future__ import annotations

import re
from typing import Optional

# ---------- Figures ----------

FIGURE_BLOCK_RE = re.compile(
    r'\\begin\{(figure\*?)\}(?P<body>.*?)\\end\{\1\}',
    flags=re.DOTALL,
)
INCLUDEGRAPHICS_RE = re.compile(
    r'\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}'
)
LABEL_RE = re.compile(r'\\label\{([^}]+)\}')
CAPTION_INNER_RE = re.compile(r'\\caption(?:\[[^\]]*\])?\s*\{')

# ---------- Tables ----------

TABLE_BLOCK_RE = re.compile(
    r'\\begin\{(table\*?|longtable|sidewaystable\*?|wraptable\*?)\}(?P<body>.*?)\\end\{\1\}',
    flags=re.DOTALL,
)
# IMPORTANT: this must respect nesting — tables sometimes embed inner \begin{tabular}
# inside \multirow{...}{\begin{tabular}...} cells. We extract via brace-aware
# scanning (see _extract_balanced_env) instead of plain regex.
TABULAR_ENVS = ('tabular', 'tabular*', 'tabularx', 'longtable')


# ---------- Equations ----------

EQUATION_BLOCK_RE = re.compile(
    r'\\begin\{(?P<env>equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?)\}'
    r'(?P<body>.*?)\\end\{(?P=env)\}',
    flags=re.DOTALL,
)


# ---------- Algorithms ----------

ALGORITHM_BLOCK_RE = re.compile(
    r'\\begin\{(?P<env>algorithm\*?|algorithmic\*?|algorithm2e)\}'
    r'(?P<body>.*?)\\end\{(?P=env)\}',
    flags=re.DOTALL,
)

# ---------- Cites ----------

CITE_RE = re.compile(
    r'\\[a-zA-Z]*[Cc]ite[a-zA-Z]*\*?\s*'
    r'(?:\[[^\[\]]*\]\s*){0,2}'
    r'\{([^}]+)\}'
)

# ---------- Metadata ----------

# Title command varies by template:
#   \title          standard / ACL / ACM / IEEE / Springer
#   \icmltitle      ICML
#   \mlsystitle     MLSys
#   \PaperTitle     some Springer variants
#   \Title          uncommon variants
# We match a permissive "<word>title" pattern and filter false friends
# (\titleformat / \titlespacing from titlesec, \subtitle, \shorttitle, etc.)
# in extract_title() below.
TITLE_RE = re.compile(
    r'\\([a-zA-Z]*[Tt]itle)\s*(?:\[[^\]]*\])?\s*\{',
    re.DOTALL,
)
_TITLE_EXCLUDE = {
    "titleformat", "titlespacing", "titleline", "titlerule", "titlecontents",
    "titlepage", "titlebreak", "subtitle", "shorttitle", "runningtitle",
    "icmltitlerunning", "mlsystitlerunning", "tcbtitle", "fonttitle",
    "coltitle",
}
ABSTRACT_RE = re.compile(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', re.DOTALL)
AUTHOR_RE = re.compile(r'\\author\s*(?:\[[^\]]*\])?\s*\{')


def _find_matching_brace(text: str, open_pos: int) -> int:
    """Return position of closing brace matching the '{' at open_pos. -1 if missing."""
    if open_pos >= len(text) or text[open_pos] != '{':
        return -1
    depth = 1
    i = open_pos + 1
    while i < len(text):
        c = text[i]
        # A backslash escapes the next single char: handles `\{` `\}` (literal
        # braces), `\\` (line break — its trailing char must NOT be read as a
        # brace), and `\cmd`. Skipping the pair unconditionally is correct for
        # all of these (regression: \texorpdfstring{\\}{} in titles broke matching).
        if c == '\\':
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


def _extract_balanced_arg(text: str, after_idx: int) -> Optional[str]:
    """Given an index immediately AFTER a command like \\title (which is the position
    of the optional '{' OR optional whitespace), return the balanced-brace argument."""
    i = after_idx
    while i < len(text) and text[i] in ' \t\n':
        i += 1
    if i >= len(text) or text[i] != '{':
        return None
    close = _find_matching_brace(text, i)
    if close == -1:
        return None
    return text[i + 1:close]


def _extract_caption(block: str) -> str:
    """Extract \\caption{...} content from a figure/table body, fully denoised.

    Uses the same inline-denoise pass as titles and abstracts so an agent sees
    "Overview: we generate test cases" rather than
    "\\textbf{Overview:} we generate test cases \\citep{x}\\label{fig:o}".
    """
    m = CAPTION_INNER_RE.search(block)
    if not m:
        return ""
    end = _find_matching_brace(block, m.end() - 1)
    if end == -1:
        return ""
    raw = block[m.end():end]
    return _denoise_inline(raw)


def _extract_label(block: str) -> Optional[str]:
    """Return the canonical label for a figure/table block.

    LaTeX convention says `\\label{}` should appear *after* `\\caption{}` because
    `\\caption` writes a counter to the aux file that `\\label` then anchors to.
    Some templates put a placeholder ``\\label{fig:stub}`` BEFORE the caption,
    and the real label after; the real label is the one referenced by
    ``\\ref{}`` calls in prose, so we pick the LAST `\\label` in the block.
    """
    labels = LABEL_RE.findall(block)
    return labels[-1] if labels else None


# ---------- Public ----------

def extract_figures(tex: str) -> list[dict]:
    """Return list of {position, id, caption, src_refs, body_tex}.

    ``src_refs`` holds EVERY ``\\includegraphics`` argument in the block —
    subfigure environments contribute all their images, not just the first.
    ``body_tex`` preserves non-image figure content (fbox'd prompt/completion
    text, TikZ, …) that would otherwise be lost; "" for image-only figures.
    ``position`` is for section-binding.
    """
    figures = []
    for m in FIGURE_BLOCK_RE.finditer(tex):
        body = m.group('body')
        label = _extract_label(body)
        caption = _extract_caption(body)
        src_refs = [s.strip() for s in INCLUDEGRAPHICS_RE.findall(body)]
        figures.append({
            "position": m.start(),
            "id": label,
            "caption": caption,
            "src_refs": src_refs,
            "body_tex": _figure_body_tex(body),
        })
    return figures


# Layout-only commands that don't count as "content" when deciding whether a
# figure body is worth preserving (they'd otherwise make every image-only
# figure look like it has a text body).
_FIG_LAYOUT_NOISE_RE = re.compile(
    r'\\(?:centering|small|footnotesize|scriptsize|tiny|normalsize|large|Large|'
    r'noindent|vspace\*?\{[^}]*\}|hspace\*?\{[^}]*\}|setlength\{[^}]*\}\{[^}]*\}|'
    r'captionsetup\{[^}]*\}|(?:begin|end)\{(?:center|subfigure|minipage)\}(?:\[[^\]]*\])?|'
    r'resizebox\{[^}]*\}\{[^}]*\})'
)
_MIN_BODY_CONTENT_CHARS = 30


def _figure_body_tex(body: str) -> str:
    """Return the figure body with caption / labels / includegraphics removed,
    or "" when what remains is layout-only (image figures)."""
    residual = body
    # Excise every \caption{...} with balanced braces (subcaptions included).
    while True:
        m = CAPTION_INNER_RE.search(residual)
        if not m:
            break
        end = _find_matching_brace(residual, m.end() - 1)
        if end == -1:
            break
        residual = residual[:m.start()] + residual[end + 1:]
    residual = LABEL_RE.sub('', residual)
    residual = INCLUDEGRAPHICS_RE.sub('', residual)
    residual = residual.strip()
    # Content test: drop layout commands and braces; require real text mass.
    probe = _FIG_LAYOUT_NOISE_RE.sub('', residual)
    probe = re.sub(r'[{}\[\]\s%]', '', probe)
    if len(probe) < _MIN_BODY_CONTENT_CHARS:
        return ""
    return residual


def _find_balanced_env(body: str, env_names: tuple[str, ...]) -> Optional[str]:
    """Find first occurrence of any ``\\begin{env}...\\end{env}`` in ``body`` with
    proper brace nesting so that inner ``\\begin{tabular}`` inside ``\\multirow``
    cells doesn't trick the outer close. Returns the FULL block or None."""
    for env in env_names:
        start_token = '\\begin{' + env + '}'
        end_token = '\\end{' + env + '}'
        start = body.find(start_token)
        if start < 0:
            continue
        # Scan with a depth counter for THIS env name
        depth = 1
        i = start + len(start_token)
        while i < len(body):
            nxt_begin = body.find(start_token, i)
            nxt_end = body.find(end_token, i)
            if nxt_end < 0:
                return None
            if 0 <= nxt_begin < nxt_end:
                depth += 1
                i = nxt_begin + len(start_token)
            else:
                depth -= 1
                if depth == 0:
                    return body[start:nxt_end + len(end_token)]
                i = nxt_end + len(end_token)
        return None
    return None


def extract_tables(tex: str) -> list[dict]:
    """Return list of table records with env, caption, label, raw tabular, position."""
    tables = []
    for m in TABLE_BLOCK_RE.finditer(tex):
        env = m.group(1)
        body = m.group('body')
        label = _extract_label(body)
        caption = _extract_caption(body)
        raw_tex = _find_balanced_env(body, TABULAR_ENVS) or body.strip()
        tex_lines = raw_tex.count('\n') + 1
        cites_inside = []
        for c in CITE_RE.finditer(body):
            cites_inside.extend(k.strip() for k in c.group(1).split(',') if k.strip())
        tables.append({
            "position": m.start(),
            "id": label,
            "env": env,
            "caption": caption,
            "raw_tex": raw_tex,
            "tex_lines": tex_lines,
            "cites_inside": cites_inside,
        })
    return tables


def extract_equations(tex: str) -> list[dict]:
    """Find every displayed equation/align/gather/etc. block.

    Some are unlabeled (anonymous numbered eqs); we still emit them so an agent
    can recover what was at that position, but they get ``id=None``.
    """
    equations: list[dict] = []
    for m in EQUATION_BLOCK_RE.finditer(tex):
        env = m.group('env')
        body = m.group('body')
        label = _extract_label(body)
        # Inner text without \label{}
        raw_tex = re.sub(r'\\label\{[^}]+\}', '', body).strip()
        equations.append({
            "position": m.start(),
            "id": label,
            "env": env,
            "raw_tex": raw_tex,
        })
    return equations


def extract_algorithms(tex: str) -> list[dict]:
    """Find every algorithm float block. Pulls caption + raw body."""
    algorithms: list[dict] = []
    for m in ALGORITHM_BLOCK_RE.finditer(tex):
        body = m.group('body')
        label = _extract_label(body)
        caption = _extract_caption(body)
        raw_tex = body.strip()
        algorithms.append({
            "position": m.start(),
            "id": label,
            "caption": caption,
            "raw_tex": raw_tex,
        })
    return algorithms


# ---------- Listings (code blocks) ----------

# Matches lstlisting / verbatim / minted with optional [options]{language}
_LSTLISTING_BLOCK_RE = re.compile(
    r'\\begin\{(?P<env>lstlisting|minted)\*?\}'
    r'\s*(?P<opts>\[[^\]]*\])?'
    # minted's language arg `{python}`; restricted to a real language token so a
    # lstlisting body that *starts* with `{"json": ...}` is not mistaken for one.
    r'\s*(?:\{(?P<lang_arg>[A-Za-z][A-Za-z0-9+#._-]*)\})?'
    r'(?P<body>.*?)'
    r'\\end\{(?P=env)\*?\}',
    flags=re.DOTALL,
)
_VERBATIM_BLOCK_RE = re.compile(
    r'\\begin\{verbatim\*?\}(?P<body>.*?)\\end\{verbatim\*?\}',
    flags=re.DOTALL,
)
# Match \begin{listing}[opts] ... \end{listing} — the *float* wrapper. Some
# authors put a labeled `\label{lst:foo}` here rather than inside lstlisting.
_LISTING_FLOAT_RE = re.compile(
    r'\\begin\{listing\*?\}\s*(?:\[[^\]]*\])?(?P<body>.*?)\\end\{listing\*?\}',
    flags=re.DOTALL,
)
_LANG_OPT_RE = re.compile(r'language\s*=\s*([A-Za-z0-9+#-]+)')


def extract_listings(tex: str) -> list[dict]:
    """Extract code-listing blocks. Each gets a stable id (\\label{} if present,
    else auto-assigned ``lst:N``) so an agent can address them by entity id.

    Handles:
      - ``\\begin{lstlisting}[language=Python] … \\end{lstlisting}``
      - ``\\begin{minted}{python} … \\end{minted}``
      - ``\\begin{verbatim} … \\end{verbatim}``
      - ``\\begin{listing}[t]\\begin{lstlisting}…\\end{lstlisting}\\caption{…}\\label{lst:foo}\\end{listing}``
        (float wrapper with a label on the outside — we attribute that label
        to the inner code block)
    """
    listings: list[dict] = []
    auto_counter = 0
    # Two phases: (A) lstlisting/minted with their lang opts; (B) bare verbatim.
    # We also walk \begin{listing} floats to harvest labels that live OUTSIDE
    # the lstlisting block.
    float_labels: dict[int, str] = {}    # position-of-inner-lstlisting → label
    for m in _LISTING_FLOAT_RE.finditer(tex):
        body = m.group('body')
        label = _extract_label(body)
        if label:
            inner = _LSTLISTING_BLOCK_RE.search(body) or _VERBATIM_BLOCK_RE.search(body)
            if inner:
                inner_abs_pos = m.start() + inner.start()
                float_labels[inner_abs_pos] = label

    for m in _LSTLISTING_BLOCK_RE.finditer(tex):
        body = m.group('body').strip('\n')
        # Prefer the float-wrapper label (canonical listing identity) over
        # any inner \\label{line:foo} (which anchors a specific code LINE,
        # not the listing as a whole).
        inner_label = _extract_label(body)
        if inner_label and inner_label.startswith('line:'):
            inner_label = None
        label = float_labels.get(m.start()) or inner_label
        if not label:
            auto_counter += 1
            label = f'lst:{auto_counter}'
        # Language: prefer the option [language=X], then the lang arg {X} (minted)
        opts = m.group('opts') or ''
        lang_arg = m.group('lang_arg') or ''
        lm = _LANG_OPT_RE.search(opts)
        language = (lm.group(1) if lm else lang_arg).strip().lower()
        listings.append({
            "position": m.start(),
            "id": label,
            "language": language,
            "caption": "",     # lstlisting captions live in outer \begin{listing} float
            "code": body,
        })

    for m in _VERBATIM_BLOCK_RE.finditer(tex):
        body = m.group('body').strip('\n')
        label = float_labels.get(m.start())
        if not label:
            auto_counter += 1
            label = f'lst:{auto_counter}'
        listings.append({
            "position": m.start(),
            "id": label,
            "language": "",
            "caption": "",
            "code": body,
        })

    # Now augment captions from outer listing floats
    for m in _LISTING_FLOAT_RE.finditer(tex):
        cap = _extract_caption(m.group('body'))
        if not cap:
            continue
        # Find which listing record falls within this float
        for lst in listings:
            if m.start() <= lst['position'] <= m.end():
                lst['caption'] = cap
                break

    return listings


_PLACEHOLDER_TITLES = {
    "my publication title",
    "my publication title --- single author",
    "single author title",
    "anonymous authors",
    "title",
    "untitled",
    "your title here",
}


def _clean_title(arg: str) -> str:
    """Reuse the shared inline-denoise pipeline so titles get the same level of
    cleanup as captions and abstracts (strip \\thanks, \\textcolor, font commands,
    nested groups left from macro expansion, etc.)."""
    # \thanks{...} is title-only and must be removed first because it can have
    # arbitrary nested braces that confuse the generic denoise.
    arg = re.sub(r'\\thanks\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', '', arg)
    return _denoise_inline(arg)


def extract_title(tex: str) -> str:
    """Try every title-like command, return the first plausible (non-placeholder) one.

    Handles template variants: \\title, \\icmltitle, \\mlsystitle, \\PaperTitle, ...
    Filters out:
      - titlesec package helpers (\\titleformat / \\titlespacing / ...)
      - running-title variants (\\icmltitlerunning, \\mlsystitlerunning, ...)
      - placeholder titles ("My Publication Title")
      - templates' subtitles / short titles (we only want the main one)
    """
    best = ""
    for m in TITLE_RE.finditer(tex):
        cmd_name = m.group(1).lower()
        if cmd_name in _TITLE_EXCLUDE:
            continue
        arg = _extract_balanced_arg(tex, m.end() - 1)
        if not arg:
            continue
        title = _clean_title(arg)
        if not title or len(title) < 5:
            continue
        if title.lower() in _PLACEHOLDER_TITLES:
            continue
        # Prefer the longest reasonable title (sometimes there are multiple variants;
        # the production one is usually the most elaborate).
        if len(title) > len(best):
            best = title
    return best


def _denoise_inline(text: str) -> str:
    """Lightweight cleanup for abstract / title / caption strings.

    Iteratively strips wrappers (textbf/textit/textcolor/etc), unwraps \\href /
    \\url, normalises escaped chars, converts \\cite + \\ref to markers, drops
    label anchors and decorative envs (tcolorbox/center/minipage).
    """
    # 1. Strip wrapper envs that show up around custom abstracts/captions
    text = re.sub(
        r'\\begin\{(tcolorbox|center|minipage|adjustwidth|quote|quotation|mdframed|framed)\}'
        r'\s*(?:\[(?:[^\[\]]|\[[^\]]*\])*\])?'
        r'\s*(?:\{[^{}]*\})?',
        ' ', text, flags=re.DOTALL,
    )
    text = re.sub(
        r'\\end\{(tcolorbox|center|minipage|adjustwidth|quote|quotation|mdframed|framed)\}',
        ' ', text,
    )

    # 2. Strip labels — captions / titles sometimes embed \label{} at the end
    text = re.sub(r'\\label\{[^}]+\}', '', text)

    # 3. \\, ~  → space
    text = re.sub(r'\\\\', ' ', text)
    text = text.replace('~', ' ')

    # 4. Cite + ref → pandoc markers (before stripping wrappers!)
    text = re.sub(
        r'\\[a-zA-Z]*[Cc]ite[a-zA-Z]*\{([^}]+)\}',
        lambda m: '[@' + '; @'.join(k.strip() for k in m.group(1).split(',')) + ']',
        text,
    )
    text = re.sub(
        r'\\(?:eq|auto|[Cc]|name|page|v|pretty)?ref\{([^}]+)\}',
        lambda m: '[' + '; '.join('#' + k.strip() for k in m.group(1).split(',')) + ']',
        text,
    )

    # 5. \href{url}{text} → text; \url{url} → url
    text = re.sub(r'\\href\{[^}]*\}\{([^{}]*)\}', r'\1', text)
    text = re.sub(r'\\url\{([^}]*)\}', r'\1', text)

    # 6. \textcolor{color}{text} → text (loop for nested)
    for _ in range(3):
        new = re.sub(r'\\textcolor\{[^}]*\}\{([^{}]*)\}', r'\1', text)
        if new == text:
            break
        text = new

    # 7. Markup wrappers — keep argument
    for _ in range(3):
        new = re.sub(
            r'\\(?:textit|textbf|texttt|emph|textsc|textsf|textrm|text|'
            r'underline|uline|sout|mbox|hbox|mathbf|mathit|mathrm)\{([^{}]*)\}',
            r'\1', text,
        )
        if new == text:
            break
        text = new

    # 8. \texorpdfstring{display}{pdf} → pdf-safe version
    text = re.sub(r'\\texorpdfstring\{[^{}]*\}\{([^{}]*)\}', r'\1', text)

    # 9. Escape-char normalisation (\\%  \\&  \\$  \\_  \\#  &amp;  &lt;  &gt;)
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = re.sub(r'\\%', '%', text)
    text = re.sub(r'\\&', '&', text)
    text = re.sub(r'\\\$', '$', text)
    text = re.sub(r'\\_', '_', text)
    text = re.sub(r'\\#', '#', text)
    text = re.sub(r'\\textquotesingle\b', "'", text)
    text = re.sub(r'\\textquotedblleft\b', '"', text)
    text = re.sub(r'\\textquotedblright\b', '"', text)
    text = re.sub(r'\\ldots\b|\\dots\b', '…', text)

    # 10. Drop 0-arg residual commands AFTER unwrapping
    text = re.sub(r'\\[a-zA-Z]+\*?\{\}', '', text)

    # 11. Drop old-style font/size commands that have no args (just modify rest
    #     of group). They show up as residue after macro expansion.
    text = re.sub(
        r'\\(?:Large|large|huge|Huge|small|footnotesize|scriptsize|tiny|'
        r'normalsize|bf|bfseries|it|itshape|sf|sffamily|tt|ttfamily|'
        r'rm|rmfamily|em|sl|slshape|sc|scshape)\b',
        ' ', text,
    )

    # 12. Drop \keyword[opts] command shells (e.g. \begin{...}[opts] residue)
    text = re.sub(r'\\[a-zA-Z]+\*?\s*(?:\[[^\]]*\])+', ' ', text)

    # 13. Unwrap single-element TeX groups left from macro expansion:
    #     ``{Crescendomation}{}`` → ``Crescendomation``. Conservative: only
    #     unwrap groups whose content has no commands. Iterate to handle nesting.
    for _ in range(3):
        new = re.sub(r'\{([^{}\\]+)\}', r'\1', text)
        # Also collapse empty groups
        new = re.sub(r'\{\s*\}', '', new)
        if new == text:
            break
        text = new

    # 14. Whitespace collapse
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Patterns for fallback abstract extraction.
# Matches \abstract, \setabstract, \Abstract, \PaperAbstract, \paperabstract, ...
ABSTRACT_CMD_RE = re.compile(r'\\([a-zA-Z]*[Aa]bstract)\s*\{', re.DOTALL)
_ABSTRACT_CMD_EXCLUDE = {
    "abstractname", "abstractauthor", "abstractkeywords",
    "shortabstract", "abstractheight",
}
MAKETITLE_RE = re.compile(r'\\maketitle\b')
BEGIN_DOC_RE = re.compile(r'\\begin\{document\}')
FIRST_SECTION_RE = re.compile(r'\\section\*?\{')


def extract_abstract(tex: str) -> tuple[str, str]:
    """Three-tier abstract extraction. Returns ``(text, source_tag)``.

    Tiers (in priority order):
      1. ``\\begin{abstract}...\\end{abstract}`` — standard env. Tag: ``abstract_env``.
      2. ``\\abstract{...}`` / ``\\setabstract{...}`` / ``\\PaperAbstract{...}``.
         Tag: ``abstract_cmd``.
      3. Region between ``\\maketitle`` (or ``\\begin{document}`` if no
         ``\\maketitle``) and the first ``\\section``. Heuristic-filtered.
         Tag: ``between_maketitle_section``.

    Returns ``("", "none")`` if every tier fails.
    """
    # Tier 1
    m = ABSTRACT_RE.search(tex)
    if m:
        return _denoise_inline(m.group(1).strip()), "abstract_env"

    # Tier 2
    for cmd_m in ABSTRACT_CMD_RE.finditer(tex):
        cmd_name = cmd_m.group(1).lower()
        if cmd_name in _ABSTRACT_CMD_EXCLUDE:
            continue
        close = _find_matching_brace(tex, cmd_m.end() - 1)
        if close != -1:
            body = _denoise_inline(tex[cmd_m.end():close])
            if len(body) >= 50:
                return body, "abstract_cmd"

    # Tier 3
    anchor = MAKETITLE_RE.search(tex) or BEGIN_DOC_RE.search(tex)
    sec_match = FIRST_SECTION_RE.search(tex)
    if not anchor or not sec_match or sec_match.start() <= anchor.end():
        return "", "none"
    region = tex[anchor.end():sec_match.start()]
    candidates = re.split(r'\n\s*\n', region)
    best = ""
    for chunk in candidates:
        cleaned = _denoise_inline(chunk)
        lc = cleaned.lower()
        if any(skip in lc for skip in (
            'author', 'affiliation', 'university', 'email', 'department',
            'tableofcontents', '@', 'figure', 'caption',
        )):
            continue
        if len(cleaned) >= 200 and len(cleaned) > len(best):
            best = cleaned
    return (best, "between_maketitle_section") if best else ("", "none")
    # (Unreachable — handled above.)
    return ""


# NOTE: there is deliberately NO extract_authors() here. Author extraction
# from raw LaTeX is unreliable across templates (ACL flat string, acmart
# per-author blocks, IEEEtran blockstyle, custom letterheads…) and produced
# misleading partial lists. Authors come from the arXiv metadata API instead —
# see arxiv_api.fetch_arxiv_metadata, wired up in core.digest().


def _clean_affiliation(arg: str) -> str:
    """Normalise one affiliation/institution string to a plain institution name.

    Strips footnote/superscript markers, emails, font commands, line breaks,
    and leftover braces, then collapses whitespace. Conservative: returns ''
    for anything that still looks like noise."""
    s = arg
    s = re.sub(r'(?<!\\)%.*', '', s)                    # drop trailing LaTeX comments
    s = re.sub(r'\$[^$]*\$', ' ', s)                    # any inline math: $^1$, $^{1,\ast}$
    s = re.sub(r'\\(thanks|footnote|email|orcid)\{[^{}]*\}', ' ', s)
    s = _denoise_inline(s)                              # fonts, \\, ~, href/url, labels
    s = re.sub(r'\\[A-Za-z@]+\d*', ' ', s)             # residual cmds (\IEEEauthorrefmark1, \mkntu, custom)
    s = re.sub(r'\b[\w.+-]+@[\w.-]+\.\w+\b', ' ', s)    # bare emails
    s = re.sub(r'[{}]', ' ', s)
    s = re.sub(r'\s*\\\\\s*', ', ', s)                  # line breaks → comma
    s = re.sub(r'^\s*[\d,\s]+', '', s)                  # leftover leading numbering "1 "
    s = re.sub(r'\s+', ' ', s).strip(' ,;.')           # incl. trailing period
    return s


# Institution keywords used to keep an author-block line only if it names an org
# (and to reject pure author-name lines). Word-boundaried to avoid false hits.
_INST_KW_RE = re.compile(
    r"(Universit|Universidad|Universit[àáé]|Institut|Laborator|College|Academ|"
    r"\bSchool\b|Research|Technolog|Polytechnic|Corporation|\bInc\b|\bLtd\b|\bLLC\b|"
    r"\bGmbH\b|\bLab\b|Ministry|Hospital|\bCenter\b|\bCentre\b|Foundation|"
    r"DeepMind|Google|Microsoft|\bMeta\b|Amazon|OpenAI|Anthropic|Tencent|Alibaba|"
    r"Huawei|Baidu|\bIBM\b|Nvidia|NVIDIA|Apple|Samsung|ByteDance|Salesforce|Naver|"
    r"Adobe|Intel|Cohere|Mistral|Stability|EleutherAI|Allen\s+Institute)",
    re.I,
)
# Corporate email domains → canonical institution (academic domains carry their
# name in the freetext already, so we only map well-known corporate ones).
_DOMAIN_MAP = {
    "google.com": "Google", "deepmind.com": "Google DeepMind",
    "microsoft.com": "Microsoft", "meta.com": "Meta", "fb.com": "Meta",
    "amazon.com": "Amazon", "openai.com": "OpenAI", "anthropic.com": "Anthropic",
    "tencent.com": "Tencent", "huawei.com": "Huawei", "baidu.com": "Baidu",
    "alibaba-inc.com": "Alibaba", "ibm.com": "IBM", "nvidia.com": "NVIDIA",
    "apple.com": "Apple", "samsung.com": "Samsung", "bytedance.com": "ByteDance",
    "salesforce.com": "Salesforce", "adobe.com": "Adobe", "intel.com": "Intel",
    "cohere.com": "Cohere", "naver.com": "Naver", "chinamobile.com": "China Mobile",
}
_EMAIL_DOMAIN_RE = re.compile(r"[\w.+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")
_AUTHOR_BLOCK_RE = re.compile(r"\\(?:author|icmlauthor|icmlauthorlist)\s*\{")


def _affils_from_structured(tex: str) -> tuple[list[str], str]:
    """acmart \\institution / ICML \\icmlaffiliation / llncs \\institute."""
    found: list[str] = []
    for m in re.finditer(r"\\institution\s*\{", tex):
        arg = _extract_balanced_arg(tex, m.end() - 1)
        if arg:
            found.append(arg)
    if found:
        return found, "acmart_institution"
    for m in re.finditer(r"\\icmlaffiliation\s*\{[^{}]*\}\s*\{", tex):
        arg = _extract_balanced_arg(tex, m.end() - 1)
        if arg:
            found.append(arg)
    if found:
        return found, "icml"
    for m in re.finditer(r"\\institute\s*\{", tex):
        arg = _extract_balanced_arg(tex, m.end() - 1)
        if arg:
            found.extend(re.split(r"\\and\b", arg))
    return found, ("llncs_institute" if found else "none")


def _affils_from_authorblock(tex: str) -> list[str]:
    """Free-text institutions written inside \\author{...} (after \\\\, on
    superscript-tagged lines, \\and-separated). Keeps a line only if it carries
    an institution keyword — that filter rejects the author-name lines."""
    out: list[str] = []
    for m in _AUTHOR_BLOCK_RE.finditer(tex):
        arg = _extract_balanced_arg(tex, m.end() - 1)
        if not arg:
            continue
        for part in re.split(r"\\\\|\\and\b|\\AND\b|\n", arg):
            if _INST_KW_RE.search(part):
                out.append(part)
    return out


def _affils_from_emails(tex: str) -> list[str]:
    """Corporate institutions inferred from author email domains (a robust signal
    when the layout gives only ``$^1$`` markers + an email)."""
    out: list[str] = []
    for dom in {m.group(1).lower() for m in _EMAIL_DOMAIN_RE.finditer(tex)}:
        if dom in _DOMAIN_MAP:
            out.append(_DOMAIN_MAP[dom])
    return out


def extract_affiliations(tex: str) -> tuple[list[str], str]:
    """Best-effort INSTITUTION SET (not author→affiliation mapping).

    Multi-signal, since ~0% of real papers are truly bare but only ~21% use the
    structured \\institution command. Merges, in priority order:
      1. structured template commands (acmart / ICML / llncs)
      2. free text inside \\author{...} (keyword-gated, rejects name-only lines)
      3. corporate email domains (maps @company.com → name)
    Returns (deduped institution list, '+'-joined source tag). When every signal
    is empty the work is treated as individual: (["individual"], "individual").
    """
    raw: list[str] = []
    sources: list[str] = []

    structured, ssrc = _affils_from_structured(tex)
    if structured:
        raw.extend(structured)
        sources.append(ssrc)

    block = _affils_from_authorblock(tex)
    if block:
        raw.extend(block)
        sources.append("authorblock")

    emails = _affils_from_emails(tex)
    if emails:
        raw.extend(emails)
        sources.append("email")

    out: list[str] = []
    seen: set[str] = set()
    for r in raw:
        name = _clean_affiliation(r)
        if not name or len(name) < 3 or len(name) > 150:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(name)

    if not out:
        return ["individual"], "individual"
    return out, "+".join(sources)


def find_appendix_start(tex: str) -> int:
    """Return position of the real ``\\appendix`` switch command, or -1 if absent.

    Filters out false matches like ``\\appto\\appendix{...}`` (etoolbox pattern
    that *defines* what happens at appendix time) and ``\\let\\appendix=...``.
    A real call must be at line start (optionally indented) and must NOT be
    immediately followed by ``{`` (which would mean it's being passed as an
    argument to another command).
    """
    pat = re.compile(r'(?:^|\n)\s*\\appendix\b(?!\s*\{)')
    m = pat.search(tex)
    if not m:
        return -1
    # Compute exact char position of the backslash inside the match
    return m.start() + m.group(0).index('\\appendix')
