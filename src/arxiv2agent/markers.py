"""Inline marker conversion: LaTeX citation/ref/footnote → pandoc-style markers.

These run BEFORE denoise (which would strip raw \\cite / \\ref otherwise).
"""

from __future__ import annotations

import re

# BibLaTeX / natbib citation commands often carry optional [pre]/[post] args:
#   \citep[e.g.,][p.10]{key}    \cite[Foo;]{a,b}    \citet*{c,d}
# Pattern: \<cmd> + optional `*` + 0..2 [bracketed args] + {keys}.
CITE_RE = re.compile(
    r'\\[a-zA-Z]*[Cc]ite[a-zA-Z]*\*?\s*'
    r'(?:\[[^\[\]]*\]\s*){0,2}'
    r'\{([^}]+)\}'
)
# Covers \ref, \eqref, \autoref, \cref, \Cref, \nameref, \pageref, \vref, \prettyref.
# `\Cref` is cleveref (capital). All map to the same marker.
REF_RE = re.compile(r'\\(?:eq|auto|[Cc]|name|page|v|pretty)?ref\{([^}]+)\}')
# Footnote with single-level braces (nested braces not handled — known limitation v0.1)
FOOTNOTE_RE = re.compile(r'\\footnote\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}')


def apply_cite_markers(text: str) -> tuple[str, list[str]]:
    """Replace \\cite[*]{keys} with [@k1; @k2; ...]. Return (new_text, ordered keys)."""
    keys_ordered: list[str] = []

    def repl(m: re.Match) -> str:
        keys = [k.strip() for k in m.group(1).split(',') if k.strip()]
        keys_ordered.extend(keys)
        return '[' + '; '.join(f'@{k}' for k in keys) + ']'

    return CITE_RE.sub(repl, text), keys_ordered


def apply_ref_markers(text: str) -> tuple[str, list[str]]:
    """Replace ``\\ref{x}`` / ``\\eqref{x}`` / ``\\autoref{x}`` / ``\\cref{a,b}`` /
    ``\\Cref{x}`` / ``\\nameref{x}`` / ``\\pageref{x}`` / ``\\vref{x}`` with
    bracketed ``[#x]`` markers.

    Multi-target cleveref forms (``\\Cref{a,b,c}``) expand to ``[#a; #b; #c]``.
    """
    refs_ordered: list[str] = []

    def repl(m: re.Match) -> str:
        keys = [k.strip() for k in m.group(1).split(',') if k.strip()]
        refs_ordered.extend(keys)
        return '[' + '; '.join(f'#{k}' for k in keys) + ']'

    return REF_RE.sub(repl, text), refs_ordered


def apply_footnote_markers(text: str, start_id: int = 1) -> tuple[str, list[tuple[str, str]]]:
    """Replace \\footnote{content} with [^fn:N] and return collected footnotes.

    Returns:
        (text with markers, list of (footnote_id, content))
    """
    counter = [start_id]
    collected: list[tuple[str, str]] = []

    def repl(m: re.Match) -> str:
        fn_id = f'fn:{counter[0]}'
        counter[0] += 1
        collected.append((fn_id, m.group(1).strip()))
        return f'[^{fn_id}]'

    return FOOTNOTE_RE.sub(repl, text), collected
