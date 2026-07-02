"""Minimal, dependency-free BibTeX (.bib) parser.

Resolves the opaque citation keys we already extracted (`[@key]`) into
human/agent-readable metadata: title, authors, year, venue.

Why in-house (no `bibtexparser` dependency)
-------------------------------------------
The corpus is ~85% standard `@type{key, field={value}, ...}` BibTeX. A small
brace-aware scanner handles that cleanly while keeping the tool's dependency
list at just `requests` + `filelock`. Anything we can't parse falls back to a
raw string, so we never lose information — we just don't structure it.

We deliberately ignore `.bbl` (compiled bibliography). It is bibstyle-specific
spaghetti (natbib / biblatex / acmart / IEEEtran all differ) and only adds
~10% coverage over `.bib`. Not worth the complexity. See round-7 design notes.
"""

from __future__ import annotations

import re
from pathlib import Path

# Entry header: @inproceedings{   /  @article {
_ENTRY_HEAD_RE = re.compile(r'@(\w+)\s*\{', re.IGNORECASE)
# Field name = ...
_FIELD_NAME_RE = re.compile(r'\s*([A-Za-z][A-Za-z0-9_\-]*)\s*=\s*')

# Field that names the venue, in priority order.
_VENUE_FIELDS = ("booktitle", "journal", "conference", "proceedings",
                 "publisher", "school", "institution", "howpublished")

# Non-data @-blocks we skip.
_SKIP_TYPES = {"comment", "preamble", "string"}


def find_bib_files(source_dir: Path) -> list[Path]:
    """All .bib files under source_dir (recursive), largest first.

    Largest first because the main bibliography is usually the biggest .bib;
    template stubs (e.g. acmart.bib) are tiny.
    """
    bibs = sorted(source_dir.rglob("*.bib"), key=lambda p: p.stat().st_size, reverse=True)
    return bibs


def _match_brace(text: str, open_pos: int) -> int:
    """Index of the '}' matching the '{' at open_pos, or -1."""
    if open_pos >= len(text) or text[open_pos] != "{":
        return -1
    depth = 1
    i = open_pos + 1
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _read_value(text: str, pos: int) -> tuple[str, int]:
    """Read a BibTeX field value starting at pos. Returns (value, next_pos).

    Handles brace-delimited {…} (nested ok), quote-delimited "…", and bare
    tokens (numbers / single words like a year or a @string macro name).
    """
    n = len(text)
    while pos < n and text[pos] in " \t\r\n":
        pos += 1
    if pos >= n:
        return "", pos
    ch = text[pos]
    if ch == "{":
        close = _match_brace(text, pos)
        if close == -1:
            return text[pos + 1:].strip(), n
        return text[pos + 1:close].strip(), close + 1
    if ch == '"':
        # Quote-delimited; respect nested braces inside quotes.
        i = pos + 1
        depth = 0
        while i < n:
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            elif c == '"' and depth == 0:
                return text[pos + 1:i].strip(), i + 1
            i += 1
        return text[pos + 1:].strip(), n
    # Bare token up to , or }
    m = re.match(r'[^,}]*', text[pos:])
    val = m.group(0).strip()
    return val, pos + m.end()


def _parse_fields(body: str) -> dict[str, str]:
    """Parse `name = value, name = value, ...` from an entry body (after key)."""
    fields: dict[str, str] = {}
    pos = 0
    n = len(body)
    while pos < n:
        m = _FIELD_NAME_RE.match(body, pos)
        if not m:
            # advance to next comma and retry
            comma = body.find(",", pos)
            if comma == -1:
                break
            pos = comma + 1
            continue
        name = m.group(1).lower()
        value, pos = _read_value(body, m.end())
        fields[name] = value
        # skip trailing comma/space
        while pos < n and body[pos] in " \t\r\n,":
            pos += 1
    return fields


def parse_bib(text: str) -> dict[str, dict]:
    """Parse a .bib string → {key: {type, fields, raw}}. Brace-aware."""
    entries: dict[str, dict] = {}
    i = 0
    n = len(text)
    while i < n:
        at = text.find("@", i)
        if at == -1:
            break
        m = _ENTRY_HEAD_RE.match(text, at)
        if not m:
            i = at + 1
            continue
        etype = m.group(1).lower()
        brace_open = m.end() - 1
        brace_close = _match_brace(text, brace_open)
        if brace_close == -1:
            i = at + 1
            continue
        if etype in _SKIP_TYPES:
            i = brace_close + 1
            continue
        body = text[brace_open + 1:brace_close]
        key, _, rest = body.partition(",")
        key = key.strip()
        if key:
            entries[key] = {
                "type": etype,
                "fields": _parse_fields(rest),
                "raw": text[at:brace_close + 1],
            }
        i = brace_close + 1
    return entries


# ── Light LaTeX cleanup for field values (titles often have braces / commands)

_LATEX_CMD_RE = re.compile(r'\\[a-zA-Z]+\*?')
_BRACES_RE = re.compile(r'[{}]')


def _clean_value(v: str) -> str:
    v = _LATEX_CMD_RE.sub("", v)
    v = _BRACES_RE.sub("", v)
    v = re.sub(r'\s+', " ", v).strip().rstrip(".,")
    return v


def _split_authors(author_field: str) -> list[str]:
    """BibTeX joins authors with ' and '. Normalize 'Last, First' → 'First Last'."""
    out: list[str] = []
    for raw in re.split(r'\s+and\s+', author_field):
        name = _clean_value(raw)
        if not name:
            continue
        if "," in name:
            last, _, first = name.partition(",")
            name = f"{first.strip()} {last.strip()}".strip()
        out.append(name)
    return out


def resolve(source_dir: Path, cited_keys: set[str]) -> dict[str, dict]:
    """Resolve cited keys against the paper's .bib files.

    Returns {key: {title, authors, year, venue, bib_raw}} for keys we found.
    Only keys in `cited_keys` are returned — shared .bib databases often hold
    hundreds of uncited entries we don't want to emit.
    """
    merged: dict[str, dict] = {}
    for bib_path in find_bib_files(source_dir):
        try:
            text = bib_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for key, rec in parse_bib(text).items():
            if key not in cited_keys or key in merged:
                continue
            f = rec["fields"]
            venue = next((f[k] for k in _VENUE_FIELDS if f.get(k)), None)
            merged[key] = {
                "title": _clean_value(f["title"]) if f.get("title") else None,
                "authors": _split_authors(f["author"]) if f.get("author") else [],
                "year": _clean_value(f["year"]) if f.get("year") else None,
                "venue": _clean_value(venue) if venue else None,
                "bib_raw": rec["raw"],
            }
    return merged
