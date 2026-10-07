"""Title → arXiv ID lookup via the arXiv API (export.arxiv.org/api/query).

Lists candidates only; the caller (usually an agent) picks the right ID.
Shares the polite 3s throttle with the abs-page metadata fetch.
"""

from __future__ import annotations

import re
import urllib.parse
import xml.etree.ElementTree as ET

from arxiv2agent.arxiv_api import _throttled_get

_API_URL = "https://export.arxiv.org/api/query?{query}"
_ATOM = "{http://www.w3.org/2005/Atom}"
_ID_RE = re.compile(r"arxiv\.org/abs/(.+?)(v\d+)?$")


def _words(title: str) -> list[str]:
    # ti: matching is token-based; punctuation (colons, hyphens) breaks phrase queries.
    return re.findall(r"[A-Za-z0-9]+", title)


def parse_atom(xml_text: str) -> list[dict]:
    """Parse an arXiv API Atom feed into [{id, title, authors, year}]."""
    root = ET.fromstring(xml_text)
    results = []
    for entry in root.iter(f"{_ATOM}entry"):
        m = _ID_RE.search(entry.findtext(f"{_ATOM}id", ""))
        if not m:
            continue
        results.append({
            "id": m.group(1),
            "title": " ".join(entry.findtext(f"{_ATOM}title", "").split()),
            "authors": [a.findtext(f"{_ATOM}name", "") for a in entry.iter(f"{_ATOM}author")],
            "year": entry.findtext(f"{_ATOM}published", "")[:4],
        })
    return results


def find(title: str, max_results: int = 5, timeout: float = 30.0) -> list[dict]:
    """Search arXiv by title. Tries the exact phrase first, then all words."""
    words = _words(title)
    if not words:
        return []
    queries = [f'ti:"{" ".join(words)}"', " AND ".join(f"ti:{w}" for w in words)]
    for q in queries:
        url = _API_URL.format(query=urllib.parse.urlencode(
            {"search_query": q, "max_results": max_results}
        ))
        results = parse_atom(_throttled_get(url, timeout=timeout))
        if results:
            return results
    return []
