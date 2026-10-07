"""Search arXiv the way a person does in the browser: ``arxiv.org/search``.

Same main-site channel as the abs-page metadata and ``/e-print/`` downloads
(no ``export.arxiv.org``), and the same in-process politeness throttle, so a
search followed by a batch digest never bursts arXiv.

The result list is parsed from the HTML that arXiv's own search app renders
(``arXiv/arxiv-search``: ``search/templates/search/search-macros.html``).
Dependency-free: stdlib ``urllib`` + ``re`` + ``html``.
"""

from __future__ import annotations

import html as _html
import re
import urllib.parse
from datetime import datetime
from typing import Optional

from arxiv2agent.arxiv_api import _throttled_get

_SEARCH_URL = "https://arxiv.org/search/"

FIELDS = ("all", "title", "author", "abstract")
SORTS = {
    "relevance": "",
    "newest": "-announced_date_first",
    "oldest": "announced_date_first",
}
# arxiv.org/search only accepts these page sizes.
_PAGE_SIZES = (25, 50, 100, 200)


def search(
    query: str,
    max_results: int = 10,
    field: str = "all",
    sort: str = "relevance",
    timeout: float = 30.0,
) -> list[dict]:
    """Return up to ``max_results`` hits for ``query``, in arXiv's own order.

    Each hit: ``{arxiv_id, title, authors, abstract, categories, published,
    updated, url}``. ``arxiv_id`` has no version suffix, so it can be passed
    straight to ``digest()``. arXiv lists at most 25 authors per hit; the
    digest's metadata has the full list.
    """
    if field not in FIELDS:
        raise ValueError(f"field must be one of {FIELDS}, got {field!r}")
    if sort not in SORTS:
        raise ValueError(f"sort must be one of {tuple(SORTS)}, got {sort!r}")
    if not query.strip():
        raise ValueError("query is empty")

    size = next((s for s in _PAGE_SIZES if s >= max_results), _PAGE_SIZES[-1])
    hits: list[dict] = []
    start = 0
    while len(hits) < max_results:
        params = {
            "query": query,
            "searchtype": field,
            "abstracts": "show",
            "order": SORTS[sort],
            "size": size,
            "start": start,
        }
        page = _throttled_get(f"{_SEARCH_URL}?{urllib.parse.urlencode(params)}", timeout)
        batch = parse_search_html(page)
        hits.extend(batch)
        if len(batch) < size:
            break  # last page
        start += size
    return hits[:max_results]


_RESULT_RE = re.compile(r'<li class="arxiv-result">(.*?)</li>', re.DOTALL)
_ID_RE = re.compile(r'<p class="list-title[^"]*">\s*<a href="[^"]*/abs/([^"]+)"')
_VERSION_SUFFIX_RE = re.compile(r"v\d+$")
_TITLE_RE = re.compile(r'<p class="title is-5 mathjax">(.*?)</p>', re.DOTALL)
_AUTHORS_RE = re.compile(r'<p class="authors">(.*?)</p>', re.DOTALL)
_AUTHOR_LINK_RE = re.compile(r"<a [^>]*>(.*?)</a>", re.DOTALL)
_ABSTRACT_RE = re.compile(
    r'<span class="abstract-full[^"]*"[^>]*>(.*?)<a class="is-size-7"', re.DOTALL
)
_CATEGORY_RE = re.compile(r'<span class="tag is-small[^"]*"\s+data-tooltip="[^"]*">([^<]+)</span>')
_SUBMITTED_RE = re.compile(r"Submitted</span>\s*([^;<]+);")
_FIRST_SUBMITTED_RE = re.compile(r"v1</span>\s*submitted\s*([^;<]+);")
_TAG_RE = re.compile(r"<[^>]+>")


def _text(fragment: str) -> str:
    return " ".join(_html.unescape(_TAG_RE.sub("", fragment)).split())


def _iso_date(text: Optional[str]) -> str:
    """'10 March, 2024' → '2024-03-10'; unparseable → '' (never a guess)."""
    if not text:
        return ""
    try:
        return datetime.strptime(text.strip(), "%d %B, %Y").date().isoformat()
    except ValueError:
        return ""


def parse_search_html(page: str) -> list[dict]:
    """Parse one arxiv.org/search results page. No results → []."""
    hits = []
    for block in _RESULT_RE.findall(page):
        id_match = _ID_RE.search(block)
        if not id_match:
            continue
        arxiv_id = _VERSION_SUFFIX_RE.sub("", id_match.group(1))

        title = _TITLE_RE.search(block)
        authors = _AUTHORS_RE.search(block)
        abstract = _ABSTRACT_RE.search(block)
        submitted = _SUBMITTED_RE.search(block)
        first_submitted = _FIRST_SUBMITTED_RE.search(block)

        updated = _iso_date(submitted.group(1) if submitted else None)
        hits.append({
            "arxiv_id": arxiv_id,
            "title": _text(title.group(1)) if title else "",
            "authors": [_text(a) for a in _AUTHOR_LINK_RE.findall(authors.group(1))]
            if authors else [],
            "abstract": _text(abstract.group(1)) if abstract else "",
            "categories": [c.strip() for c in _CATEGORY_RE.findall(block)],
            # Only a revised paper shows a separate v1 date; otherwise the one
            # "Submitted" date is both.
            "published": _iso_date(first_submitted.group(1)) if first_submitted else updated,
            "updated": updated,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
        })
    return hits
