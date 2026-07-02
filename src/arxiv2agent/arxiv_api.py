"""arXiv metadata via the MAIN-SITE abs page — authors / title / abstract / version.

Author extraction from raw LaTeX is unreliable across templates (ACL flat
strings, acmart per-author blocks, IEEEtran blockstyle, custom letterheads…);
heuristics produce misleading partial lists. Instead we read the metadata
arXiv itself publishes on every paper's ``arxiv.org/abs/<id>`` page as
Google-Scholar ``citation_*`` meta tags.

Deliberately NOT the ``export.arxiv.org`` API service: the abs page is served
by the same main-site file service that source downloads (``/e-print/``) use,
so the whole pipeline depends on exactly one arXiv channel. One extra GET per
paper, cached next to the downloaded source — repeated digests stay offline.

Dependency-free: stdlib ``urllib`` + ``re`` + ``html``.
"""

from __future__ import annotations

import html as _html
import json
import logging
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

_ABS_URL = "https://arxiv.org/abs/{arxiv_id}"
_CACHE_FILENAME = ".arxiv_api_meta.json"
_USER_AGENT = "arxiv2agent/0.5 (structured paper digests for agents)"

# arXiv asks for ≥3s between requests. Enforce it in-process so batch loops
# (10 papers in a list comprehension) don't get 429'd — cache hits skip the
# throttle entirely.
_MIN_REQUEST_INTERVAL_SECONDS = 3.0
_RETRY_STATUSES = (429, 503)
_MAX_RETRIES = 2
_last_request_at = 0.0


def _throttled_get(url: str, timeout: float) -> str:
    """GET with polite spacing and backoff on 429/503."""
    global _last_request_at
    for attempt in range(_MAX_RETRIES + 1):
        wait = _MIN_REQUEST_INTERVAL_SECONDS - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            if exc.code in _RETRY_STATUSES and attempt < _MAX_RETRIES:
                backoff = _MIN_REQUEST_INTERVAL_SECONDS * (attempt + 2)
                logging.warning(
                    f"arXiv {exc.code}; retrying in {backoff:.0f}s "
                    f"({attempt + 1}/{_MAX_RETRIES})"
                )
                time.sleep(backoff)
                continue
            raise


def fetch_arxiv_metadata(
    arxiv_id: str,
    cache_dir: Optional[Path] = None,
    use_cache: bool = True,
    timeout: float = 30.0,
) -> Optional[dict]:
    """Return {title, authors, abstract, published, updated, arxiv_version}
    for an arXiv id, or None when arXiv is unreachable / id unknown.

    When ``cache_dir`` is given the parsed metadata is cached there as
    ``.arxiv_api_meta.json`` (the same folder that holds the LaTeX source).
    """
    cache_path = Path(cache_dir) / _CACHE_FILENAME if cache_dir else None
    if use_cache and cache_path and cache_path.is_file():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass  # corrupt cache → refetch

    try:
        page = _throttled_get(_ABS_URL.format(arxiv_id=arxiv_id), timeout=timeout)
    except (OSError, urllib.error.URLError) as exc:
        logging.warning(f"arXiv abs page unreachable for {arxiv_id}: {exc}")
        return None

    meta = parse_arxiv_abs_html(page)
    if meta is None:
        logging.warning(f"arXiv abs page for {arxiv_id} had no citation metadata")
        return None
    if cache_path:
        try:
            cache_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass  # cache write failure is non-fatal
    return meta


_META_TAG_RE = re.compile(
    r'<meta\s+name="(citation_[a-z_]+)"\s+content="(.*?)"\s*/?>',
    re.DOTALL,
)
# "[v3]" markers in the Submission history block; the highest is the
# current revision the abs page (and /e-print/ download) serves.
_VERSION_RE = re.compile(r'\[v(\d+)\]')


def _clean(content: str) -> str:
    return " ".join(_html.unescape(content).split())


def _author_display(name: str) -> str:
    """citation_author is 'Last, First' → 'First Last'. Single-token names
    (collectives like 'OpenAI') pass through unchanged."""
    if "," in name:
        last, _, first = name.partition(",")
        first, last = first.strip(), last.strip()
        if first and last:
            return f"{first} {last}"
    return name.strip()


def parse_arxiv_abs_html(page: str) -> Optional[dict]:
    """Parse the citation_* meta tags of an arxiv.org/abs page. None when the
    page carries no usable metadata (unknown id / error page)."""
    tags: dict[str, list[str]] = {}
    for name, content in _META_TAG_RE.findall(page):
        tags.setdefault(name, []).append(_clean(content))

    authors = [_author_display(a) for a in tags.get("citation_author", []) if a]
    if not authors:
        return None

    def _first(name: str) -> str:
        vals = tags.get(name, [])
        return vals[0] if vals else ""

    versions = [int(v) for v in _VERSION_RE.findall(page)]
    return {
        "title": _first("citation_title"),
        "authors": authors,
        "abstract": _first("citation_abstract"),
        "published": _first("citation_date").replace("/", "-"),
        "updated": _first("citation_online_date").replace("/", "-"),
        "arxiv_version": f"v{max(versions)}" if versions else None,
    }
