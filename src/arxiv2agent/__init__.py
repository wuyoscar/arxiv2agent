"""arxiv2agent — find, fetch, and digest arXiv papers for agents."""

from arxiv2agent.core import digest
from arxiv2agent.fetch import fetch
from arxiv2agent.schema import (
    Algorithm,
    Citation,
    Equation,
    Figure,
    Footnote,
    Listing,
    Metadata,
    Paper,
    Section,
    Table,
)
from arxiv2agent.search import find
from arxiv2agent.writer import write_digest

__version__ = "0.6.0"
__all__ = [
    "find",
    "fetch",
    "digest",
    "write_digest",
    "Paper",
    "Section",
    "Figure",
    "Table",
    "Equation",
    "Algorithm",
    "Listing",
    "Citation",
    "Footnote",
    "Metadata",
]
