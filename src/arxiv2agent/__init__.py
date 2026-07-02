"""arxiv2agent — turn an arXiv paper into an agent-friendly digest folder."""

from arxiv2agent.core import digest
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
from arxiv2agent.writer import write_digest

__version__ = "0.4.0"
__all__ = [
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
