"""Output schema for arxiv2agent.

All dataclasses are frozen for safe JSON serialization via dataclasses.asdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

SCHEMA_VERSION = "0.5"     # 0.5: Figure.src_refs (multi-image) + Figure.body_tex
                           # (text-body figures) + is_appendix on every entity


@dataclass(frozen=True)
class Metadata:
    title: str
    abstract: str
    authors: tuple[str, ...] = field(default_factory=tuple)
    # Institution set (NOT author->affil mapping). Best-effort from structured
    # template commands only (acmart \institution, \icmlaffiliation, llncs
    # \institute); empty when the template uses custom/superscript layouts.
    affiliations: tuple[str, ...] = field(default_factory=tuple)
    # Provenance — how each field was extracted. Useful for future training
    # pipelines that want to filter/score by extraction confidence.
    title_source: str = "title_cmd"            # title_cmd | arxiv_api | none
    abstract_source: str = "none"              # abstract_env | abstract_cmd | between_maketitle_section | arxiv_api | none
    authors_source: str = "none"               # arxiv_api | none (LaTeX heuristics are unreliable — API only)
    affiliations_source: str = "none"          # acmart_institution | icml | llncs_institute | none
    # Reproducibility — which arXiv revision this digest was built from.
    arxiv_version: Optional[str] = None        # "v2" … (None for local folders / API miss)
    published: Optional[str] = None            # first arXiv submission timestamp
    updated: Optional[str] = None              # timestamp of the digested revision


@dataclass(frozen=True)
class Section:
    id: str                           # sec:1, sec:1.2, ...
    title: str
    level: int                        # 0=section, 1=subsection, 2=subsubsection
    parent_id: Optional[str]
    order: int                        # global flat order
    is_appendix: bool
    text: str                         # denoised + inline markers
    char_count: int
    cites: tuple[str, ...] = field(default_factory=tuple)
    refs: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Figure:
    id: str                           # canonical id, always type-prefixed: "fig:*"
    latex_label: Optional[str]        # original \label{} value if present (else None)
    caption: str
    src_refs: tuple[str, ...]         # ALL \includegraphics{...} values (subfigures too)
    body_tex: str                     # raw non-image figure body (fbox'd prompts,
                                      # completions, TikZ …) sans caption/label; "" when
                                      # the figure is image-only
    text: str                         # reader-facing text: caption + denoised body
    defined_in: Optional[str]         # which section the figure block was found in
    is_appendix: bool = False         # True when defined_in is an appendix section
    referenced_in: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Table:
    id: str                           # canonical id, always "tab:*"
    latex_label: Optional[str]
    env: str                          # table | table* | longtable | sidewaystable | wraptable
    caption: str
    raw_tex: str                      # \begin{tabular}...\end{tabular} only, no outer styling
    text: str                         # caption + raw_tex for non-random-access readers
    tex_lines: int
    defined_in: Optional[str]
    is_appendix: bool = False
    referenced_in: tuple[str, ...] = field(default_factory=tuple)
    cites_inside: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Citation:
    key: str
    count: int                        # times cited across the paper
    cited_in: tuple[str, ...]         # section ids
    first_context: Optional[str]      # section id of first appearance
    # Resolved from the paper's .bib (None / empty when unresolved):
    title: Optional[str] = None
    authors: tuple[str, ...] = ()
    year: Optional[str] = None
    venue: Optional[str] = None       # booktitle / journal / publisher / …
    bib_raw: Optional[str] = None     # the raw BibTeX entry, verbatim


@dataclass(frozen=True)
class Equation:
    id: str                           # canonical id, always "eq:*" (auto-numbered if unlabeled)
    latex_label: Optional[str]
    env: str                          # equation | align | gather | multline | eqnarray
    raw_tex: str
    text: str                         # raw LaTeX equation body
    defined_in: Optional[str]
    is_appendix: bool = False
    referenced_in: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Algorithm:
    id: str                           # canonical id, always "alg:*"
    latex_label: Optional[str]
    caption: str
    raw_tex: str
    text: str                         # caption + raw_tex for non-random-access readers
    defined_in: Optional[str]
    is_appendix: bool = False
    referenced_in: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Listing:
    id: str                           # canonical id, always "lst:*"
    latex_label: Optional[str]
    language: str                     # 'python' | 'tex' | '' (unspecified)
    caption: str
    code: str                         # literal code, no LaTeX \label residue
    text: str                         # caption + code for non-random-access readers
    defined_in: Optional[str]
    is_appendix: bool = False
    line_labels: tuple[str, ...] = field(default_factory=tuple)   # \label{line:foo} embedded inside the code
    referenced_in: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Footnote:
    id: str                           # fn:1, fn:2, ...
    text: str


@dataclass(frozen=True)
class Paper:
    schema_version: str
    paper_id: str
    source_kind: str                  # "arxiv" | "local-folder"
    metadata: Metadata
    sections: tuple[Section, ...]
    figures: tuple[Figure, ...]
    tables: tuple[Table, ...]
    equations: tuple[Equation, ...]
    algorithms: tuple[Algorithm, ...]
    listings: tuple[Listing, ...]
    citations: tuple[Citation, ...]
    footnotes: tuple[Footnote, ...]
