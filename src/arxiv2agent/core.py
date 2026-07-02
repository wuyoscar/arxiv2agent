"""Main orchestration: arxiv id / local folder → Paper digest.

`digest()` is the only public entry point. It returns a plain dict
representing the paper — title, sections, figures, tables, equations,
algorithms, listings, citations, footnotes — all denoised and entity-tagged.

For folder output (the recommended deliverable shape), see
``writer.write_paper_folder``.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from arxiv2agent import bib as _bib
from arxiv2agent._tex import (
    parse_section_tree,
    process_latex_source,
    SectionNode,
)

from arxiv2agent import denoise as _denoise
from arxiv2agent import extract as _extract
from arxiv2agent import markers as _markers
from arxiv2agent.schema import (
    SCHEMA_VERSION,
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


def digest(
    arxiv_id: Optional[str] = None,
    local_folder: Optional[str] = None,
    use_cache: bool = True,
    expand_macros: bool = True,
) -> dict:
    """Structure an arXiv paper into a flat JSON-ready dict.

    Exactly one of arxiv_id or local_folder must be provided.

    The download / cache path is always full-fidelity: appendix is fetched and
    parsed. Filtering of appendix is a downstream concern (the CLI's
    `--no-appendix` flag filters Paper.sections at emit time).
    """
    if (arxiv_id is None) == (local_folder is None):
        raise ValueError("Provide exactly one of arxiv_id or local_folder.")

    # Resolve the on-disk source folder (for .bib resolution and asset copy).
    if local_folder:
        source_dir = Path(local_folder)
    else:
        from arxiv2agent._tex import get_default_cache_dir
        source_dir = get_default_cache_dir() / arxiv_id

    # ── Step 1 + 2: download (cached) + flatten + denoise comments + (optional) macros
    tex = process_latex_source(
        arxiv_id=arxiv_id,
        local_folder=local_folder,
        keep_comments=False,
        remove_appendix_section=False,    # never lose appendix at fetch time
        use_cache=use_cache,
        expand_macros_flag=expand_macros,
    )
    if not tex:
        raise RuntimeError(
            f"arxiv-to-prompt failed to produce TeX for "
            f"arxiv_id={arxiv_id} local_folder={local_folder}"
        )

    # ── Metadata (with extraction provenance for future training pipelines)
    title = _extract.extract_title(tex)
    abstract_text, abstract_source = _extract.extract_abstract(tex)
    affiliations, affiliations_source = _extract.extract_affiliations(tex)
    title_source = "title_cmd" if title else "none"

    # Authors come from the abs page's citation_* meta tags — the metadata
    # arXiv itself publishes (LaTeX author blocks vary too much across
    # templates to parse honestly). Same main-site channel as the source
    # download; the result is cached next to the source, and also backfills
    # title/abstract when LaTeX extraction came up empty.
    api_meta = None
    if arxiv_id:
        from arxiv2agent.arxiv_api import fetch_arxiv_metadata
        api_meta = fetch_arxiv_metadata(arxiv_id, cache_dir=source_dir, use_cache=use_cache)
    authors = list(api_meta["authors"]) if api_meta else []
    if not title and api_meta and api_meta.get("title"):
        title, title_source = api_meta["title"], "arxiv_api"
    if not abstract_text and api_meta and api_meta.get("abstract"):
        abstract_text, abstract_source = api_meta["abstract"], "arxiv_api"

    metadata = Metadata(
        title=title,
        abstract=abstract_text,
        authors=tuple(authors),
        affiliations=tuple(affiliations),
        title_source=title_source,
        abstract_source=abstract_source,
        authors_source="arxiv_api" if authors else "none",
        affiliations_source=affiliations_source,
        arxiv_version=api_meta.get("arxiv_version") if api_meta else None,
        published=api_meta.get("published") if api_meta else None,
        updated=api_meta.get("updated") if api_meta else None,
    )

    # ── Pre-extract side tables BEFORE per-section denoise
    figures_raw = _extract.extract_figures(tex)
    tables_raw = _extract.extract_tables(tex)
    equations_raw = _extract.extract_equations(tex)
    algorithms_raw = _extract.extract_algorithms(tex)
    listings_raw = _extract.extract_listings(tex)

    # ── Section tree
    tree = parse_section_tree(tex)
    appendix_start = _extract.find_appendix_start(tex)

    sections: list[Section] = []
    footnotes_all: list[tuple[str, str]] = []
    fn_counter = 1

    # Citation aggregation across the whole paper
    cite_section_index: dict[str, list[str]] = {}   # cite_key → [section_id, ...]
    cite_count: Counter = Counter()

    # Flat walk preserving DFS order; assign hierarchical ids (sec:1, sec:1.1, ...)
    flat_nodes: list[tuple[SectionNode, list[int]]] = []   # (node, path of ordinals)

    def walk(nodes: list[SectionNode], path: list[int]) -> None:
        for i, node in enumerate(nodes, start=1):
            current = path + [i]
            flat_nodes.append((node, current))
            if node.children:
                walk(node.children, current)

    walk(tree, [])

    id_by_node: dict[int, str] = {}    # id(node) → "sec:1.2"
    for node, path in flat_nodes:
        id_by_node[id(node)] = "sec:" + ".".join(str(p) for p in path)

    # Track appendix scope by title heuristic too (some papers use
    # \section{Appendix} instead of \appendix command). Once a node is marked
    # as appendix, all its DFS-descendants and same-level siblings after it
    # are appendix too.
    appendix_by_title_start = -1
    for node, _path in flat_nodes:
        if (appendix_by_title_start < 0
                and node.level == 0
                and re.match(r'^\s*Appendi(?:x|ces)\b', node.name, re.IGNORECASE)):
            appendix_by_title_start = node.start_pos

    effective_appendix_start = appendix_start
    if appendix_by_title_start >= 0 and (
        effective_appendix_start < 0
        or appendix_by_title_start < effective_appendix_start
    ):
        effective_appendix_start = appendix_by_title_start

    for order, (node, path) in enumerate(flat_nodes):
        sec_id = id_by_node[id(node)]
        parent_id = id_by_node[id(node.parent)] if node.parent else None

        # Slice text: from after the \section{} command up to first child OR end_pos
        sec_start = node.start_pos
        sec_end = (
            node.children[0].start_pos if node.children else node.end_pos
        )
        raw_slice = tex[sec_start:sec_end]
        is_appendix = (
            effective_appendix_start >= 0 and node.start_pos >= effective_appendix_start
        )

        # 1) Markers first (preserve \cite / \ref / \footnote semantics)
        with_cite_markers, cite_keys = _markers.apply_cite_markers(raw_slice)
        with_ref_markers, ref_keys = _markers.apply_ref_markers(with_cite_markers)
        with_fn_markers, fn_pairs = _markers.apply_footnote_markers(
            with_ref_markers, start_id=fn_counter
        )
        fn_counter += len(fn_pairs)
        footnotes_all.extend(fn_pairs)

        # 2) Then denoise via PIPELINE — keeps markup as markdown so author
        # emphasis survives ("**important**" rather than "important").
        clean_text = _denoise.denoise(with_fn_markers)

        # Section titles are raw TeX as captured by parse_section_tree — run
        # the same inline-denoise pass we use on captions/abstracts so the title
        # field is agent-friendly (\\textbf{...}, \\textit{...}, \\Cref{...},
        # \\textcolor{...}{...} etc. all get unwrapped).
        cleaned_title = _extract._denoise_inline(node.name)

        sections.append(
            Section(
                id=sec_id,
                title=cleaned_title,
                level=node.level,
                parent_id=parent_id,
                order=order,
                is_appendix=is_appendix,
                text=clean_text,
                char_count=len(clean_text),
                cites=tuple(cite_keys),
                refs=tuple(ref_keys),
            )
        )

        for k in cite_keys:
            cite_section_index.setdefault(k, []).append(sec_id)
            cite_count[k] += 1

    # ── Bind figures / tables to enclosing section by position
    def _bind(record_position: int) -> Optional[str]:
        # Find the section whose [start_pos, end_pos] contains record_position
        # Walk flat_nodes for the *deepest* (highest level) matching node
        best: Optional[Section] = None
        for sec, (node, _) in zip(sections, flat_nodes):
            if node.start_pos <= record_position < node.end_pos:
                if best is None or sec.level > best.level:
                    best = sec
        return best.id if best else None

    # Build reverse-index: label → ordered, de-duplicated list of section ids
    # that reference it via [#label]. Used for figures / tables / equations /
    # algorithms alike.
    refs_index: dict[str, list[str]] = {}
    for s in sections:
        seen_per_section: set[str] = set()
        for r in s.refs:
            if r in seen_per_section:
                continue
            seen_per_section.add(r)
            refs_index.setdefault(r, []).append(s.id)

    # Appendix scope per section id — every entity inherits it from its
    # enclosing section so agents can filter main-text vs appendix content
    # without a join against sections.
    sec_is_appendix: dict[str, bool] = {s.id: s.is_appendix for s in sections}

    def _appx(position: int) -> bool:
        return sec_is_appendix.get(_bind(position) or "", False)

    # Every entity gets a canonical type-prefixed id ALWAYS — never null,
    # never a bare LaTeX label like `linear`. The agent-facing id is the
    # canonical one; the original \label{} is preserved as `latex_label`.
    figures = _build_canonical(
        figures_raw, prefix="fig", refs_index=refs_index, bind=_bind,
        build=lambda r, cid, lbl: Figure(
            id=cid, latex_label=lbl,
            caption=r["caption"], src_refs=tuple(r["src_refs"]),
            body_tex=r["body_tex"],
            text=_entity_text(
                caption=r["caption"],
                body=_denoise.denoise(r["body_tex"]) if r["body_tex"] else "",
            ),
            defined_in=_bind(r["position"]),
            is_appendix=_appx(r["position"]),
            referenced_in=_refs_for(refs_index, cid, lbl),
        ),
    )
    tables = _build_canonical(
        tables_raw, prefix="tab", refs_index=refs_index, bind=_bind,
        build=lambda r, cid, lbl: Table(
            id=cid, latex_label=lbl, env=r["env"], caption=r["caption"],
            raw_tex=r["raw_tex"], text=_entity_text(caption=r["caption"], body=r["raw_tex"]),
            tex_lines=r["tex_lines"],
            defined_in=_bind(r["position"]),
            is_appendix=_appx(r["position"]),
            referenced_in=_refs_for(refs_index, cid, lbl),
            cites_inside=tuple(r["cites_inside"]),
        ),
    )
    equations = _build_canonical(
        equations_raw, prefix="eq", refs_index=refs_index, bind=_bind,
        build=lambda r, cid, lbl: Equation(
            id=cid, latex_label=lbl, env=r["env"], raw_tex=r["raw_tex"],
            text=r["raw_tex"],
            defined_in=_bind(r["position"]),
            is_appendix=_appx(r["position"]),
            referenced_in=_refs_for(refs_index, cid, lbl),
        ),
    )
    algorithms = _build_canonical(
        algorithms_raw, prefix="alg", refs_index=refs_index, bind=_bind,
        build=lambda r, cid, lbl: Algorithm(
            id=cid, latex_label=lbl, caption=r["caption"], raw_tex=r["raw_tex"],
            text=_entity_text(caption=r["caption"], body=r["raw_tex"]),
            defined_in=_bind(r["position"]),
            is_appendix=_appx(r["position"]),
            referenced_in=_refs_for(refs_index, cid, lbl),
        ),
    )
    # Listings need code-cleanup AND line-label harvesting before building.
    listings = []
    listing_counter = 0
    for lr in listings_raw:
        listing_counter += 1
        latex_label = lr.get("id")
        cid = _canonical_id("lst", latex_label, listing_counter)
        # Strip embedded `$\label{line:foo}$` markers from code, keep them as
        # structured `line_labels` so lst-1.py is real, runnable code.
        clean_code, line_labels = _strip_line_labels(lr["code"])
        listings.append(
            Listing(
                id=cid,
                latex_label=latex_label,
                language=lr["language"],
                caption=lr["caption"],
                code=clean_code,
                text=_entity_text(caption=lr["caption"], body=clean_code),
                line_labels=tuple(line_labels),
                defined_in=_bind(lr["position"]),
                is_appendix=_appx(lr["position"]),
                referenced_in=_refs_for(refs_index, cid, latex_label),
            )
        )

    # ── Rewrite every section's `refs` field AND inline `[#ref]` markers in
    # its text to use the canonical id (not the raw LaTeX label). After this,
    # an agent never has to know two namespaces — `Table [#tab:linear]` in
    # text maps directly to the table whose `id == "tab:linear"`.
    canonical_by_label: dict[str, str] = {}
    for entity_list in (figures, tables, equations, algorithms, listings):
        for e in entity_list:
            if e.latex_label:
                canonical_by_label[e.latex_label] = e.id

    def _canonicalize_ref(text: str) -> str:
        # Replace [#raw_label] with [#canonical_id] where we have a mapping.
        return re.sub(
            r'\[#([^\]]+)\]',
            lambda m: f'[#{canonical_by_label.get(m.group(1), m.group(1))}]',
            text,
        )

    rewritten_sections: list[Section] = []
    for s in sections:
        new_refs = tuple(canonical_by_label.get(r, r) for r in s.refs)
        new_text = _canonicalize_ref(s.text)
        rewritten_sections.append(
            Section(
                id=s.id, title=s.title, level=s.level, parent_id=s.parent_id,
                order=s.order, is_appendix=s.is_appendix, text=new_text,
                char_count=len(new_text), cites=s.cites, refs=new_refs,
            )
        )
    sections = rewritten_sections

    # Citations side table. Resolve cited keys against the paper's .bib so
    # [@key] is no longer an opaque token (title / authors / year / venue).
    bib_records = _bib.resolve(source_dir, set(cite_section_index.keys()))

    citations: list[Citation] = []
    for key, sec_ids in cite_section_index.items():
        # de-dup preserving order
        seen = set()
        unique = []
        for sid in sec_ids:
            if sid not in seen:
                seen.add(sid)
                unique.append(sid)
        rec = bib_records.get(key, {})
        citations.append(
            Citation(
                key=key,
                count=cite_count[key],
                cited_in=tuple(unique),
                first_context=unique[0] if unique else None,
                title=rec.get("title"),
                authors=tuple(rec.get("authors", [])),
                year=rec.get("year"),
                venue=rec.get("venue"),
                bib_raw=rec.get("bib_raw"),
            )
        )
    citations.sort(key=lambda c: (-c.count, c.key))

    footnotes = tuple(Footnote(id=fid, text=ftxt) for fid, ftxt in footnotes_all)

    paper = Paper(
        schema_version=SCHEMA_VERSION,
        paper_id=arxiv_id or _derive_paper_id(local_folder),
        source_kind="arxiv" if arxiv_id else "local-folder",
        metadata=metadata,
        sections=tuple(sections),
        figures=tuple(figures),
        tables=tuple(tables),
        equations=tuple(equations),
        algorithms=tuple(algorithms),
        listings=tuple(listings),
        citations=tuple(citations),
        footnotes=footnotes,
    )

    paper_dict = _to_jsonable(paper)
    paper_dict["warnings"] = _scan_residue(sections)
    return paper_dict


_RESIDUE_RE = re.compile(r'\\[a-zA-Z]+')


def _scan_residue(sections) -> dict:
    """Find LaTeX commands that slipped through denoise into section text.

    Returns ``{residue_top: [{command, count}, ...], residue_section_count: N}``
    where `residue_top` lists the most frequent surviving ``\\cmd`` tokens.
    An empty `residue_top` means the prose is clean. Use this to gauge
    extraction quality per paper — useful for ML-training filtering.
    """
    # Math spans ($…$, $$…$$, \[…\], \(…\)) are intentionally PRESERVED verbatim
    # (BUG-1 fix), so the LaTeX commands inside them (\to, \mathbb, \displaystyle)
    # are content, not residue. Strip math before counting so the metric measures
    # genuine prose leakage, not preserved math.
    counter: Counter = Counter()
    sections_with_residue = 0
    for s in sections:
        prose = _MATH_SPAN_SCAN_RE.sub(' ', s.text)
        hits = _RESIDUE_RE.findall(prose)
        if hits:
            sections_with_residue += 1
            counter.update(hits)
    return {
        "residue_top": [{"command": cmd, "count": n}
                        for cmd, n in counter.most_common(20)],
        "residue_section_count": sections_with_residue,
    }


# Math spans are preserved content (see denoise math-mask). Display first so the
# inline `$…$` pass can't mis-pair doubled dollars.
_MATH_SPAN_SCAN_RE = re.compile(
    r'\$\$.*?\$\$|\\\[.*?\\\]|\\\(.*?\\\)|(?<!\\)\$(?:\\.|[^$\\])+?\$',
    re.DOTALL,
)


# Common long LaTeX label prefixes we normalize into our short typed ids.
_LABEL_PREFIX_ALIASES: dict[str, dict[str, None]] = {
    "fig": {"fig": None, "figure": None, "Figure": None, "Fig": None},
    "tab": {"tab": None, "table": None, "Table": None, "Tab": None},
    "eq":  {"eq": None, "equation": None, "Equation": None, "Eq": None},
    "alg": {"alg": None, "algorithm": None, "Algorithm": None, "Alg": None},
    "lst": {"lst": None, "listing": None, "Listing": None, "code": None},
}


def _canonical_id(prefix: str, latex_label: Optional[str], counter: int) -> str:
    """Return the canonical type-prefixed id for an entity.

    Rules (in order):
      1. No label → ``<prefix>:<counter>``.
      2. Label starts with known long alias (``figure:overview``) → strip the
         alias and use our short prefix: ``fig:overview``.
      3. Label already short-prefixed (``fig:overview``) → keep as-is.
      4. Bare label (``linear``) → prepend prefix → ``tab:linear``.
    """
    if not latex_label:
        return f"{prefix}:{counter}"
    if ":" in latex_label:
        head, tail = latex_label.split(":", 1)
        if head in _LABEL_PREFIX_ALIASES[prefix]:
            return f"{prefix}:{tail}"
        # Some other typed prefix (e.g. `app:t1` for an appendix table)
        return f"{prefix}:{latex_label}"
    return f"{prefix}:{latex_label}"


def _entity_text(caption: str = "", body: str = "") -> str:
    """Build a reader-facing text field for non-section entities."""
    parts: list[str] = []
    caption = (caption or "").strip()
    body = (body or "").strip()
    if caption:
        parts.append(f"Caption: {caption}")
    if body:
        parts.append(body)
    return "\n\n".join(parts)


def _build_canonical(records, *, prefix, refs_index, bind, build):
    """Generic builder for entity types. Assigns a canonical type-prefixed
    id to every record, preserving the original LaTeX label separately."""
    out = []
    counter = 0
    for r in records:
        counter += 1
        latex_label = r.get("id")
        cid = _canonical_id(prefix, latex_label, counter)
        out.append(build(r, cid, latex_label))
    return out


def _refs_for(refs_index, canonical_id, latex_label):
    """Both ``[#fig:overview]`` and ``[#overview]`` should resolve to the same
    figure when the LaTeX author used either form. Merge both reverse-index
    entries, de-duplicating section ids."""
    seen, out = set(), []
    for src in (canonical_id, latex_label):
        if not src:
            continue
        for sid in refs_index.get(src, []):
            if sid not in seen:
                seen.add(sid)
                out.append(sid)
    return tuple(out)


def _strip_line_labels(code: str) -> tuple[str, list[str]]:
    """Remove embedded ``$\\label{line:foo}$`` (and bare ``\\label{...}``)
    markers from a code listing body. Return (clean_code, label_list)."""
    labels: list[str] = []
    def _collect(m):
        labels.append(m.group(1))
        return ""
    # $\label{...}$ — math-wrapped variant common in lstlisting bodies
    code = re.sub(r"\$\\label\{([^}]+)\}\$", _collect, code)
    # bare \label{...}
    code = re.sub(r"\\label\{([^}]+)\}", _collect, code)
    # Tidy leftover whitespace
    code = re.sub(r"[ \t]+\n", "\n", code)
    return code, labels


def _derive_paper_id(local_folder: Optional[str]) -> str:
    """Derive paper_id from a local folder path.

    Convention:
      - If the folder is named 'arxiv-to-prompt', walk up to find the paper-slug
        ancestor (typical AgenticCarlini-style layout
        data/papers/<slug>/source/arxiv-to-prompt).
      - Otherwise, use the deepest folder name.
    """
    if not local_folder:
        return "unknown"
    import os
    parts = os.path.normpath(local_folder).split(os.sep)
    # walk up while folder name is uninformative
    skip_names = {"arxiv-to-prompt", "source", "tex", "main"}
    for name in reversed(parts):
        if name and name not in skip_names:
            return name
    return parts[-1] if parts else "unknown"


def _to_jsonable(paper: Paper) -> dict:
    """Convert nested dataclass with tuples → dict with lists for JSON."""
    d = asdict(paper)

    def _walk(obj):
        if isinstance(obj, tuple):
            return [_walk(x) for x in obj]
        if isinstance(obj, list):
            return [_walk(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _walk(v) for k, v in obj.items()}
        return obj

    return _walk(d)
