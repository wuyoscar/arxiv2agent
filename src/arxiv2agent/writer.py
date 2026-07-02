"""Paper-digest folder writer.

Writes an agent-friendly directory where every section is its own markdown
file with YAML frontmatter, and every entity (figure / table / equation /
algorithm / listing) is a pair: content-file + metadata-json.

Layout
------
    out/<paper-slug>/
    ├── README.md                ← entry point, markdown TOC + links to every file
    ├── paper.json               ← machine-readable index (same info, JSON form)
    ├── sections/
    │   ├── 01-introduction.md   ← YAML frontmatter (id/title/cites/refs) + body
    │   ├── 02-background.md
    │   ├── 02.1-terminologies.md
    │   └── ...
    ├── figures/
    │   ├── fig-overview.pdf     ← real image binary (copied from source)
    │   └── fig-overview.json    ← caption, defined_in, referenced_in
    ├── tables/
    │   ├── tab-results.tex      ← raw \\begin{tabular}…\\end{tabular}
    │   └── tab-results.json
    ├── equations/
    │   ├── eq-loss.tex
    │   └── eq-loss.json
    ├── algorithms/
    │   ├── alg-tool.tex
    │   └── alg-tool.json
    ├── listings/
    │   ├── lst-1.py             ← extension follows the listing's language
    │   ├── lst-1.json
    │   ├── lst-2.txt            ← prompts go to .txt
    │   └── lst-2.json
    ├── references.json          ← citation key aggregation
    ├── footnotes.json
    └── source/                  ← original LaTeX, only when --include-source
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Optional


# Map listing language → file extension.
_LANG_EXT: dict[str, str] = {
    "python": ".py", "py": ".py",
    "javascript": ".js", "js": ".js",
    "typescript": ".ts", "ts": ".ts",
    "bash": ".sh", "shell": ".sh", "sh": ".sh",
    "json": ".json",
    "yaml": ".yaml", "yml": ".yaml",
    "xml": ".xml",
    "html": ".html",
    "css": ".css",
    "c": ".c", "cpp": ".cpp", "c++": ".cpp",
    "rust": ".rs", "go": ".go",
    "java": ".java",
    "ruby": ".rb",
    "tex": ".tex", "latex": ".tex",
    "sql": ".sql",
    "r": ".r",
    "matlab": ".m",
}


# Entity kinds we write to per-file layout (in iteration order).
ENTITY_KINDS = ("figures", "tables", "equations", "algorithms", "listings")


# ─────────────────────────────────────────────────────────────────────────────
# Filename slug helpers
# ─────────────────────────────────────────────────────────────────────────────

def _slugify(text: str, max_len: int = 60) -> str:
    """Lowercase ASCII slug: 'Hello World!' → 'hello-world'."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text[:max_len] if text else "untitled"


def _section_filename(s: dict) -> str:
    """sec:1 / "Introduction" → 01-introduction.md
       sec:2.1 / "Terminologies" → 02.1-terminologies.md"""
    raw = s["id"].split(":", 1)[1] if s["id"].startswith("sec:") else s["id"]
    parts = raw.split(".")
    padded = ".".join(p.zfill(2) for p in parts)
    return f"{padded}-{_slugify(s['title'])}.md"


def _entity_basename(prefix: str, entity_id: str) -> str:
    """fig:overview → fig-overview ; tab:Tab1 → tab-tab1"""
    raw = entity_id.split(":", 1)[1] if ":" in entity_id else entity_id
    return f"{prefix}-{_slugify(raw)}"


def _ext_for_language(language: str) -> str:
    return _LANG_EXT.get((language or "").lower(), ".txt")


# ─────────────────────────────────────────────────────────────────────────────
# YAML frontmatter
# ─────────────────────────────────────────────────────────────────────────────

def _yaml_frontmatter(meta: dict) -> str:
    """Tiny YAML emitter — handles strings, ints, bools, list[str]."""
    lines = ["---"]
    for k, v in meta.items():
        if isinstance(v, bool):
            lines.append(f"{k}: {'true' if v else 'false'}")
        elif v is None:
            lines.append(f"{k}: null")
        elif isinstance(v, (int, float)):
            lines.append(f"{k}: {v}")
        elif isinstance(v, (list, tuple)):
            if not v:
                lines.append(f"{k}: []")
            else:
                items = ", ".join(_yaml_inline(x) for x in v)
                lines.append(f"{k}: [{items}]")
        else:
            lines.append(f"{k}: {_yaml_inline(v)}")
    lines.append("---")
    return "\n".join(lines)


def _yaml_inline(v) -> str:
    if isinstance(v, str):
        if any(c in v for c in ":#\n\"'[]{}|>") or v.startswith(" ") or v.endswith(" "):
            esc = v.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{esc}"'
        return v
    return str(v)


# ─────────────────────────────────────────────────────────────────────────────
# IO helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dump_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content if content.endswith("\n") else content + "\n", encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def write_digest(
    paper: dict,
    output_dir: Optional[str | Path] = None,
    source_folder: Optional[str | Path] = None,
    include_source: bool = False,
    dest_dir: Optional[str | Path] = None,
) -> Path:
    """Write a paper digest folder.

    Args:
        paper: dict returned by ``arxiv2agent.digest(...)``.
        output_dir: parent directory; the paper lands at ``<output_dir>/<paper_id>/``.
            Provide either this or ``dest_dir``.
        source_folder: original LaTeX folder. If given, referenced figure assets
            are copied to ``figures/``. The whole tree is mirrored to ``source/``
            only when ``include_source=True``.
        include_source: when True, copy the original LaTeX tree into ``source/``
            for audit purposes. Default False — most agents don't need the
            verbatim TeX and it inflates the digest folder ~50×.
        dest_dir: write the digest directly INTO this exact directory (instead of
            ``<output_dir>/<paper_id>/``). Useful for embedding a digest at a
            fixed path like ``data/papers/<slug>/digest/``.

    Returns:
        Path to the created paper directory.
    """
    if (output_dir is None) == (dest_dir is None):
        raise ValueError("Provide exactly one of output_dir or dest_dir.")
    paper_id = paper["paper_id"]
    root = Path(dest_dir) if dest_dir is not None else Path(output_dir) / paper_id
    root.mkdir(parents=True, exist_ok=True)

    # ── Sections: per-file markdown with YAML frontmatter
    sections_dir = root / "sections"
    sections_dir.mkdir(parents=True, exist_ok=True)
    section_files: list[tuple[dict, str]] = []
    for s in paper["sections"]:
        fname = _section_filename(s)
        section_files.append((s, fname))
        frontmatter_meta = {
            "id": s["id"],
            "title": s["title"],
            "level": s["level"],
            "parent_id": s.get("parent_id"),
            "order": s["order"],
            "is_appendix": s["is_appendix"],
            "char_count": s["char_count"],
            "cites": list(s.get("cites", [])),
            "refs": list(s.get("refs", [])),
        }
        body = (
            _yaml_frontmatter(frontmatter_meta)
            + "\n\n"
            + ("#" * (s["level"] + 1)) + " " + s["title"]
            + "\n\n"
            + s["text"]
        )
        _write_text(sections_dir / fname, body)

    # ── Figure image assets FIRST so entity jsons can embed `asset_files`
    figure_assets = _copy_figure_assets(paper, root, source_folder) if source_folder else {}

    # ── Entities: per-item content + json pairs
    entity_paths = _write_entities(paper, root, figure_assets)

    # ── Aggregate small entities
    _dump_json(root / "references.json", paper["citations"])
    _dump_json(root / "footnotes.json", paper["footnotes"])

    # ── (optional) source mirror
    if source_folder and include_source:
        src_root = root / "source"
        if src_root.exists():
            shutil.rmtree(src_root)
        shutil.copytree(source_folder, src_root)

    # ── paper.json: COMPLETE machine-readable form
    # Contains everything: full sections text, full entity content (raw_tex,
    # code). This is THE canonical artifact for bulk-paper processing — one
    # paper.json per paper, same schema across all papers, single
    # `json.load()` per paper exposes the full content.
    #
    # The per-file layout written above (sections/*.md, listings/*.py, …)
    # is a derived random-access view of this same data. Keep it for human/
    # agent convenience but treat paper.json as the source of truth.
    full = {
        "schema_version": paper["schema_version"],
        "paper_id": paper_id,
        "source_kind": paper["source_kind"],
        "metadata": paper["metadata"],
        "sections":   _augment_with_file(paper["sections"], section_files, "sections", "fname"),
        "figures":    _augment_entities(paper["figures"],    entity_paths["figures"]),
        "tables":     _augment_entities(paper["tables"],     entity_paths["tables"]),
        "equations":  _augment_entities(paper["equations"],  entity_paths["equations"]),
        "algorithms": _augment_entities(paper["algorithms"], entity_paths["algorithms"]),
        "listings":   _augment_entities(paper["listings"],   entity_paths["listings"]),
        "citations":  paper["citations"],
        "footnotes":  paper["footnotes"],
        "warnings":   paper.get("warnings", {}),
    }
    _dump_json(root / "paper.json", full)

    # ── README (entry point — links + glossary)
    _write_text(root / "README.md", _render_readme(paper, section_files, entity_paths, figure_assets))

    return root


def _augment_with_file(items, section_files, _kind, _key):
    """Attach the per-file relative path to each section dict so that
    paper.json doubles as a navigation map into the per-file layout."""
    by_id = {s["id"]: fname for s, fname in section_files}
    return [{**s, "file": f"sections/{by_id[s['id']]}"} for s in items]


def _augment_entities(items, paths_by_id):
    """Attach `file` field pointing to the per-file copy when one exists."""
    out = []
    for e in items:
        e = _with_entity_text(e)
        eid = e.get("id")
        if eid and eid in paths_by_id:
            out.append({**e, "file": paths_by_id[eid]})
        else:
            out.append(dict(e))
    return out


def _with_entity_text(entity: dict) -> dict:
    """Ensure every non-section entity has a reader-facing `text` field."""
    if (entity.get("text") or "").strip():
        return dict(entity)
    caption = (entity.get("caption") or "").strip()
    body = (entity.get("raw_tex") or entity.get("code") or "").strip()
    parts: list[str] = []
    if caption:
        parts.append(f"Caption: {caption}")
    if body:
        parts.append(body)
    out = dict(entity)
    out["text"] = "\n\n".join(parts)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Entity writing
# ─────────────────────────────────────────────────────────────────────────────

def _write_entities(
    paper: dict, root: Path, figure_assets: dict[str, list[str]],
) -> dict[str, dict[str, str]]:
    """Write per-item entity files. Return {entity_type: {entity_id: relative_path}}."""
    paths: dict[str, dict[str, str]] = {k: {} for k in ENTITY_KINDS}

    # figures: image binaries are copied separately (figure_assets maps
    # fig_id → copied files); here we write metadata json + the text body
    # (fbox'd prompts / completions / TikZ) as a standalone .txt when present.
    for f in paper["figures"]:
        if not f["id"]:
            continue
        base = _entity_basename("fig", f["id"])
        meta = _with_entity_text(f)
        meta["asset_files"] = figure_assets.get(f["id"], [])
        if (f.get("body_tex") or "").strip():
            txt_name = f"{base}.txt"
            _write_text(root / "figures" / txt_name, meta["text"])
            meta["content_file"] = txt_name
        _dump_json(root / "figures" / f"{base}.json", meta)
        paths["figures"][f["id"]] = f"figures/{base}.json"

    # tables: .tex (raw_tex) + .json (metadata sans raw_tex)
    for t in paper["tables"]:
        if not t["id"]:
            continue
        base = _entity_basename("tab", t["id"])
        _write_text(root / "tables" / f"{base}.tex", t["raw_tex"])
        meta = {k: v for k, v in _with_entity_text(t).items() if k != "raw_tex"}
        meta["content_file"] = f"{base}.tex"
        _dump_json(root / "tables" / f"{base}.json", meta)
        paths["tables"][t["id"]] = f"tables/{base}.tex"

    # equations
    for e in paper["equations"]:
        if e["id"]:
            base = _entity_basename("eq", e["id"])
        else:
            base = f"eq-anon-{paper['equations'].index(e) + 1}"
        _write_text(root / "equations" / f"{base}.tex", e["raw_tex"])
        meta = {k: v for k, v in _with_entity_text(e).items() if k != "raw_tex"}
        meta["content_file"] = f"{base}.tex"
        _dump_json(root / "equations" / f"{base}.json", meta)
        if e["id"]:
            paths["equations"][e["id"]] = f"equations/{base}.tex"

    # algorithms
    for a in paper["algorithms"]:
        if a["id"]:
            base = _entity_basename("alg", a["id"])
        else:
            base = f"alg-anon-{paper['algorithms'].index(a) + 1}"
        _write_text(root / "algorithms" / f"{base}.tex", a["raw_tex"])
        meta = {k: v for k, v in _with_entity_text(a).items() if k != "raw_tex"}
        meta["content_file"] = f"{base}.tex"
        _dump_json(root / "algorithms" / f"{base}.json", meta)
        if a["id"]:
            paths["algorithms"][a["id"]] = f"algorithms/{base}.tex"

    # listings: extension follows language
    for L in paper["listings"]:
        base = _entity_basename("lst", L["id"])
        ext = _ext_for_language(L.get("language", ""))
        _write_text(root / "listings" / f"{base}{ext}", L["code"])
        meta = {k: v for k, v in _with_entity_text(L).items() if k != "code"}
        meta["content_file"] = f"{base}{ext}"
        _dump_json(root / "listings" / f"{base}.json", meta)
        paths["listings"][L["id"]] = f"listings/{base}{ext}"

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Asset copying
# ─────────────────────────────────────────────────────────────────────────────

_IMAGE_EXTS = (".pdf", ".png", ".jpg", ".jpeg", ".eps", ".svg")


def _resolve_asset(source_root: Path, ref: str) -> Optional[Path]:
    p = source_root / ref
    if p.is_file():
        return p
    if not Path(ref).suffix:
        for ext in _IMAGE_EXTS:
            cand = source_root / (ref + ext)
            if cand.is_file():
                return cand
    name = Path(ref).name
    matches = list(source_root.rglob(name))
    if matches:
        return matches[0]
    if not Path(ref).suffix:
        for ext in _IMAGE_EXTS:
            matches = list(source_root.rglob(name + ext))
            if matches:
                return matches[0]
    return None


def _copy_figure_assets(
    paper: dict, root: Path, source_folder: str | Path,
) -> dict[str, list[str]]:
    """Copy EVERY image of every figure (subfigures included) into figures/.

    Single-image figures land as ``fig-<slug>.<ext>``; multi-image figures as
    ``fig-<slug>-1.<ext>``, ``fig-<slug>-2.<ext>``, … (index = position in
    ``src_refs``, so gaps reveal unresolved refs). Returns fig_id → [relpaths].
    """
    src = Path(source_folder)
    fig_dir = root / "figures"
    copied: dict[str, list[str]] = {}
    for f in paper["figures"]:
        refs = f.get("src_refs") or []
        if not f["id"] or not refs:
            continue
        base = _entity_basename("fig", f["id"])
        multi = len(refs) > 1
        files: list[str] = []
        for idx, ref in enumerate(refs, start=1):
            resolved = _resolve_asset(src, ref)
            if resolved is None:
                continue
            stem = f"{base}-{idx}" if multi else base
            dest = fig_dir / f"{stem}{resolved.suffix}"
            fig_dir.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copyfile(resolved, dest)
                files.append(f"figures/{dest.name}")
            except (OSError, shutil.SameFileError):
                pass
        if files:
            copied[f["id"]] = files
    return copied



# ─────────────────────────────────────────────────────────────────────────────
# README rendering
# ─────────────────────────────────────────────────────────────────────────────

def _render_readme(
    paper: dict,
    section_files: list[tuple[dict, str]],
    entity_paths: dict[str, dict[str, str]],
    figure_assets: dict[str, list[str]],
) -> str:
    md = paper["metadata"]
    title = md.get("title") or paper["paper_id"]

    out: list[str] = []
    out.append(f"# {title}")
    out.append("")
    out.append(f"> Paper ID: `{paper['paper_id']}`")
    out.append(f"> Schema: v{paper['schema_version']} (arXiv digest)")
    out.append("")

    # ── Quick start
    out.append("## How to read this folder")
    out.append("")
    out.append("Every paper digest has the SAME shape across the whole corpus. Two complementary views:")
    out.append("")
    out.append("- **`paper.json`** — the canonical complete record. Contains all metadata,")
    out.append("  all sections (full text), every entity (with raw_tex / code), citations,")
    out.append("  footnotes. **Use this for bulk processing** — one `json.load()` per paper")
    out.append("  exposes everything with a fixed schema.")
    out.append("")
    out.append("- **Per-file layout** — random-access view of the same data. Use when you")
    out.append("  want one entity at a time without parsing JSON:")
    out.append("    - `sections/NN-slug.md` — one markdown file per section, YAML frontmatter")
    out.append("    - `figures/fig-*.{pdf,png,…}` (all images, subfigures too) + `figures/fig-*.txt` (text-body figures: prompts, examples) + `figures/fig-*.json`")
    out.append("    - `tables/tab-*.tex` + `tables/tab-*.json`")
    out.append("    - `equations/eq-*.tex` + `equations/eq-*.json`")
    out.append("    - `algorithms/alg-*.tex` + `algorithms/alg-*.json`")
    out.append("    - `listings/lst-*.{py,txt,…}` + `listings/lst-*.json`")
    out.append("")
    out.append("Other:")
    out.append("")
    out.append("- [`references.json`](references.json), [`footnotes.json`](footnotes.json) — small aggregates.")
    out.append("- `source/` (only when generated with `--include-source`) — the original LaTeX verbatim.")
    out.append("")

    # ── Abstract preview (first 400 chars). Full abstract lives in
    # `paper.json.metadata.abstract` — no separate abstract.md file.
    abstract = md.get("abstract") or ""
    if abstract:
        out.append("## Abstract")
        out.append("")
        preview = abstract[:400] + ("…" if len(abstract) > 400 else "")
        out.append(preview)
        out.append("")

    # ── Outline with links
    out.append("## Outline")
    out.append("")
    for s, fname in section_files:
        indent = "  " * s["level"]
        appx = " *(appendix)*" if s["is_appendix"] else ""
        out.append(f"{indent}- [`{s['id']}` — {s['title']}](sections/{fname}){appx}")
    out.append("")

    # ── Entity index
    out.append("## Entity index")
    out.append("")
    out.append("Stable IDs you can reference: `sec:*` `fig:*` `tab:*` `eq:*` `alg:*` `lst:*` `fn:*`.")
    out.append("")

    for kind, label in [
        ("figures", "Figures"), ("tables", "Tables"), ("equations", "Equations"),
        ("algorithms", "Algorithms"), ("listings", "Listings"),
    ]:
        items = paper[kind]
        if not items:
            continue
        out.append(f"### {label} ({len(items)})")
        out.append("")
        for e in items:
            eid = e.get("id") or "(no id)"
            caption = e.get("caption") or e.get("env") or ""
            cap_short = (caption[:80] + "…") if len(caption) > 80 else caption
            appx = " *(appendix)*" if e.get("is_appendix") else ""
            content_path = entity_paths[kind].get(e.get("id"))
            if content_path:
                out.append(f"- `{eid}` — {cap_short} → [{content_path}]({content_path}){appx}")
            else:
                out.append(f"- `{eid}` — {cap_short}{appx}")
        out.append("")

    if paper["citations"]:
        top = paper["citations"][:5]
        out.append(f"### Citations ({len(paper['citations'])} unique keys)")
        out.append("")
        n_resolved = sum(1 for c in paper["citations"] if c.get("title"))
        out.append(
            f"See [`references.json`](references.json) — resolved against .bib "
            f"({n_resolved}/{len(paper['citations'])} keys have title/authors/year/venue). "
            f"Top {len(top)} most-cited:"
        )
        out.append("")
        for c in top:
            meta = f" — {c['title']}" if c.get("title") else ""
            out.append(
                f"- `[@{c['key']}]` ({c['count']}× cited){meta}"
            )
        out.append("")

    if paper["footnotes"]:
        out.append(f"### Footnotes ({len(paper['footnotes'])})")
        out.append("")
        out.append("See [`footnotes.json`](footnotes.json).")
        out.append("")

    # ── Marker reference
    out.append("## Inline markers in section text")
    out.append("")
    out.append("Section markdown uses pandoc-style markers so cross-references stay agent-readable:")
    out.append("")
    out.append("| Marker         | LaTeX origin                       | Where to look it up                   |")
    out.append("|---------------|------------------------------------|---------------------------------------|")
    out.append("| `[@key]`       | `\\cite{key}` / `\\parencite{key}`  | `references.json`                     |")
    out.append("| `[#fig:foo]`   | `\\ref{fig:foo}` / `\\Cref{fig:foo}` | `figures/fig-foo.json` (or its pdf)   |")
    out.append("| `[#tab:foo]`   | same                                | `tables/tab-foo.tex`                  |")
    out.append("| `[#eq:foo]`    | same                                | `equations/eq-foo.tex`                |")
    out.append("| `[#alg:foo]`   | same                                | `algorithms/alg-foo.tex`              |")
    out.append("| `[^fn:N]`      | `\\footnote{...}`                    | `footnotes.json`                      |")
    out.append("| `**X**`        | `\\textbf{X}`                        | (inline — author emphasis)            |")
    out.append("| `*X*`          | `\\textit{X}` / `\\emph{X}`          | (inline)                              |")
    out.append("| `` `X` ``      | `\\texttt{X}`                        | (inline)                              |")
    out.append("")

    # ── Provenance
    out.append("## Provenance")
    out.append("")
    out.append(f"- `title_source`: `{md.get('title_source', 'unknown')}`")
    out.append(f"- `abstract_source`: `{md.get('abstract_source', 'unknown')}`")
    out.append(f"- `authors_source`: `{md.get('authors_source', 'unknown')}` ({len(md.get('authors') or [])} authors)")
    if md.get("arxiv_version"):
        out.append(f"- arXiv revision: `{md['arxiv_version']}` (updated {md.get('updated') or '?'})")
    out.append("")
    out.append("Possible values for `*_source`:")
    out.append("")
    out.append("- `title_cmd` / `abstract_env` / `abstract_cmd`: standard LaTeX commands or environments.")
    out.append("- `between_maketitle_section`: heuristic fallback — the text between `\\maketitle` and the first `\\section`.")
    out.append("- `none`: the paper has a structural quirk that prevented extraction.")
    out.append("")

    # ── Counts
    out.append("## Counts")
    out.append("")
    out.append(f"- sections:   {len(paper['sections'])}")
    n_fig_files = sum(len(v) for v in figure_assets.values())
    n_fig_body = sum(1 for f in paper["figures"] if (f.get("body_tex") or "").strip())
    out.append(
        f"- figures:    {len(paper['figures'])} "
        f"({n_fig_files} image files copied, {n_fig_body} with text body)"
    )
    out.append(f"- tables:     {len(paper['tables'])}")
    out.append(f"- equations:  {len(paper['equations'])}")
    out.append(f"- algorithms: {len(paper['algorithms'])}")
    out.append(f"- listings:   {len(paper['listings'])}")
    out.append(f"- citations:  {len(paper['citations'])}")
    out.append(f"- footnotes:  {len(paper['footnotes'])}")
    out.append("")

    return "\n".join(out)
