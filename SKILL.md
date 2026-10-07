---
name: arxiv2agent
description: >
  Find arXiv papers by title and download them as LaTeX source (raw tree +
  flattened .tex) or as a structured digest (paper.json + per-section markdown
  + tables/equations/figures as files). Use whenever the user asks to read,
  quote, compare, or answer questions about a paper, or wants to see how a
  paper typesets something (tables, figures, macros, packages). Do NOT answer
  paper-content questions from search snippets; download the paper and read it.
---

# arxiv2agent

## Setup (once)

```bash
arxiv2agent --help || uv tool install git+https://github.com/wuyoscar/arxiv2agent
```

## 1. Find the arXiv ID

```bash
arxiv2agent find "Attention Is All You Need"          # TSV: id  year  title  first-author
arxiv2agent find "Attention Is All You Need" --json   # [{id, title, authors, year}]
```

Pick the right candidate yourself; `find` never auto-selects. If it prints
nothing (exit 1), the venue title may differ from the arXiv title: do ONE web
search for `site:arxiv.org <title>` and take the ID from the URL. Still nothing?
Tell the user the paper isn't on arXiv. Don't fall back to answering from snippets.

## 2. Download

| need | command | you get |
|---|---|---|
| how something is typeset (table, figure, macro, package) | `arxiv2agent fetch ID -o papers/` | `papers/ID/source/` (untouched tree: `.tex`, `.sty`, `.bib`, figures) and `papers/ID/ID.tex` (`\input` expanded, comments kept) |
| read / quote / answer questions / compare | `arxiv2agent digest ID -o papers/` | structured folder, see below |

- Several papers: pass them as ONE list (`arxiv2agent digest ID1 ID2 ID3 -o papers/`). Requests are rate-limited internally (≥3s apart). Don't parallelize, and don't run one command per paper. A failing ID doesn't abort the batch.
- Everything is cached; re-running is cheap and offline.
- `fetch` and `digest` can write to the same `papers/ID/` folder.

## 3. Read the digest

| you need | read |
|---|---|
| overview / navigation | `README.md` (outline + entity index) |
| one section | `sections/NN-slug.md` |
| everything, in bulk | `paper.json`, fixed schema across all papers |
| a figure | `figures/fig-<slug>.json` + original image files; text-body figures (prompt boxes) in `figures/fig-<slug>.txt` |
| table / equation / algorithm | `tables/tab-*.tex`, `equations/eq-*.tex`, `algorithms/alg-*.tex` (raw LaTeX) |
| code from the paper | `listings/lst-*.py` |
| what `[@key]` cites | `references.json` (BibTeX in `bib_raw`, `cited_in` sections) |
| footnote `[^fn:N]` | `footnotes.json` |

Inline markers in section text: `[@key]` = citation, `[#fig:x]` / `[#tab:x]` / `[#eq:x]` = entity (matches `id` in `paper.json`), `[^fn:N]` = footnote.

Entity ID prefixes: `sec:` `fig:` `tab:` `eq:` `alg:` `lst:` `fn:`; unlabeled entities get auto-numbered (`eq:1`), never null. The original `\label{}` is kept as `latex_label`.

**Read the whole paper.** Don't stop at the Abstract; all sections are local. `is_appendix` (on sections and entities) separates main-text evidence from appendix evidence.

**Honesty fields.** `metadata.*_source` says how each field was obtained (`arxiv_api`, `title_cmd`, `none`, …). A citation with `title: null` means the paper shipped no `.bib`: quote `bib_raw` / the key, don't guess. `warnings.residue_top` lists LaTeX that survived cleaning; mention it if you quote an affected section.

## Many papers: write one script

```python
import json
from pathlib import Path

def load(pid, root="papers"):
    return json.loads(Path(root, pid, "paper.json").read_text())

ids = ["2305.13860", "1706.03762", "2005.14165"]
intros = {pid: [s["text"] for s in load(pid)["sections"]
                if "introduction" in s["title"].lower()] for pid in ids}
cited_in = {c["key"]: c["cited_in"] for c in load("2305.13860")["citations"]}
```
