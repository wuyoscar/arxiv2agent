---
name: arxiv2agent
description: >
  Digest arXiv papers into structured local folders (paper.json + per-section
  markdown + figures/tables/equations/listings as real files), then read and
  cross-check them programmatically. Use whenever the user asks to read,
  compare, extract from, or verify claims in arXiv papers — especially for
  multiple papers at once. Do NOT web-search a paper's content when its
  arXiv ID is known; digest it and read the structured files instead.
---

# arxiv2agent — structured paper reading for agents

## Setup (once)

Check availability first; install only if missing:

```bash
arxiv2agent --help || (git clone https://github.com/wuyoscar/arxiv2agent /tmp/arxiv2agent \
  && cd /tmp/arxiv2agent && uv tool install .)
```

## Digest papers

```bash
arxiv2agent 2305.13860 -o corpus/                          # one paper
arxiv2agent 2305.13860 1706.03762 2005.14165 -o corpus/    # many: pass a LIST of IDs
```

- For multiple papers ALWAYS pass them as a list to ONE command (sequential + rate-limited internally; one failure doesn't abort the batch). Do NOT run one command per paper, and do NOT parallelize.
- Cheap to re-run: sources and metadata are cached locally; a cached paper digests in ~0.2s offline. 10 new papers ≈ 40s end to end.
- No arXiv ID? Resolve title → ID first (one web search: `site:arxiv.org <title>`), then digest. Never answer content questions from search snippets.

## Read the digest — rules

The digest at `corpus/<id>/` is the full paper. Read files, don't search:

| you need                  | read                                                    |
|---------------------------|---------------------------------------------------------|
| overview / navigation     | `README.md` (outline + entity index with stable IDs)    |
| one section               | `sections/NN-slug.md` (YAML frontmatter + markdown)     |
| everything, in bulk       | `paper.json` — fixed schema across ALL papers           |
| a figure                  | `figures/fig-<slug>.json` + its image files; text-body figures (prompt boxes, examples) are in `figures/fig-<slug>.txt` |
| a table / equation / algo | `tables/tab-*.tex`, `equations/eq-*.tex`, `algorithms/alg-*.tex` |
| code from the paper       | `listings/lst-*.py` (real, runnable extensions)         |
| what `[@key]` cites       | `references.json`                                       |
| footnote `[^fn:N]`        | `footnotes.json`                                        |

Inline markers in section text: `[@key]` = citation, `[#fig:x]`/`[#tab:x]`/`[#eq:x]` = entity reference (the ID after `#` matches the entity's `id` in `paper.json` exactly).

**Completeness rule:** when assessing a paper's claims, do not stop at the Abstract — the sections are all local and cheap. Check `is_appendix` (on sections AND entities) to distinguish main-text evidence from appendix evidence.

**Honesty fields:** `metadata.*_source` says how each field was extracted (`arxiv_api`, `title_cmd`, `none`…). A citation with `title: null` means the paper shipped no `.bib` — quote `bib_raw`/key, don't guess the reference. `warnings.residue_top` lists LaTeX that survived cleaning; mention it if quoting affected sections.

## Program-calling patterns (preferred for ≥2 papers)

Don't `cat` paper-by-paper. Write one small script over `paper.json`:

```python
import json
from pathlib import Path

def load(pid, root="corpus"):
    return json.loads(Path(root, pid, "paper.json").read_text())

# Compare the Method sections of many papers
ids = ["2305.13860", "1706.03762", "2005.14165"]
methods = {
    pid: "\n\n".join(s["text"] for s in load(pid)["sections"]
                     if "method" in s["title"].lower() and not s["is_appendix"])
    for pid in ids
}

# Which sections cite a given work?
cited_in = {c["key"]: c["cited_in"] for c in load("2305.13860")["citations"]}

# All prompt/example text-figures of an LLM paper (main text only)
prompts = [f["text"] for f in load("2005.14165")["figures"]
           if f["body_tex"] and not f["is_appendix"]]
```

The schema is identical for every paper (`schema_version` in `paper.json`), so these loops never need per-paper special-casing.
