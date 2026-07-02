#!/usr/bin/env python3
"""Audit every Paper2Agent digest for quality anomalies, ranked by severity.

Phase-1 (Corpus hardening) diagnostic: scan all data/papers/<slug>/digest/
paper.json and surface systematic extraction problems so we can decide which
are TOOL bugs (fix in extract/denoise/core) vs legit content variance.

Anomaly classes & severity:
  HIGH   schema_version != 0.4            (non-uniform corpus)
         sections <= 1                    (structural parse failure)
         title empty                      (lost the title)
         listing language looks like content (regression of the fixed bug)
  MED    abstract empty / abstract_source=none
         >30% of sections have empty text
         residue: a single leftover command repeated a lot (noisy denoise)
  LOW    citations present but 0% resolved (usually .bbl-only — known limit)
         figure/listing entity with no content/binary (often tikz — expected)
  INFO   per-paper residue total, section/citation counts (for distributions)

Usage:
  python audit_digests.py                 # ranked summary to stdout
  python audit_digests.py --json OUT.json # also dump full per-paper findings
  python audit_digests.py --top 15        # show N worst examples per anomaly
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PAPERS = ROOT / "data" / "papers"

# severity rank for sorting
SEV = {"HIGH": 0, "MED": 1, "LOW": 2, "INFO": 3}

# heuristics (tunable)
RESIDUE_CMD_HIGH = 25          # one leftover command repeated >= this = noisy denoise
EMPTY_SECTION_FRAC = 0.30      # >this fraction of empty-text sections = MED
VISUAL_FAILED = {
    "asset_missing",
    "unsupported_asset",
    "render_failed",
    "codex_prompt_failed",
    "vlm_failed",
}


def lang_looks_like_content(lang: str) -> bool:
    return bool(lang) and ("\n" in lang or len(lang) > 30)


def figure_visual_status_counts(digest: dict) -> Counter:
    counts: Counter = Counter()
    for fig in digest.get("figures", []) or []:
        ve = fig.get("visual_extraction") if isinstance(fig, dict) else None
        status = ve.get("status") if isinstance(ve, dict) else None
        counts[status or "missing"] += 1
    return counts


def figure_file_visual_status_counts(digest_dir: Path) -> Counter:
    counts: Counter = Counter()
    figdir = digest_dir / "figures"
    if not figdir.is_dir():
        return counts
    for fig_json in figdir.glob("*.json"):
        try:
            fig = json.loads(fig_json.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            counts["unreadable_figure_json"] += 1
            continue
        ve = fig.get("visual_extraction") if isinstance(fig, dict) else None
        status = ve.get("status") if isinstance(ve, dict) else None
        counts[status or "missing"] += 1
    return counts


def figure_structure_counts(digest: dict, digest_dir: Path) -> Counter:
    counts: Counter = Counter()
    paper_ids = [fig.get("id") for fig in digest.get("figures", []) or [] if isinstance(fig, dict)]
    seen: Counter = Counter(fid for fid in paper_ids if fid)
    counts["paper_json_duplicate_entries"] = sum(count - 1 for count in seen.values() if count > 1)

    figdir = digest_dir / "figures"
    if not figdir.is_dir():
        if paper_ids:
            counts["paper_json_ids_without_figure_file"] = len({fid for fid in paper_ids if fid})
        return counts

    file_ids = set()
    for fig_json in figdir.glob("*.json"):
        try:
            fig = json.loads(fig_json.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        fid = fig.get("id") if isinstance(fig, dict) else None
        if fid:
            file_ids.add(fid)

    paper_id_set = {fid for fid in paper_ids if fid}
    counts["paper_json_ids_without_figure_file"] = len(paper_id_set - file_ids)
    counts["figure_file_ids_without_paper_json"] = len(file_ids - paper_id_set)
    return counts


def audit_one(paper_json: Path) -> list[dict]:
    """Return a list of {severity, kind, detail} anomalies for one paper."""
    slug = paper_json.parent.parent.name
    out: list[dict] = []
    try:
        d = json.loads(paper_json.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:  # noqa: BLE001
        return [{"slug": slug, "severity": "HIGH", "kind": "unreadable_paper_json",
                 "detail": f"{type(exc).__name__}: {exc}"}]

    def add(sev, kind, detail=""):
        out.append({"slug": slug, "severity": sev, "kind": kind, "detail": detail})

    # --- schema uniformity ---
    if d.get("schema_version") != "0.4":
        add("HIGH", "schema_mismatch", f"schema_version={d.get('schema_version')!r}")

    # --- sections ---
    sections = d.get("sections", [])
    if len(sections) <= 1:
        add("HIGH", "too_few_sections", f"n_sections={len(sections)}")
    empty_sec = sum(1 for s in sections if not (s.get("text") or "").strip())
    if sections and empty_sec / len(sections) > EMPTY_SECTION_FRAC:
        add("MED", "many_empty_sections",
            f"{empty_sec}/{len(sections)} sections have empty text")

    # --- title / abstract ---
    md = d.get("metadata", {})
    if not (md.get("title") or "").strip():
        add("HIGH", "empty_title", f"title_source={md.get('title_source')}")
    if not (md.get("abstract") or "").strip():
        add("MED", "empty_abstract", f"abstract_source={md.get('abstract_source')}")

    # --- listings: language-as-content regression ---
    for lst in d.get("listings", []):
        if lang_looks_like_content(lst.get("language") or ""):
            add("HIGH", "listing_language_is_content",
                f"{lst.get('id')}: language={ (lst.get('language') or '')[:40]!r}")
            break

    # --- residue (noisy denoise / unhandled macros) ---
    residue = (d.get("warnings", {}) or {}).get("residue_top", []) or []
    residue_total = sum(r.get("count", 0) for r in residue)
    worst = max((r.get("count", 0) for r in residue), default=0)
    if worst >= RESIDUE_CMD_HIGH:
        top = residue[0] if residue else {}
        add("MED", "high_residue_command",
            f"{top.get('command')!r}×{top.get('count')} (residue_total={residue_total})")

    # --- citation resolution ---
    cites = d.get("citations", [])
    if cites:
        resolved = sum(1 for c in cites if c.get("title"))
        if resolved == 0:
            add("LOW", "zero_citation_resolution", f"{len(cites)} cites, 0 resolved")

    # --- entity content/binary gaps (figures w/o binary, empty listings) ---
    digest = paper_json.parent
    figdir = digest / "figures"
    if figdir.is_dir():
        figjson = list(figdir.glob("*.json"))
        figbin = [f for f in figdir.iterdir() if f.suffix != ".json"]
        if figjson and not figbin:
            add("LOW", "figures_without_binary",
                f"{len(figjson)} fig json, 0 image binaries")

    visual_counts = figure_visual_status_counts(d)
    if visual_counts:
        missing = visual_counts.get("missing", 0)
        failed = sum(visual_counts.get(k, 0) for k in VISUAL_FAILED)
        if missing:
            add("LOW", "figures_missing_visual_extraction",
                f"{missing}/{sum(visual_counts.values())} figures missing visual_extraction")
        if failed:
            failed_detail = ", ".join(
                f"{k}={visual_counts[k]}" for k in sorted(VISUAL_FAILED) if visual_counts.get(k)
            )
            add("LOW", "figures_visual_extraction_failed", failed_detail)

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default=None, help="dump full findings JSON here")
    ap.add_argument("--top", type=int, default=10, help="worst examples per anomaly kind")
    args = ap.parse_args()

    digests = sorted(PAPERS.glob("*/digest/paper.json"))
    no_digest = sorted(p for p in PAPERS.glob("*/digest/STATUS.json"))
    all_findings: list[dict] = []
    # distributions
    sec_counts, cite_counts, residue_totals = [], [], []
    visual_status_counts: Counter = Counter()
    visual_file_status_counts: Counter = Counter()
    visual_structure_counts: Counter = Counter()

    for pj in digests:
        all_findings.extend(audit_one(pj))
        try:
            d = json.loads(pj.read_text(encoding="utf-8", errors="replace"))
            sec_counts.append(len(d.get("sections", [])))
            cite_counts.append(len(d.get("citations", [])))
            residue_top = (d.get("warnings", {}) or {}).get("residue_top", []) or []
            residue_totals.append(sum(r.get("count", 0) for r in residue_top))
            visual_status_counts.update(figure_visual_status_counts(d))
            visual_file_status_counts.update(figure_file_visual_status_counts(pj.parent))
            visual_structure_counts.update(figure_structure_counts(d, pj.parent))
        except Exception:
            pass

    # aggregate by (severity, kind)
    by_kind: dict[tuple, list[dict]] = defaultdict(list)
    for f in all_findings:
        by_kind[(f["severity"], f["kind"])].append(f)

    print(
        f"=== Digest audit: {len(digests)} digests, "
        f"{len(no_digest)} STATUS-only (no paper.json) ===\n"
    )

    def dist(name, xs):
        if not xs:
            return
        xs2 = sorted(xs)
        print(f"  {name}: min={xs2[0]} p50={statistics.median(xs2):.0f} "
              f"p95={xs2[int(0.95*(len(xs2)-1))]} max={xs2[-1]} mean={statistics.mean(xs2):.1f}")
    print("distributions:")
    dist("sections/paper", sec_counts)
    dist("citations/paper", cite_counts)
    dist("residue_total/paper", residue_totals)
    if visual_status_counts:
        total_figures = sum(visual_status_counts.values())
        print("  figure visual extraction (paper.json entries):")
        for status, count in sorted(visual_status_counts.items()):
            print(f"    {status}: {count}")
        print(f"    total: {total_figures}")
    if visual_file_status_counts:
        total_figure_files = sum(visual_file_status_counts.values())
        print("  figure visual extraction (digest/figures JSON files):")
        for status, count in sorted(visual_file_status_counts.items()):
            print(f"    {status}: {count}")
        print(f"    total: {total_figure_files}")
    if visual_structure_counts:
        print("  figure structure:")
        for key, count in sorted(visual_structure_counts.items()):
            if count:
                print(f"    {key}: {count}")
    print()

    print("anomalies (severity-ranked):")
    for (sev, kind) in sorted(by_kind, key=lambda k: (SEV[k[0]], -len(by_kind[k]))):
        items = by_kind[(sev, kind)]
        print(f"\n[{sev}] {kind}: {len(items)} papers")
        for it in items[:args.top]:
            print(f"    {it['slug']}  {it['detail']}")
        if len(items) > args.top:
            print(f"    … +{len(items) - args.top} more")

    if args.json:
        Path(args.json).write_text(json.dumps(all_findings, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        print(f"\nfull findings -> {args.json}")


if __name__ == "__main__":
    main()
