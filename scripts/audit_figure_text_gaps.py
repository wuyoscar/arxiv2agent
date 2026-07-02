#!/usr/bin/env python3
"""Audit figure text extraction gaps in Paper2Agent digests.

The script is read-only for ``data/papers``. It scans per-figure digest JSON
files plus matching ``digest/paper.json`` entries, classifies likely extraction
gaps, and writes review artifacts under ``.workspace/figure-text-audit``.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PAPERS = ROOT / "data" / "papers"
OUT_DIR = ROOT / ".workspace" / "figure-text-audit"

FAILURE_STATUSES = {
    "asset_missing",
    "unsupported_asset",
    "render_failed",
    "codex_prompt_failed",
    "vlm_failed",
}
REFUSAL_RE = re.compile(
    r"\b("
    r"i\s*(?:am|'m|’m)\s+sorry|"
    r"i\s+apologize|"
    r"sorry[, ]|"
    r"i\s+can(?:not|'t|’t)|"
    r"cannot\s+(?:assist|help|fulfill|provide|comply)|"
    r"can(?:not|'t|’t)\s+(?:assist|help|fulfill|provide|comply)|"
    r"unable\s+to\s+(?:assist|help|fulfill|provide|comply)|"
    r"content\s+policy|"
    r"harmful\s+request|"
    r"unsafe\s+request"
    r")\b",
    re.IGNORECASE,
)
TRUNCATION_RE = re.compile(r"(\.\.\.|…|\[truncated\]|<truncated>|truncated due to|cut off)", re.IGNORECASE)
TEXT_CAPS = {
    "visible_text": 2400,
    "summary": 2400,
    "safety_relevant_content": 1200,
    "model_response": 700,
}
STRICT_ISSUE_BUCKETS = {
    "missing_visual_extraction",
    "status_no_visible_text",
    "status_asset_missing",
    "status_vlm_failed",
    "status_render_failed",
    "status_codex_prompt_failed",
    "ok_empty_visible_text",
    "refusal_like_only_or_too_short",
}


@dataclass
class PdfFallback:
    status: str = "not_requested"
    path: str = ""
    chars: int = 0
    error: str = ""


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def iter_figure_jsons(slugs: list[str] | None) -> list[Path]:
    if slugs:
        paths: list[Path] = []
        for slug in slugs:
            paths.extend(sorted((PAPERS / slug / "digest" / "figures").glob("*.json")))
        return paths
    return sorted(PAPERS.glob("*/digest/figures/*.json"))


def get_text_items(visual: dict[str, Any] | None) -> list[tuple[str, str, str]]:
    if not isinstance(visual, dict):
        return []
    items: list[tuple[str, str, str]] = []
    for item in visual.get("visible_text") or []:
        if isinstance(item, dict):
            text = str(item.get("text") or "")
            items.append(("visible_text", str(item.get("role") or ""), text))
        elif isinstance(item, str):
            items.append(("visible_text", "", item))
    for item in visual.get("safety_relevant_content") or []:
        if isinstance(item, dict):
            text = str(item.get("text") or "")
            items.append(("safety_relevant_content", str(item.get("kind") or ""), text))
        elif isinstance(item, str):
            items.append(("safety_relevant_content", "", item))
    for key in ("summary", "uncertain"):
        value = visual.get(key)
        if isinstance(value, str):
            items.append((key, "", value))
    return items


def count_items(visual: dict[str, Any] | None, key: str) -> int:
    if not isinstance(visual, dict):
        return 0
    value = visual.get(key)
    return len(value) if isinstance(value, list) else 0


def text_stats(items: list[tuple[str, str, str]]) -> tuple[int, int, list[str], bool]:
    joined = "\n".join(text for _, _, text in items if text)
    markers = sorted({m.group(0) for m in REFUSAL_RE.finditer(joined)})
    possible_truncation = False
    if TRUNCATION_RE.search(joined):
        possible_truncation = True
    for source, role, text in items:
        cap = TEXT_CAPS.get(source)
        if role in {"response", "model_response"}:
            cap = min(cap or TEXT_CAPS["model_response"], TEXT_CAPS["model_response"])
        if cap and len(text) >= cap - 20:
            possible_truncation = True
        if text.rstrip().endswith(("...", "…")):
            possible_truncation = True
    return len(joined), len(joined.split()), markers, possible_truncation


def classify(
    *,
    figure: dict[str, Any],
    visual: dict[str, Any] | None,
    paper_visual: dict[str, Any] | None,
    asset_exists: bool,
) -> tuple[list[str], dict[str, Any]]:
    buckets: list[str] = []
    status = visual.get("status") if isinstance(visual, dict) else ""
    items = get_text_items(visual)
    chars, words, markers, possible_truncation = text_stats(items)
    visible_items = count_items(visual, "visible_text")
    safety_items = count_items(visual, "safety_relevant_content")

    if not isinstance(visual, dict):
        buckets.append("missing_visual_extraction")
    elif status in FAILURE_STATUSES:
        buckets.append(f"status_{status}")
    elif status and status != "ok":
        buckets.append(f"status_{status}")

    if isinstance(visual, dict) and status == "ok" and visible_items == 0:
        buckets.append("ok_empty_visible_text")

    if isinstance(visual, dict) and not asset_exists:
        buckets.append("asset_file_missing_now")

    if paper_visual is None and isinstance(visual, dict):
        buckets.append("paper_json_missing_visual_extraction")
    elif isinstance(visual, dict) and isinstance(paper_visual, dict):
        paper_status = paper_visual.get("status")
        if paper_status != status:
            buckets.append("paper_json_status_mismatch")

    if markers:
        # Rich prompt/response examples often legitimately include target-model
        # refusals. Short, refusal-only outputs are more likely extraction gaps.
        non_refusal = REFUSAL_RE.sub("", "\n".join(text for _, _, text in items))
        if chars < 450 and len(non_refusal.strip()) < 180 and len(items) <= 3:
            buckets.append("refusal_like_only_or_too_short")
        else:
            buckets.append("contains_refusal_text_rich_context")

    if possible_truncation:
        buckets.append("possible_truncated_text")

    if not buckets:
        buckets.append("ok")

    stats = {
        "status": status or "missing",
        "visible_items": visible_items,
        "safety_items": safety_items,
        "text_chars": chars,
        "text_words": words,
        "refusal_markers": markers,
        "caption_chars": len(str(figure.get("caption") or "")),
    }
    return buckets, stats


def paper_json_index(slug: str) -> dict[str, dict[str, Any]]:
    path = PAPERS / slug / "digest" / "paper.json"
    if not path.is_file():
        return {}
    data = load_json(path)
    index: dict[str, dict[str, Any]] = {}
    for fig in data.get("figures") or []:
        if not isinstance(fig, dict):
            continue
        for key in (fig.get("file"), fig.get("id"), fig.get("latex_label")):
            if key:
                index[str(key)] = fig
    return index


def resolve_asset(fig_path: Path, figure: dict[str, Any]) -> Path | None:
    for ext in (".pdf", ".png", ".jpg", ".jpeg", ".svg", ".eps"):
        candidate = fig_path.with_suffix(ext)
        if candidate.is_file():
            return candidate
    src_ref = figure.get("src_ref")
    if src_ref:
        candidate = fig_path.parent / Path(str(src_ref)).name
        if candidate.is_file():
            return candidate
    return None


def find_pdf(slug: str) -> Path | None:
    paper_dir = PAPERS / slug
    candidates = [
        paper_dir / "raw" / "paper.pdf",
        paper_dir / "paper.pdf",
        paper_dir / "source" / "paper.pdf",
    ]
    candidates.extend(sorted((paper_dir / "raw").glob("*.pdf")) if (paper_dir / "raw").is_dir() else [])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def write_pdf_text(slug: str, out_dir: Path) -> PdfFallback:
    pdftotext = shutil.which("pdftotext")
    if not pdftotext:
        return PdfFallback(status="tool_missing", error="pdftotext not found")
    pdf = find_pdf(slug)
    if not pdf:
        return PdfFallback(status="pdf_missing", error="no raw PDF found")
    target_dir = out_dir / "pdf_text"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{slug}.txt"
    cmd = [pdftotext, "-layout", "-enc", "UTF-8", str(pdf), str(target)]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=120)
    if proc.returncode != 0:
        return PdfFallback(status="failed", error=(proc.stderr or proc.stdout).strip()[:500])
    text = target.read_text(encoding="utf-8", errors="replace")
    return PdfFallback(status="ok", path=rel(target), chars=len(text))


def write_outputs(rows: list[dict[str, Any]], pdf_fallbacks: dict[str, PdfFallback], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "figure_text_audit.json"
    csv_path = out_dir / "figure_text_audit.csv"
    strict_csv_path = out_dir / "strict_issues.csv"
    refusal_csv_path = out_dir / "refusal_context_review.csv"
    truncated_csv_path = out_dir / "truncated_review.csv"
    summary_path = out_dir / "SUMMARY.md"

    status_counts = Counter(row["status"] for row in rows)
    bucket_counts = Counter(bucket for row in rows for bucket in row["buckets"])
    issue_rows = [row for row in rows if row["buckets"] != ["ok"]]
    strict_issue_rows = [row for row in rows if STRICT_ISSUE_BUCKETS & set(row["buckets"])]
    paper_issue_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in issue_rows:
        for bucket in row["buckets"]:
            if bucket != "ok":
                paper_issue_counts[row["slug"]][bucket] += 1

    payload = {
        "schema_version": "figure_text_audit.v0.1",
        "root": str(ROOT),
        "total_figures": len(rows),
        "issue_figures": len(issue_rows),
        "strict_issue_figures": len(strict_issue_rows),
        "status_counts": dict(status_counts),
        "bucket_counts": dict(bucket_counts),
        "pdf_fallbacks": {slug: vars(item) for slug, item in sorted(pdf_fallbacks.items())},
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fieldnames = [
        "slug",
        "figure_file",
        "figure_id",
        "status",
        "buckets",
        "visible_items",
        "safety_items",
        "text_chars",
        "text_words",
        "refusal_markers",
        "asset_path",
        "caption_chars",
    ]
    def write_csv(path: Path, selected_rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in selected_rows:
                writer.writerow(
                    {
                        **{key: row.get(key, "") for key in fieldnames},
                        "buckets": ";".join(row["buckets"]),
                        "refusal_markers": ";".join(row["refusal_markers"]),
                    }
                )

    write_csv(csv_path, rows)
    write_csv(strict_csv_path, strict_issue_rows)
    write_csv(
        refusal_csv_path,
        [row for row in rows if "contains_refusal_text_rich_context" in row["buckets"]],
    )
    write_csv(
        truncated_csv_path,
        [row for row in rows if "possible_truncated_text" in row["buckets"]],
    )

    lines = [
        "# Figure Text Audit",
        "",
        f"- Total figure JSON files: {len(rows)}",
        f"- Strict issue figures: {len(strict_issue_rows)}",
        f"- Figures with any issue/manual-review bucket: {len(issue_rows)}",
        "",
        "## Status Counts",
        "",
    ]
    for key, value in status_counts.most_common():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Bucket Counts", ""])
    for key, value in bucket_counts.most_common():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Top Papers By Issue Buckets", ""])
    for slug, counts in sorted(paper_issue_counts.items(), key=lambda item: sum(item[1].values()), reverse=True)[:40]:
        rendered = ", ".join(f"{key}={value}" for key, value in counts.most_common())
        lines.append(f"- `{slug}`: {rendered}")
    if pdf_fallbacks:
        lines.extend(["", "## PDF Text Fallbacks", ""])
        pdf_counts = Counter(item.status for item in pdf_fallbacks.values())
        for key, value in pdf_counts.most_common():
            lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Manual Review Priority", ""])
    priority_buckets = [
        "missing_visual_extraction",
        "status_vlm_failed",
        "status_render_failed",
        "status_codex_prompt_failed",
        "ok_empty_visible_text",
        "refusal_like_only_or_too_short",
        "possible_truncated_text",
    ]
    for bucket in priority_buckets:
        matches = [row for row in rows if bucket in row["buckets"]]
        if not matches:
            continue
        lines.append(f"### `{bucket}` ({len(matches)})")
        for row in matches[:25]:
            lines.append(
                f"- `{row['slug']}` `{row['figure_file']}` "
                f"status=`{row['status']}` chars={row['text_chars']} asset=`{row['asset_path']}`"
            )
        lines.append("")
    summary_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", action="append", help="Limit audit to one slug. May be repeated.")
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="Output directory for audit artifacts.")
    parser.add_argument(
        "--pdf-text",
        action="store_true",
        help="Also write pdftotext output for papers selected by --pdf-text-scope.",
    )
    parser.add_argument(
        "--pdf-text-scope",
        choices=("strict", "review"),
        default="strict",
        help="PDF fallback scope. strict excludes rich-context refusal and truncation-only review buckets.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    rows: list[dict[str, Any]] = []
    paper_indexes: dict[str, dict[str, dict[str, Any]]] = {}

    for fig_path in iter_figure_jsons(args.slug):
        slug = fig_path.parts[fig_path.parts.index("papers") + 1]
        paper_indexes.setdefault(slug, paper_json_index(slug))
        figure = load_json(fig_path)
        visual = figure.get("visual_extraction")
        paper_fig = (
            paper_indexes[slug].get(str(figure.get("file") or ""))
            or paper_indexes[slug].get(rel(fig_path).split(f"data/papers/{slug}/digest/", 1)[-1])
            or paper_indexes[slug].get(str(figure.get("id") or ""))
            or paper_indexes[slug].get(str(figure.get("latex_label") or ""))
        )
        paper_visual = paper_fig.get("visual_extraction") if isinstance(paper_fig, dict) else None
        asset = resolve_asset(fig_path, figure)
        buckets, stats = classify(
            figure=figure,
            visual=visual if isinstance(visual, dict) else None,
            paper_visual=paper_visual if isinstance(paper_visual, dict) else None,
            asset_exists=bool(asset),
        )
        rows.append(
            {
                "slug": slug,
                "figure_file": rel(fig_path),
                "figure_id": str(figure.get("id") or figure.get("latex_label") or fig_path.stem),
                "status": stats["status"],
                "buckets": buckets,
                "visible_items": stats["visible_items"],
                "safety_items": stats["safety_items"],
                "text_chars": stats["text_chars"],
                "text_words": stats["text_words"],
                "refusal_markers": stats["refusal_markers"],
                "asset_path": rel(asset) if asset else "",
                "caption_chars": stats["caption_chars"],
            }
        )

    pdf_fallbacks: dict[str, PdfFallback] = {}
    if args.pdf_text:
        if args.pdf_text_scope == "strict":
            issue_slugs = sorted(
                {row["slug"] for row in rows if STRICT_ISSUE_BUCKETS & set(row["buckets"])}
            )
        else:
            issue_slugs = sorted({row["slug"] for row in rows if row["buckets"] != ["ok"]})
        for slug in issue_slugs:
            pdf_fallbacks[slug] = write_pdf_text(slug, out_dir)

    write_outputs(rows, pdf_fallbacks, out_dir)
    issue_count = sum(1 for row in rows if row["buckets"] != ["ok"])
    strict_count = sum(1 for row in rows if STRICT_ISSUE_BUCKETS & set(row["buckets"]))
    print(f"wrote {rel(out_dir / 'SUMMARY.md')}")
    print(f"figures={len(rows)} strict_issue={strict_count} issue_or_review={issue_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
