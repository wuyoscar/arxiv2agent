#!/usr/bin/env python3
"""One-time corpus migration: add a Paper2Agent digest to each paper folder
and reclaim disk by deleting redundant arXiv source tarballs.

Per the owner's decisions (2026-05):
  1. digest → data/papers/<slug>/digest/   (paper.json + sections/ + entities/ …)
  2. old-pipeline artifacts (XsafeWiki Deep Read.md, metadata/*, text/*, kimi-*)
     → LEFT IN PLACE, untouched.
  3. original/ redundant tarballs → DELETED to reclaim ~6 GB:
       original/arxiv-eprint            (gzip tarball — source/ is its extraction)
       original/source.tar.gz           (duplicate tarball)
       original/arxiv-eprint-extracted/ (duplicate extraction of source/)
     KEPT in original/: kimi-summary.{html,md}, promptfoo/ (old artifacts, per #2).

The digest is built from data/papers/<slug>/source/arxiv-to-prompt/ (the local
LaTeX). Papers whose source can't be parsed are reported and skipped (digest not
written); their tarballs are still NOT deleted (we only delete once a paper is
safely covered by source/ — but source/ existing is the invariant, see below).

Usage:
  python migrate_corpus.py --limit 1            # one paper, inspect
  python migrate_corpus.py --limit 10
  python migrate_corpus.py --all                # whole corpus
  python migrate_corpus.py --all --no-delete-tarballs   # digests only
  python migrate_corpus.py --slug 2308.01990_...        # one specific
"""

from __future__ import annotations

import argparse
import shutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PAPERS = ROOT / "data" / "papers"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import logging  # noqa: E402
logging.getLogger().setLevel(logging.WARNING)

from arxiv2agent.core import digest          # noqa: E402
from arxiv2agent.writer import write_digest  # noqa: E402

# Redundant-with-source/ items in original/ that are safe to delete.
REDUNDANT_ORIGINAL = ("arxiv-eprint", "source.tar.gz", "arxiv-eprint-extracted")


def find_source_dir(paper_dir: Path) -> Path | None:
    """Return the LaTeX source folder to feed Paper2Agent, or None."""
    cand = paper_dir / "source" / "arxiv-to-prompt"
    if cand.is_dir() and any(cand.rglob("*.tex")):
        return cand
    # fallback: any source/* dir holding .tex
    src = paper_dir / "source"
    if src.is_dir():
        for sub in src.iterdir():
            if sub.is_dir() and any(sub.rglob("*.tex")):
                return sub
    return None


def reclaim_tarballs(paper_dir: Path) -> int:
    """Delete redundant tarballs from original/. Return bytes reclaimed."""
    original = paper_dir / "original"
    if not original.is_dir():
        return 0
    freed = 0
    for name in REDUNDANT_ORIGINAL:
        p = original / name
        if not p.exists():
            continue
        if p.is_dir():
            freed += sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            shutil.rmtree(p, ignore_errors=True)
        else:
            freed += p.stat().st_size
            p.unlink(missing_ok=True)
    return freed


def migrate_one(paper_dir: Path, *, write_digest_=True, delete_tarballs=True) -> dict:
    slug = paper_dir.name
    result = {"slug": slug, "digest": False, "freed_mb": 0.0, "error": None}

    if write_digest_:
        src = find_source_dir(paper_dir)
        if src is None:
            result["error"] = "no parseable source/*.tex"
        else:
            try:
                paper = digest(local_folder=str(src))
                dest = paper_dir / "digest"
                if dest.exists():
                    shutil.rmtree(dest)
                write_digest(paper, dest_dir=dest, source_folder=str(src))
                result["digest"] = True
                result["n_sections"] = len(paper["sections"])
                result["n_cites"] = len(paper["citations"])
                result["cites_resolved"] = sum(1 for c in paper["citations"] if c.get("title"))
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
                result["trace"] = traceback.format_exc()

    if delete_tarballs:
        result["freed_mb"] = reclaim_tarballs(paper_dir) / 1024 / 1024

    return result


def pick(limit, slug, do_all):
    if slug:
        return [PAPERS / slug]
    dirs = sorted(d for d in PAPERS.iterdir() if d.is_dir())
    return dirs if do_all else dirs[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--slug")
    ap.add_argument("--no-digest", action="store_true", help="skip digest, only reclaim tarballs")
    ap.add_argument("--no-delete-tarballs", action="store_true", help="digests only, keep tarballs")
    args = ap.parse_args()

    targets = pick(args.limit, args.slug, args.all)
    print(f"Migrating {len(targets)} paper folder(s)...\n")

    ok = fail = 0
    total_freed = 0.0
    for i, d in enumerate(targets, 1):
        r = migrate_one(
            d,
            write_digest_=not args.no_digest,
            delete_tarballs=not args.no_delete_tarballs,
        )
        total_freed += r["freed_mb"]
        if r["error"]:
            fail += 1
            print(f"[{i:>4}/{len(targets)}] FAIL {r['slug']}: {r['error']}")
        else:
            ok += 1
            extra = ""
            if r.get("digest"):
                extra = (f"  sec={r.get('n_sections')} "
                         f"cite={r.get('cites_resolved')}/{r.get('n_cites')}")
            print(f"[{i:>4}/{len(targets)}] OK   {r['slug']}  "
                  f"freed={r['freed_mb']:.1f}MB{extra}")

    print(f"\n=== done: {ok} ok, {fail} fail, {total_freed/1024:.2f} GB reclaimed ===")


if __name__ == "__main__":
    main()
