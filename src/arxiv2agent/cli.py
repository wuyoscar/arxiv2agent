"""arxiv2agent CLI — find, fetch, and digest arXiv papers for agents.

    arxiv2agent find "Attention Is All You Need"     # title → candidate IDs
    arxiv2agent fetch 1706.03762 -o papers/           # raw LaTeX source + flattened .tex
    arxiv2agent digest 1706.03762 -o papers/          # structured digest folder

fetch/digest accept a LIST of IDs and process them sequentially (the built-in
politeness throttle spaces the requests). One failing paper does not abort
the batch.
"""

from __future__ import annotations

import argparse
import json
import sys

from arxiv2agent.core import digest
from arxiv2agent.fetch import fetch
from arxiv2agent.search import find
from arxiv2agent.writer import write_digest


def _cmd_find(args) -> int:
    results = find(" ".join(args.title), max_results=args.n)
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for r in results:
            first = r["authors"][0] if r["authors"] else "?"
            etal = " et al." if len(r["authors"]) > 1 else ""
            print(f"{r['id']}\t{r['year']}\t{r['title']}\t{first}{etal}")
    if not results:
        print("No arXiv match.", file=sys.stderr)
        return 1
    return 0


def _digest_one(arxiv_id: str | None, local_folder: str | None, args) -> None:
    paper = digest(arxiv_id=arxiv_id, local_folder=local_folder)
    if local_folder:
        source_folder = local_folder
    else:
        from arxiv2agent._tex import get_default_cache_dir
        source_folder = str(get_default_cache_dir() / arxiv_id)
    out = write_digest(paper, output_dir=args.output, source_folder=source_folder)
    print(f"Wrote: {out}", file=sys.stderr)


def _fetch_one(arxiv_id: str, _local_folder, args) -> None:
    print(f"Wrote: {fetch(arxiv_id, args.output)}", file=sys.stderr)


def _batch(run_one, args) -> int:
    failures: list[str] = []
    for arxiv_id in args.arxiv_ids:
        try:
            run_one(arxiv_id, None, args)
        except Exception as exc:  # keep the batch going; report at the end
            failures.append(arxiv_id)
            print(f"FAILED: {arxiv_id} — {exc}", file=sys.stderr)
    n = len(args.arxiv_ids)
    if n > 1:
        print(f"Done: {n - len(failures)}/{n} papers.", file=sys.stderr)
    if failures:
        print(f"Failed IDs: {' '.join(failures)}", file=sys.stderr)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="arxiv2agent",
        description="Find, download, and extract arXiv papers for agents.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("find", help="Search arXiv by title; print candidate IDs.")
    pf.add_argument("title", nargs="+")
    pf.add_argument("-n", type=int, default=5, help="max candidates (default 5)")
    pf.add_argument("--json", action="store_true", help="print JSON instead of TSV")

    for name, help_ in (
        ("fetch", "Download LaTeX source: <out>/<id>/source/ + <out>/<id>/<id>.tex"),
        ("digest", "Structured digest: paper.json + sections/ + figures/tables/…"),
    ):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("arxiv_ids", nargs="*", metavar="ARXIV_ID")
        sp.add_argument("-o", "--output", default=".", help="parent dir (default: .)")
        if name == "digest":
            sp.add_argument("--local-folder", help="digest a local LaTeX folder instead")

    args = p.parse_args(argv)

    if args.cmd == "find":
        return _cmd_find(args)
    if args.cmd == "fetch":
        if not args.arxiv_ids:
            p.error("fetch: provide at least one ARXIV_ID.")
        return _batch(_fetch_one, args)

    if bool(args.arxiv_ids) == bool(args.local_folder):
        p.error("digest: provide ARXIV_IDs or --local-folder (not both).")
    if args.local_folder:
        _digest_one(None, args.local_folder, args)
        return 0
    return _batch(_digest_one, args)


if __name__ == "__main__":
    raise SystemExit(main())
