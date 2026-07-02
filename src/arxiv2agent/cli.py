"""arxiv2agent CLI — turn arXiv papers into agent-friendly digest folders.

Accepts a LIST of arXiv IDs and processes them sequentially (the built-in
politeness throttle spaces the network requests). One failing paper does not
abort the batch. Output is always self-contained folders; see README.md
inside any output folder for navigation guidance.
"""

from __future__ import annotations

import argparse
import sys

from arxiv2agent.core import digest
from arxiv2agent.writer import write_digest


def _run_one(arxiv_id: str | None, local_folder: str | None, args) -> int:
    paper = digest(arxiv_id=arxiv_id, local_folder=local_folder)

    if local_folder:
        source_folder = local_folder
    else:
        from arxiv2agent._tex import get_default_cache_dir
        source_folder = str(get_default_cache_dir() / arxiv_id)

    out_root = write_digest(
        paper, output_dir=args.output, source_folder=source_folder,
        include_source=args.include_source,
    )
    print(f"Wrote: {out_root}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="arxiv2agent",
        description=(
            "Convert arXiv papers into agent-friendly DIGEST folders "
            "(README.md + paper.json + sections/*.md + entities + assets). "
            "Pass multiple IDs to batch-process sequentially."
        ),
    )
    p.add_argument(
        "arxiv_ids",
        nargs="*",
        metavar="ARXIV_ID",
        help="One or more arXiv IDs (e.g. 2305.13860 1706.03762). "
             "Omit with --local-folder.",
    )
    p.add_argument(
        "-o", "--output",
        default=".",
        help="Parent directory; each digest lands at <output>/<arxiv_id>/ "
             "(default: current dir).",
    )
    p.add_argument(
        "--local-folder",
        help="Use a local LaTeX folder instead of downloading from arXiv.",
    )
    p.add_argument(
        "--include-source",
        action="store_true",
        help="Also mirror the original LaTeX tree into <digest>/source/ for audit. "
             "Off by default (most agents don't need it; it inflates the digest ~50×).",
    )
    args = p.parse_args(argv)

    if not args.arxiv_ids and not args.local_folder:
        p.error("Provide at least one arxiv_id or --local-folder.")
    if args.arxiv_ids and args.local_folder:
        p.error("arxiv_ids and --local-folder are mutually exclusive.")

    if args.local_folder:
        return _run_one(None, args.local_folder, args)

    failures: list[str] = []
    for arxiv_id in args.arxiv_ids:
        try:
            _run_one(arxiv_id, None, args)
        except Exception as exc:  # keep the batch going; report at the end
            failures.append(arxiv_id)
            print(f"FAILED: {arxiv_id} — {exc}", file=sys.stderr)

    n = len(args.arxiv_ids)
    if n > 1:
        print(f"Done: {n - len(failures)}/{n} papers digested.", file=sys.stderr)
    if failures:
        print(f"Failed IDs: {' '.join(failures)}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
