"""Shared pytest configuration.

Adds --update-golden for the corpus regression suite (tests/test_corpus.py):
run `pytest -m corpus --update-golden` to (re)generate tests/golden/*.json
snapshots after an intentional extraction change.
"""

from __future__ import annotations


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Rewrite tests/golden/*.json snapshots instead of asserting against them.",
    )
