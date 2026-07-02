"""Unit tests for the arXiv abs-page metadata parser (offline — fixture HTML)."""

from arxiv2agent.arxiv_api import _author_display, parse_arxiv_abs_html

_PAGE = """<!DOCTYPE html><html><head>
<meta name="citation_title" content="Jailbreaking ChatGPT via Prompt Engineering:
  An Empirical Study" />
<meta name="citation_author" content="Liu, Yi" />
<meta name="citation_author" content="Deng, Gelei" />
<meta name="citation_author" content="OpenAI" />
<meta name="citation_date" content="2023/05/23" />
<meta name="citation_online_date" content="2024/03/10" />
<meta name="citation_abstract" content="We study   jailbreak
prompts &amp; their taxonomy." />
</head><body>
<h2>Submission history</h2>
<strong>[v1]</strong> Tue, 23 May 2023 ...
<strong>[v2]</strong> Sun, 10 Mar 2024 ...
</body></html>"""


def test_parse_abs_page_full():
    meta = parse_arxiv_abs_html(_PAGE)
    assert meta is not None
    # 'Last, First' → 'First Last'; collective names pass through
    assert meta["authors"] == ["Yi Liu", "Gelei Deng", "OpenAI"]
    # whitespace/newlines collapse; HTML entities unescape
    assert meta["title"] == (
        "Jailbreaking ChatGPT via Prompt Engineering: An Empirical Study"
    )
    assert meta["abstract"] == "We study jailbreak prompts & their taxonomy."
    assert meta["published"] == "2023-05-23"
    assert meta["updated"] == "2024-03-10"
    # highest [vN] in submission history = current revision
    assert meta["arxiv_version"] == "v2"


def test_parse_abs_page_rejects_pages_without_authors():
    assert parse_arxiv_abs_html("<html><body>No such paper</body></html>") is None
    assert parse_arxiv_abs_html("") is None


def test_author_display_forms():
    assert _author_display("He, Kaiming") == "Kaiming He"
    assert _author_display("OpenAI") == "OpenAI"
    assert _author_display(" Brown,  Tom B. ") == "Tom B. Brown"
