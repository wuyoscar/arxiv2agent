"""Unit tests for arXiv title search (offline — HTTP stubbed)."""

from arxiv2agent import search

_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v7</id>
    <published>2017-06-12T17:57:34Z</published>
    <title>Attention Is All
      You Need</title>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
  </entry>
</feed>"""
_EMPTY = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'


def test_parse_atom_strips_version_and_whitespace():
    [r] = search.parse_atom(_FEED)
    assert r == {"id": "1706.03762", "title": "Attention Is All You Need",
                 "authors": ["Ashish Vaswani", "Noam Shazeer"], "year": "2017"}


def test_find_falls_back_from_phrase_to_words(monkeypatch):
    urls = []
    def fake_get(url, timeout):
        urls.append(url)
        return _EMPTY if len(urls) == 1 else _FEED
    monkeypatch.setattr(search, "_throttled_get", fake_get)
    assert search.find("Attention: Is All You Need?")[0]["id"] == "1706.03762"
    assert len(urls) == 2
    assert "ti%3A%22Attention+Is+All+You+Need%22" in urls[0]  # punctuation stripped


def test_find_empty_title_makes_no_request(monkeypatch):
    monkeypatch.setattr(search, "_throttled_get", lambda *a, **k: 1 / 0)
    assert search.find("?!") == []
