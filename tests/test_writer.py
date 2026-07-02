"""Unit tests for writer.py."""

from __future__ import annotations

import json
from pathlib import Path

from arxiv2agent.writer import write_digest


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def minimal_paper() -> dict:
    return {
        "schema_version": "0.4",
        "paper_id": "paper-a",
        "source_kind": "local-folder",
        "metadata": {
            "title": "Paper A",
            "abstract": "",
            "authors": [],
            "affiliations": [],
            "title_source": "title_cmd",
            "abstract_source": "none",
            "affiliations_source": "none",
        },
        "sections": [],
        "figures": [
            {
                "id": "fig:overview",
                "latex_label": "fig:overview",
                "caption": "Overview figure.",
                "src_refs": ["figures/overview.pdf"],
                "body_tex": "",
                "defined_in": "sec:1",
                "referenced_in": [],
            }
        ],
        "tables": [
            {
                "id": "tab:results",
                "latex_label": "tab:results",
                "env": "table",
                "caption": "Main results.",
                "raw_tex": "\\begin{tabular}{lc}A & 1\\\\\\end{tabular}",
                "tex_lines": 1,
                "defined_in": "sec:1",
                "referenced_in": [],
                "cites_inside": [],
            }
        ],
        "equations": [
            {
                "id": "eq:loss",
                "latex_label": "eq:loss",
                "env": "equation",
                "raw_tex": "L = -\\sum_i y_i \\log p_i",
                "defined_in": "sec:1",
                "referenced_in": [],
            }
        ],
        "algorithms": [
            {
                "id": "alg:train",
                "latex_label": "alg:train",
                "caption": "Training loop.",
                "raw_tex": "\\State update model",
                "defined_in": "sec:1",
                "referenced_in": [],
            }
        ],
        "listings": [
            {
                "id": "lst:demo",
                "latex_label": "lst:demo",
                "language": "python",
                "caption": "Demo code.",
                "code": "print('hello')",
                "defined_in": "sec:1",
                "line_labels": [],
                "referenced_in": [],
            }
        ],
        "citations": [],
        "footnotes": [],
        "warnings": {},
    }


def test_write_digest_adds_reader_text_to_entity_json_and_paper_json(tmp_path):
    out = write_digest(minimal_paper(), output_dir=tmp_path)

    figure = read_json(out / "figures" / "fig-overview.json")
    table = read_json(out / "tables" / "tab-results.json")
    equation = read_json(out / "equations" / "eq-loss.json")
    algorithm = read_json(out / "algorithms" / "alg-train.json")
    listing = read_json(out / "listings" / "lst-demo.json")
    paper = read_json(out / "paper.json")

    assert figure["text"] == "Caption: Overview figure."
    assert "Caption: Main results." in table["text"]
    assert "\\begin{tabular}" in table["text"]
    assert equation["text"] == "L = -\\sum_i y_i \\log p_i"
    assert "Caption: Training loop." in algorithm["text"]
    assert "\\State update model" in algorithm["text"]
    assert "Caption: Demo code." in listing["text"]
    assert "print('hello')" in listing["text"]

    assert paper["figures"][0]["text"] == figure["text"]
    assert paper["tables"][0]["text"] == table["text"]
    assert paper["equations"][0]["text"] == equation["text"]
    assert paper["algorithms"][0]["text"] == algorithm["text"]
    assert paper["listings"][0]["text"] == listing["text"]


def test_multi_image_figure_assets_and_text_body(tmp_path):
    import json
    src = tmp_path / "src"
    (src / "images").mkdir(parents=True)
    (src / "images" / "a.pdf").write_bytes(b"%PDF-1.4 a")
    (src / "images" / "b.png").write_bytes(b"\x89PNG b")

    paper = {
        "schema_version": "0.5",
        "paper_id": "t1",
        "source_kind": "local-folder",
        "metadata": {
            "title": "T", "abstract": "", "authors": [], "affiliations": [],
            "title_source": "title_cmd", "abstract_source": "none",
            "affiliations_source": "none",
        },
        "sections": [],
        "figures": [
            {
                "id": "fig:panels", "latex_label": "fig:panels",
                "caption": "Two panels.",
                "src_refs": ["images/a.pdf", "images/b.png"],
                "body_tex": "", "text": "Caption: Two panels.",
                "defined_in": None, "is_appendix": False, "referenced_in": [],
            },
            {
                "id": "fig:prompt", "latex_label": "fig:prompt",
                "caption": "System prompt.",
                "src_refs": [],
                "body_tex": "\\fbox{Prompt: be helpful}",
                "text": "Caption: System prompt.\n\nPrompt: be helpful",
                "defined_in": None, "is_appendix": True, "referenced_in": [],
            },
        ],
        "tables": [], "equations": [], "algorithms": [], "listings": [],
        "citations": [], "footnotes": [], "warnings": {},
    }
    root = write_digest(paper, output_dir=tmp_path / "out", source_folder=src)

    # Multi-image figure: BOTH assets copied with indexed names.
    assert (root / "figures" / "fig-panels-1.pdf").is_file()
    assert (root / "figures" / "fig-panels-2.png").is_file()
    meta = json.loads((root / "figures" / "fig-panels.json").read_text())
    assert meta["asset_files"] == ["figures/fig-panels-1.pdf", "figures/fig-panels-2.png"]

    # Text-body figure: body lands as a standalone .txt next to the json.
    txt = (root / "figures" / "fig-prompt.txt").read_text()
    assert "Prompt: be helpful" in txt
    meta = json.loads((root / "figures" / "fig-prompt.json").read_text())
    assert meta["content_file"] == "fig-prompt.txt"
    assert meta["is_appendix"] is True

    # README marks the appendix figure.
    readme = (root / "README.md").read_text()
    assert "`fig:prompt` — System prompt." in readme
    assert "*(appendix)*" in readme
