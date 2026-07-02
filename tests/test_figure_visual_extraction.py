"""Unit tests for the figure visual extraction enrichment tool."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import urllib.error
from contextlib import contextmanager
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "extract_figure_content.py"
SPEC = importlib.util.spec_from_file_location("extract_figure_content", SCRIPT)
extract_figure_content = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = extract_figure_content
SPEC.loader.exec_module(extract_figure_content)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_digest(tmp_path: Path) -> tuple[Path, Path, Path]:
    digest = tmp_path / "data" / "papers" / "paper-a" / "digest"
    fig_json = digest / "figures" / "fig-demo.json"
    asset = digest / "figures" / "fig-demo.png"
    figure = {
        "id": "fig:demo",
        "caption": "Demo figure",
        "src_ref": "images/original.png",
        "defined_in": "sec:1",
        "referenced_in": ["sec:1"],
    }
    write_json(fig_json, figure)
    write_json(
        digest / "paper.json",
        {
            "schema_version": "0.4",
            "paper_id": "paper-a",
            "figures": [dict(figure)],
        },
    )
    asset.write_bytes(b"not-a-real-png-but-good-enough-for-hash")
    return digest, fig_json, asset


def test_defaults_use_selected_vlm_and_large_output_budget():
    cfg = extract_figure_content.Config()

    assert cfg.model == "qwen/qwen3-vl-8b-instruct"
    assert cfg.backend == "direct-vlm"
    assert cfg.codex_extractor_model == "gpt-5.4-mini"
    assert cfg.max_tokens == 12000
    assert cfg.max_image_bytes == 4_000_000
    assert cfg.local_text_gate is True
    assert cfg.tesseract_timeout == 45


def test_load_openrouter_key_caches_successful_preflight(monkeypatch):
    calls = []

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(extract_figure_content, "_load_project_env_key", lambda: None)
    monkeypatch.setattr(
        extract_figure_content,
        "_key_live",
        lambda key: calls.append(key) or True,
    )
    extract_figure_content._VERIFIED_OPENROUTER_KEYS.clear()
    cfg = extract_figure_content.Config(key_preflight=True)

    assert extract_figure_content.load_openrouter_key(cfg) == "test-key"
    assert extract_figure_content.load_openrouter_key(cfg) == "test-key"

    assert calls == ["test-key"]


def test_resolve_figure_asset_prefers_sibling(tmp_path):
    _, fig_json, asset = make_digest(tmp_path)
    figure = read_json(fig_json)

    resolved = extract_figure_content.resolve_figure_asset(fig_json, figure)

    assert resolved == asset


def test_build_codex_command_includes_images_and_prompt(tmp_path):
    cfg = extract_figure_content.Config(
        codex_bin="codex",
        codex_model="gpt-test",
        codex_config=("model_context_window=128000",),
    )
    image = tmp_path / "figure.png"
    out = tmp_path / "planner.txt"

    cmd = extract_figure_content.build_codex_command(cfg, [image], out)

    assert cmd[:8] == [
        "codex",
        "-a",
        "never",
        "exec",
        "-C",
        str(extract_figure_content.ROOT),
        "-s",
        "read-only",
    ]
    assert ["-c", "model_context_window=128000"] == cmd[8:10]
    assert ["-m", "gpt-test"] == cmd[10:12]
    assert ["--output-last-message", str(out)] == cmd[12:14]
    assert f"--image={image}" in cmd
    assert cmd[-1] == "-"


def test_run_codex_planner_uses_configured_timeout(monkeypatch, tmp_path):
    image = tmp_path / "figure.png"
    image.write_bytes(b"png-bytes")
    captured = {}

    class FakeProc:
        returncode = 0
        stdout = '{"figure_role":"chart","visual_focus":"axes","vlm_prompt":"Extract text."}'
        stderr = ""

    def fake_run(cmd, input, check, text, capture_output, timeout):
        captured["timeout"] = timeout
        return FakeProc()

    monkeypatch.setattr(extract_figure_content.subprocess, "run", fake_run)
    cfg = extract_figure_content.Config(codex_timeout=321)

    planner = extract_figure_content.run_codex_planner(cfg, {"id": "fig:test"}, [image])

    assert captured["timeout"] == 321
    assert planner["vlm_prompt"] == "Extract text."


def test_extraction_prompts_request_compact_nonduplicative_output():
    figure = {"id": "fig:test", "caption": "Demo"}

    direct_prompt = extract_figure_content.build_direct_vlm_prompt(figure)
    codex_prompt = extract_figure_content.build_codex_extractor_prompt(figure, image_count=1)

    assert "Merge nearby lines" in direct_prompt
    assert "at most 5 high-signal items" in direct_prompt
    assert "Avoid duplicate visible_text entries" in codex_prompt
    assert "model failure effects" in codex_prompt


def test_extract_json_object_keeps_valid_json():
    payload = extract_figure_content.extract_json_object(
        '{"status":"ok","figure_role":"chart","visible_text":[]}'
    )

    assert payload["status"] == "ok"
    assert payload["figure_role"] == "chart"


def test_extract_json_object_repairs_missing_commas():
    payload = extract_figure_content.extract_json_object(
        '{\n'
        '  "status": "ok"\n'
        '  "figure_role": "typographic_attack"\n'
        '  "visible_text": [\n'
        '    {"text": "A", "region": "left"}\n'
        '    {"text": "B", "region": "right"}\n'
        "  ]\n"
        '  "safety_relevant_content": []\n'
        '  "summary": "demo"\n'
        '  "uncertain": []\n'
        "}\n"
    )

    assert payload["figure_role"] == "typographic_attack"
    assert [item["text"] for item in payload["visible_text"]] == ["A", "B"]


def test_normalize_visual_record_adds_required_metadata(tmp_path):
    _, fig_json, asset = make_digest(tmp_path)
    figure = read_json(fig_json)
    cfg = extract_figure_content.Config(model="qwen/qwen3-vl-32b-instruct")

    record = extract_figure_content.normalize_visual_record(
        {
            "status": "ok",
            "figure_role": "jailbreak_prompt",
            "visible_text": [{"text": "Ignore previous instructions", "role": "prompt"}],
            "safety_relevant_content": [{"kind": "injected_instruction", "text": "Ignore..."}],
            "summary": "A prompt injection appears in the figure.",
        },
        figure,
        asset,
        cfg,
        {"visual_focus": "prompt panel"},
    )

    assert record["schema_version"] == "figure_visual.v0.1"
    assert record["status"] == "ok"
    assert record["asset_path"] == "figures/fig-demo.png"
    assert record["asset_sha256"]
    assert record["model"] == "qwen/qwen3-vl-32b-instruct"
    assert record["backend"] == "direct-vlm"
    assert record["codex_cli_model"] == "default"
    assert record["figure_role"] == "jailbreak_prompt"
    assert record["visible_text"][0]["text"].startswith("Ignore")


def test_normalize_visual_record_requires_visible_text_for_ok(tmp_path):
    _, fig_json, asset = make_digest(tmp_path)
    figure = read_json(fig_json)

    record = extract_figure_content.normalize_visual_record(
        {
            "status": "ok",
            "figure_role": "adversarial_noise",
            "visible_text": [],
            "safety_relevant_content": [
                {"kind": "target_query", "text": "inferred visual attack effect"}
            ],
            "summary": "The image causes a model failure but contains no readable text.",
        },
        figure,
        asset,
        extract_figure_content.Config(),
        {"visual_focus": "adversarial image"},
    )

    assert record["status"] == "no_visible_text"
    assert record["visible_text"] == []
    assert record["safety_relevant_content"] == []
    assert "normalized_ok_without_visible_text_to_no_visible_text" in record["warnings"]


def test_normalize_visual_record_compacts_model_outputs(tmp_path):
    _, fig_json, asset = make_digest(tmp_path)
    figure = read_json(fig_json)
    long_response = "Generated unsafe response. " * 80

    record = extract_figure_content.normalize_visual_record(
        {
            "status": "ok",
            "figure_role": "attack_pipeline",
            "visible_text": [
                {"text": "Visible user prompt", "role": "prompt"},
                {"text": long_response, "role": "model_response"},
            ],
            "safety_relevant_content": [
                {"kind": "target_query", "text": "Visible user prompt"},
                {
                    "kind": "injected_instruction",
                    "text": long_response,
                    "notes": "Harmful output generated under adversarial visual input.",
                },
                {"kind": "model_response", "text": long_response},
            ],
            "summary": "Chat transcript in a figure.",
        },
        figure,
        asset,
        extract_figure_content.Config(),
        {"visual_focus": "chat panels"},
    )

    assert record["status"] == "ok"
    assert record["visible_text"][1]["role"] == "model_response"
    assert record["visible_text"][1]["truncated"] is True
    assert len(record["visible_text"][1]["text"]) < len(long_response)
    assert [item["kind"] for item in record["safety_relevant_content"]] == ["target_query"]


def test_normalize_visual_record_deduplicates_and_caps_safety_items(tmp_path):
    _, fig_json, asset = make_digest(tmp_path)
    figure = read_json(fig_json)

    record = extract_figure_content.normalize_visual_record(
        {
            "status": "ok",
            "figure_role": "attack_pipeline",
            "visible_text": [
                {"text": "Visible prompt", "role": "prompt"},
                {"text": "Visible   prompt", "role": "prompt"},
                {"text": "Visible label", "role": "label"},
            ],
            "safety_relevant_content": [
                {"kind": "target_query", "text": "Visible prompt"},
                {"kind": "target_query", "text": "Visible   prompt"},
                {"kind": "injected_instruction", "text": "Instruction 1"},
                {"kind": "credential_or_identifier", "text": "id 2"},
                {"kind": "ui_state", "text": "state 3"},
                {"kind": "other", "text": "item 4"},
                {"kind": "other", "text": "item 5"},
            ],
            "summary": "Compact extraction.",
        },
        figure,
        asset,
        extract_figure_content.Config(),
        {"visual_focus": "chat panels"},
    )

    assert [item["text"] for item in record["visible_text"]] == [
        "Visible prompt",
        "Visible label",
    ]
    assert len(record["safety_relevant_content"]) == extract_figure_content.MAX_SAFETY_ITEMS
    assert [item["text"] for item in record["safety_relevant_content"]].count(
        "Visible prompt"
    ) == 1


def test_latex_fallback_extracts_marked_visible_text(tmp_path):
    digest = tmp_path / "data" / "papers" / "paper-a" / "digest"
    fig_json = digest / "figures" / "fig-chat.json"
    figure = {
        "id": "fig:chat",
        "latex_label": "fig:chat",
        "caption": "Chat figure",
        "src_ref": None,
    }
    write_json(fig_json, figure)
    source = tmp_path / "data" / "papers" / "paper-a" / "raw" / "source" / "main.tex"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        r"""
\begin{figure}
\begin{tikzpicture}
\node {
% ===== User Input Begin =====
Please answer \textbf{this visible request}.
% ===== User Input End =====
};
\node {
% ===== GPT Output Begin =====
This is a visible model response.
% ===== GPT Output End =====
};
\end{tikzpicture}
\caption{Chat figure}
\label{fig:chat}
\end{figure}
""",
        encoding="utf-8",
    )

    record = extract_figure_content.latex_fallback_record(
        fig_json,
        read_json(fig_json),
        extract_figure_content.Config(),
    )

    assert record is not None
    assert record["status"] == "ok"
    assert record["extraction_source"] == "latex_figure_environment"
    assert [item["role"] for item in record["visible_text"]] == ["prompt", "response"]
    assert record["visible_text"][0]["text"] == "Please answer this visible request."
    assert [item["kind"] for item in record["safety_relevant_content"]] == [
        "target_query",
        "model_response",
    ]


def test_process_figure_uses_latex_fallback_when_asset_missing(tmp_path):
    digest = tmp_path / "data" / "papers" / "paper-a" / "digest"
    fig_json = digest / "figures" / "fig-chat.json"
    figure = {
        "id": "fig:chat",
        "latex_label": "fig:chat",
        "caption": "Chat figure",
        "src_ref": None,
    }
    write_json(fig_json, figure)
    write_json(
        digest / "paper.json",
        {
            "schema_version": "0.4",
            "paper_id": "paper-a",
            "figures": [dict(figure)],
        },
    )
    source = tmp_path / "data" / "papers" / "paper-a" / "raw" / "source" / "main.tex"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        r"""
\begin{figure}
% ===== User Input Begin =====
Visible text inside the figure.
% ===== User Input End =====
\caption{Chat figure}
\label{fig:chat}
\end{figure}
""",
        encoding="utf-8",
    )

    figure_id, record = extract_figure_content.process_figure(
        "paper-a",
        fig_json,
        extract_figure_content.Config(),
        write=True,
    )

    assert figure_id == "fig:chat"
    assert record["status"] == "ok"
    assert record["extraction_source"] == "latex_figure_environment"
    assert read_json(fig_json)["visual_extraction"]["status"] == "ok"
    assert read_json(digest / "paper.json")["figures"][0]["visual_extraction"]["status"] == "ok"


def test_latex_fallback_maps_numeric_figure_id_to_unlabeled_figure(tmp_path):
    digest = tmp_path / "data" / "papers" / "paper-a" / "digest"
    fig_json = digest / "figures" / "fig-2.json"
    figure = {"id": "fig:2", "caption": "", "src_ref": None, "defined_in": "sec:1"}
    write_json(fig_json, figure)
    source = tmp_path / "data" / "papers" / "paper-a" / "raw" / "source" / "main.tex"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        r"""
% \begin{figure}
% commented out and should not count
% \end{figure}
\begin{figure}
\begin{mybox}{\textbf{First Prompt}}
First visible box.
\end{mybox}
\end{figure}
\begin{figure*}
\begin{mybox}{\textbf{Second System Prompt}}
Second visible box.
\end{mybox}
\end{figure*}
""",
        encoding="utf-8",
    )

    record = extract_figure_content.latex_fallback_record(
        fig_json,
        read_json(fig_json),
        extract_figure_content.Config(),
    )

    assert record is not None
    assert record["status"] == "ok"
    assert record["visible_text"][0]["text"].startswith("Second System Prompt")
    assert "Second visible box." in record["visible_text"][0]["text"]


def test_write_visual_extraction_updates_figure_and_paper_json(tmp_path):
    _, fig_json, _ = make_digest(tmp_path)
    record = {
        "schema_version": "figure_visual.v0.1",
        "status": "no_visible_text",
        "visible_text": [],
        "safety_relevant_content": [],
    }

    extract_figure_content.write_visual_extraction(fig_json, "fig:demo", record)

    assert read_json(fig_json)["visual_extraction"]["status"] == "no_visible_text"
    paper = read_json(fig_json.parents[1] / "paper.json")
    assert paper["figures"][0]["visual_extraction"]["status"] == "no_visible_text"


def test_write_visual_extraction_updates_duplicate_paper_entries(tmp_path):
    _, fig_json, _ = make_digest(tmp_path)
    paper_json = fig_json.parents[1] / "paper.json"
    paper = read_json(paper_json)
    paper["figures"].append(dict(paper["figures"][0]))
    write_json(paper_json, paper)
    record = {
        "schema_version": "figure_visual.v0.1",
        "status": "ok",
        "visible_text": [{"text": "Visible label"}],
        "safety_relevant_content": [],
    }

    extract_figure_content.write_visual_extraction(fig_json, "fig:demo", record)

    paper = read_json(paper_json)
    assert [fig["visual_extraction"]["status"] for fig in paper["figures"]] == ["ok", "ok"]


def test_resume_skips_failed_record_unless_retry_failed(tmp_path):
    _, fig_json, asset = make_digest(tmp_path)
    failed = {
        "schema_version": "figure_visual.v0.1",
        "status": "vlm_failed",
        "asset_sha256": extract_figure_content.sha256_file(asset),
    }
    payload = read_json(fig_json)
    payload["visual_extraction"] = failed
    write_json(fig_json, payload)

    assert extract_figure_content.existing_record_matches(fig_json, asset)
    assert not extract_figure_content.existing_record_matches(fig_json, asset, retry_failed=True)


def test_select_figures_resume_skips_failed_by_default(tmp_path, monkeypatch):
    _, fig_json, asset = make_digest(tmp_path)
    failed = {
        "schema_version": "figure_visual.v0.1",
        "status": "vlm_failed",
        "asset_sha256": extract_figure_content.sha256_file(asset),
    }
    payload = read_json(fig_json)
    payload["visual_extraction"] = failed
    write_json(fig_json, payload)
    monkeypatch.setattr(extract_figure_content, "figure_json_paths", lambda slug: [fig_json])

    assert extract_figure_content.select_figures("paper-a", None, resume=True) == []
    assert extract_figure_content.select_figures(
        "paper-a",
        None,
        resume=True,
        retry_failed=True,
    ) == [fig_json]


def test_select_figures_retry_failed_without_resume_only_selects_failures(tmp_path, monkeypatch):
    digest, fig_json, asset = make_digest(tmp_path)
    ok_fig = digest / "figures" / "fig-ok.json"
    ok_asset = digest / "figures" / "fig-ok.png"
    missing_fig = digest / "figures" / "fig-missing.json"
    ok_asset.write_bytes(b"ok-image")
    write_json(
        ok_fig,
        {
            "id": "fig:ok",
            "caption": "Already extracted",
            "visual_extraction": {
                "schema_version": "figure_visual.v0.1",
                "status": "ok",
                "asset_sha256": extract_figure_content.sha256_file(ok_asset),
            },
        },
    )
    write_json(missing_fig, {"id": "fig:missing", "caption": "Not extracted yet"})
    payload = read_json(fig_json)
    payload["visual_extraction"] = {
        "schema_version": "figure_visual.v0.1",
        "status": "vlm_failed",
        "asset_sha256": extract_figure_content.sha256_file(asset),
    }
    write_json(fig_json, payload)
    monkeypatch.setattr(
        extract_figure_content,
        "figure_json_paths",
        lambda slug: [ok_fig, fig_json, missing_fig],
    )

    assert extract_figure_content.select_figures(
        "paper-a",
        None,
        resume=False,
        retry_failed=True,
    ) == [fig_json]


def test_resume_skips_latex_fallback_success_without_asset(tmp_path):
    digest = tmp_path / "data" / "papers" / "paper-a" / "digest"
    fig_json = digest / "figures" / "fig-chat.json"
    payload = {
        "id": "fig:chat",
        "latex_label": "fig:chat",
        "caption": "Chat figure",
        "src_ref": None,
        "visual_extraction": {
            "schema_version": "figure_visual.v0.1",
            "status": "ok",
            "extraction_source": "latex_figure_environment",
            "visible_text": [{"text": "Visible text"}],
            "safety_relevant_content": [],
        },
    }
    write_json(fig_json, payload)

    assert extract_figure_content.existing_record_matches(fig_json, None)


def test_normalize_image_if_needed_respects_upload_byte_limit(tmp_path):
    Image = pytest.importorskip("PIL.Image")

    src = tmp_path / "large-bytes.png"
    pixels = bytearray()
    for y in range(512):
        for x in range(512):
            pixels.extend(((x * 17 + y * 3) % 256, (x * 7 + y * 19) % 256, (x + y) % 256))
    Image.frombytes("RGB", (512, 512), bytes(pixels)).save(src)

    out = extract_figure_content.normalize_image_if_needed(
        src,
        tmp_path / "work",
        max_long_side=1024,
        max_image_bytes=80_000,
    )

    assert out != src
    assert out.suffix == ".jpg"
    assert out.stat().st_size <= 80_000


def test_normalize_image_restores_pillow_pixel_guard(tmp_path):
    Image = pytest.importorskip("PIL.Image")

    src = tmp_path / "image.png"
    Image.new("RGB", (16, 16), "white").save(src)
    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = 12345
    try:
        extract_figure_content.normalize_image_if_needed(
            src,
            tmp_path / "work",
            max_long_side=8,
            max_image_bytes=80_000,
        )
        assert Image.MAX_IMAGE_PIXELS == 12345
    finally:
        Image.MAX_IMAGE_PIXELS = previous


def test_openrouter_payload_uses_configured_max_tokens(monkeypatch, tmp_path):
    image = tmp_path / "figure.png"
    image.write_bytes(b"png-bytes")
    captured = {}

    class FakeResponse:
        def read(self):
            return (
                b'{"choices":[{"message":{"content":"{\\"status\\":\\"no_visible_text\\",'
                b'\\"visible_text\\":[],\\"safety_relevant_content\\":[],'
                b'\\"summary\\":\\"\\",\\"uncertain\\":[]}"}}]}'
            )

    def fake_urlopen(request, timeout):
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr(extract_figure_content.urllib.request, "urlopen", fake_urlopen)
    cfg = extract_figure_content.Config(max_tokens=16000, request_timeout=9, retries=0)

    result = extract_figure_content.call_openrouter_vlm(cfg, "test-key", "Extract JSON.", [image])

    assert result["status"] == "no_visible_text"
    assert captured["timeout"] == 9
    assert captured["body"]["model"] == "qwen/qwen3-vl-8b-instruct"
    assert captured["body"]["max_tokens"] == 16000
    assert captured["body"]["messages"][1]["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )


def test_openrouter_uses_configured_wall_timeout(monkeypatch, tmp_path):
    image = tmp_path / "figure.png"
    image.write_bytes(b"png-bytes")
    captured = {}

    class FakeResponse:
        def read(self):
            return (
                b'{"choices":[{"message":{"content":"{\\"status\\":\\"no_visible_text\\",'
                b'\\"visible_text\\":[],\\"safety_relevant_content\\":[],'
                b'\\"summary\\":\\"\\",\\"uncertain\\":[]}"}}]}'
            )

    @contextmanager
    def fake_wall_clock_timeout(seconds, label):
        captured["wall_timeout"] = seconds
        captured["wall_label"] = label
        yield

    def fake_urlopen(request, timeout):
        captured["socket_timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(extract_figure_content, "wall_clock_timeout", fake_wall_clock_timeout)
    monkeypatch.setattr(extract_figure_content.urllib.request, "urlopen", fake_urlopen)
    cfg = extract_figure_content.Config(
        request_timeout=30,
        request_wall_timeout=7,
        retries=0,
    )

    result = extract_figure_content.call_openrouter_vlm(cfg, "test-key", "Extract JSON.", [image])

    assert result["status"] == "no_visible_text"
    assert captured["socket_timeout"] == 30
    assert captured["wall_timeout"] == 7
    assert captured["wall_label"] == "OpenRouter response"


def test_openrouter_empty_content_reports_finish_reason(monkeypatch, tmp_path):
    image = tmp_path / "figure.png"
    image.write_bytes(b"png-bytes")

    class FakeResponse:
        def read(self):
            return (
                b'{"choices":[{"finish_reason":"content_filter",'
                b'"message":{"content":null,"refusal":"blocked by provider",'
                b'"reasoning":"{\\"text\\":\\"SECRET_VISIBLE_TEXT\\"}"}}]}'
            )

    def fake_urlopen(request, timeout):  # noqa: ARG001
        return FakeResponse()

    monkeypatch.setattr(extract_figure_content.urllib.request, "urlopen", fake_urlopen)
    cfg = extract_figure_content.Config(retries=0)

    with pytest.raises(RuntimeError) as excinfo:
        extract_figure_content.call_openrouter_vlm(cfg, "test-key", "Extract JSON.", [image])

    message = str(excinfo.value)
    assert "content was empty" in message
    assert "content_filter" in message
    assert "blocked by provider" in message
    assert "reasoning_present" in message
    assert "SECRET_VISIBLE_TEXT" not in message


def test_openrouter_retries_transient_http_error(monkeypatch, tmp_path):
    image = tmp_path / "figure.png"
    image.write_bytes(b"png-bytes")
    calls = {"count": 0}
    sleeps = []

    class FakeResponse:
        def read(self):
            return (
                b'{"choices":[{"message":{"content":"{\\"status\\":\\"ok\\",'
                b'\\"visible_text\\":[{\\"text\\":\\"Attack pipeline\\"}],'
                b'\\"safety_relevant_content\\":[],'
                b'\\"summary\\":\\"visible text\\",\\"uncertain\\":[]}"}}]}'
            )

    def fake_urlopen(request, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            raise urllib.error.HTTPError(
                request.full_url,
                429,
                "rate limited",
                hdrs=None,
                fp=io.BytesIO(b'{"error":{"message":"temporarily rate-limited upstream"}}'),
            )
        return FakeResponse()

    monkeypatch.setattr(extract_figure_content.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(extract_figure_content.time, "sleep", lambda seconds: sleeps.append(seconds))
    cfg = extract_figure_content.Config(retries=1, retry_sleep_base=3.0)

    result = extract_figure_content.call_openrouter_vlm(cfg, "test-key", "Extract JSON.", [image])

    assert result["status"] == "ok"
    assert calls["count"] == 2
    assert sleeps == [3.0]


def test_process_figure_direct_vlm_skips_codex_planner(monkeypatch, tmp_path):
    _, fig_json, _ = make_digest(tmp_path)
    cfg = extract_figure_content.Config(backend="direct-vlm")

    monkeypatch.setattr(
        extract_figure_content,
        "prepare_figure_images",
        lambda asset, slug, figure_id, cfg: ([asset], []),
    )

    def fail_planner(*args, **kwargs):
        raise AssertionError("direct-vlm should not call Codex planner")

    monkeypatch.setattr(extract_figure_content, "run_codex_planner", fail_planner)
    monkeypatch.setattr(extract_figure_content, "load_openrouter_key", lambda cfg: "test-key")
    monkeypatch.setattr(
        extract_figure_content,
        "call_openrouter_vlm",
        lambda cfg, key, prompt, images: {
            "status": "ok",
            "figure_role": "chart",
            "visible_text": [{"text": "Visible label", "role": "label"}],
            "safety_relevant_content": [],
            "summary": "The figure contains a visible label.",
        },
    )

    _, record = extract_figure_content.process_figure("paper-a", fig_json, cfg, write=False)

    assert record["status"] == "ok"
    assert record["backend"] == "direct-vlm"
    assert record["model"] == "qwen/qwen3-vl-8b-instruct"


def test_process_figure_local_text_gate_skips_vlm_when_ocr_finds_no_text(
    monkeypatch, tmp_path
):
    _, fig_json, _ = make_digest(tmp_path)
    cfg = extract_figure_content.Config(backend="direct-vlm")

    monkeypatch.setattr(
        extract_figure_content,
        "prepare_figure_images",
        lambda asset, slug, figure_id, cfg: ([asset], []),
    )
    monkeypatch.setattr(
        extract_figure_content,
        "_tesseract_text",
        lambda image, timeout: ("", None),
    )

    def fail_vlm(*args, **kwargs):
        raise AssertionError("local no-text gate should skip the VLM")

    monkeypatch.setattr(extract_figure_content, "load_openrouter_key", fail_vlm)
    monkeypatch.setattr(extract_figure_content, "call_openrouter_vlm", fail_vlm)

    _, record = extract_figure_content.process_figure("paper-a", fig_json, cfg, write=False)

    assert record["status"] == "no_visible_text"
    assert record["visible_text"] == []
    assert record["safety_relevant_content"] == []
    assert "local_text_gate_no_readable_text" in record["warnings"]


def test_local_text_gate_skips_visual_only_attack_with_weak_labels(monkeypatch, tmp_path):
    _, fig_json, _ = make_digest(tmp_path)
    payload = read_json(fig_json)
    payload["caption"] = "A visual adversarial example perturbs the image itself."
    write_json(fig_json, payload)
    cfg = extract_figure_content.Config(backend="direct-vlm")

    monkeypatch.setattr(
        extract_figure_content,
        "prepare_figure_images",
        lambda asset, slug, figure_id, cfg: ([asset], []),
    )
    monkeypatch.setattr(
        extract_figure_content,
        "_tesseract_text",
        lambda image, timeout: ("Original image Adversarial image", None),
    )

    def fail_vlm(*args, **kwargs):
        raise AssertionError("visual-only weak labels should skip the VLM")

    monkeypatch.setattr(extract_figure_content, "load_openrouter_key", fail_vlm)
    monkeypatch.setattr(extract_figure_content, "call_openrouter_vlm", fail_vlm)

    _, record = extract_figure_content.process_figure("paper-a", fig_json, cfg, write=False)

    assert record["status"] == "no_visible_text"
    assert record["visible_text"] == []
    assert record["safety_relevant_content"] == []
    assert "local_text_gate_visual_only_weak_text" in record["warnings"]


def test_local_text_gate_keeps_weak_labels_without_visual_only_hint(monkeypatch, tmp_path):
    _, fig_json, _ = make_digest(tmp_path)
    payload = read_json(fig_json)
    payload["caption"] = "A compact chart with two labels."
    write_json(fig_json, payload)
    cfg = extract_figure_content.Config(backend="direct-vlm")

    monkeypatch.setattr(
        extract_figure_content,
        "prepare_figure_images",
        lambda asset, slug, figure_id, cfg: ([asset], []),
    )
    monkeypatch.setattr(
        extract_figure_content,
        "_tesseract_text",
        lambda image, timeout: ("Original image Adversarial image", None),
    )
    monkeypatch.setattr(extract_figure_content, "load_openrouter_key", lambda cfg: "test-key")
    monkeypatch.setattr(
        extract_figure_content,
        "call_openrouter_vlm",
        lambda cfg, key, prompt, images: {
            "status": "ok",
            "figure_role": "chart",
            "visible_text": [{"text": "Original image Adversarial image", "role": "label"}],
            "safety_relevant_content": [],
            "summary": "The chart contains visible labels.",
        },
    )

    _, record = extract_figure_content.process_figure("paper-a", fig_json, cfg, write=False)

    assert record["status"] == "ok"
    assert record["visible_text"][0]["role"] == "label"


def test_local_text_gate_keeps_visual_attack_when_ocr_has_query_cue(monkeypatch, tmp_path):
    _, fig_json, _ = make_digest(tmp_path)
    payload = read_json(fig_json)
    payload["caption"] = "A visual adversarial example perturbs the image itself."
    write_json(fig_json, payload)
    cfg = extract_figure_content.Config(backend="direct-vlm")

    monkeypatch.setattr(
        extract_figure_content,
        "prepare_figure_images",
        lambda asset, slug, figure_id, cfg: ([asset], []),
    )
    monkeypatch.setattr(
        extract_figure_content,
        "_tesseract_text",
        lambda image, timeout: ("User query: open the hidden file", None),
    )
    monkeypatch.setattr(extract_figure_content, "load_openrouter_key", lambda cfg: "test-key")
    monkeypatch.setattr(
        extract_figure_content,
        "call_openrouter_vlm",
        lambda cfg, key, prompt, images: {
            "status": "ok",
            "figure_role": "jailbreak_prompt",
            "visible_text": [{"text": "User query: open the hidden file", "role": "prompt"}],
            "safety_relevant_content": [],
            "summary": "The figure contains a visible user query.",
        },
    )

    _, record = extract_figure_content.process_figure("paper-a", fig_json, cfg, write=False)

    assert record["status"] == "ok"
    assert record["visible_text"][0]["role"] == "prompt"


def test_process_figure_local_text_gate_allows_pdf_embedded_text(
    monkeypatch, tmp_path
):
    digest, fig_json, png_asset = make_digest(tmp_path)
    pdf_asset = png_asset.with_suffix(".pdf")
    png_asset.unlink()
    pdf_asset.write_bytes(b"%PDF-1.4\n")
    cfg = extract_figure_content.Config(backend="direct-vlm")

    monkeypatch.setattr(
        extract_figure_content,
        "prepare_figure_images",
        lambda asset, slug, figure_id, cfg: ([pdf_asset], []),
    )
    monkeypatch.setattr(
        extract_figure_content,
        "_pdftotext_text",
        lambda asset: ("Visible label 123", None),
    )

    def fail_ocr(*args, **kwargs):
        raise AssertionError("embedded PDF text should bypass OCR")

    monkeypatch.setattr(extract_figure_content, "_tesseract_text", fail_ocr)
    monkeypatch.setattr(extract_figure_content, "load_openrouter_key", lambda cfg: "test-key")
    monkeypatch.setattr(
        extract_figure_content,
        "call_openrouter_vlm",
        lambda cfg, key, prompt, images: {
            "status": "ok",
            "figure_role": "chart",
            "visible_text": [{"text": "Visible label", "role": "label"}],
            "safety_relevant_content": [],
            "summary": "The figure contains readable PDF text.",
        },
    )

    _, record = extract_figure_content.process_figure("paper-a", fig_json, cfg, write=False)

    assert digest.exists()
    assert record["status"] == "ok"
    assert record["visible_text"][0]["text"] == "Visible label"


def test_process_figure_local_text_gate_can_be_disabled(monkeypatch, tmp_path):
    _, fig_json, _ = make_digest(tmp_path)
    cfg = extract_figure_content.Config(backend="direct-vlm", local_text_gate=False)

    monkeypatch.setattr(
        extract_figure_content,
        "prepare_figure_images",
        lambda asset, slug, figure_id, cfg: ([asset], []),
    )

    def fail_probe(*args, **kwargs):
        raise AssertionError("disabled local text gate should not run OCR")

    monkeypatch.setattr(extract_figure_content, "_tesseract_text", fail_probe)
    monkeypatch.setattr(extract_figure_content, "load_openrouter_key", lambda cfg: "test-key")
    monkeypatch.setattr(
        extract_figure_content,
        "call_openrouter_vlm",
        lambda cfg, key, prompt, images: {
            "status": "ok",
            "figure_role": "chart",
            "visible_text": [{"text": "VLM text", "role": "label"}],
            "safety_relevant_content": [],
            "summary": "The VLM path ran.",
        },
    )

    _, record = extract_figure_content.process_figure("paper-a", fig_json, cfg, write=False)

    assert record["status"] == "ok"
    assert record["visible_text"][0]["text"] == "VLM text"


def test_process_figure_codex_cli_backend_uses_codex_extractor(monkeypatch, tmp_path):
    _, fig_json, _ = make_digest(tmp_path)
    cfg = extract_figure_content.Config(
        backend="codex-cli",
        codex_extractor_model="gpt-5.4-mini",
    )

    monkeypatch.setattr(
        extract_figure_content,
        "prepare_figure_images",
        lambda asset, slug, figure_id, cfg: ([asset], []),
    )
    monkeypatch.setattr(
        extract_figure_content,
        "run_codex_extractor",
        lambda cfg, figure, images: {
            "status": "ok",
            "figure_role": "chart",
            "visible_text": [{"text": "Visible axis", "role": "axis"}],
            "safety_relevant_content": [],
            "summary": "The figure contains a visible axis.",
        },
    )

    _, record = extract_figure_content.process_figure("paper-a", fig_json, cfg, write=False)

    assert record["status"] == "ok"
    assert record["backend"] == "codex-cli"
    assert record["model"] == "gpt-5.4-mini"
    assert record["codex_cli_model"] == "gpt-5.4-mini"


def test_process_figure_direct_vlm_falls_back_to_codex_cli(monkeypatch, tmp_path):
    _, fig_json, _ = make_digest(tmp_path)
    cfg = extract_figure_content.Config(
        backend="direct-vlm",
        fallback_backend="codex-cli",
        fallback_codex_extractor_model="gpt-5.4-mini",
    )

    monkeypatch.setattr(
        extract_figure_content,
        "prepare_figure_images",
        lambda asset, slug, figure_id, cfg: ([asset], []),
    )
    monkeypatch.setattr(extract_figure_content, "load_openrouter_key", lambda cfg: "test-key")

    def fail_vlm(cfg, key, prompt, images):
        raise RuntimeError("primary timeout")

    monkeypatch.setattr(extract_figure_content, "call_openrouter_vlm", fail_vlm)
    monkeypatch.setattr(
        extract_figure_content,
        "run_codex_extractor",
        lambda cfg, figure, images: {
            "status": "ok",
            "figure_role": "chart",
            "visible_text": [{"text": "Fallback text", "role": "label"}],
            "safety_relevant_content": [],
            "summary": "The fallback extractor found visible text.",
        },
    )

    _, record = extract_figure_content.process_figure("paper-a", fig_json, cfg, write=False)

    assert record["status"] == "ok"
    assert record["backend"] == "codex-cli"
    assert record["model"] == "gpt-5.4-mini"
    assert record["visible_text"][0]["text"] == "Fallback text"
    assert record["fallback_from"]["backend"] == "direct-vlm"
    assert record["fallback_from"]["status"] == "vlm_failed"


def test_process_figure_no_visible_text_does_not_fallback_by_default(monkeypatch, tmp_path):
    _, fig_json, _ = make_digest(tmp_path)
    cfg = extract_figure_content.Config(
        backend="direct-vlm",
        fallback_backend="codex-cli",
    )

    monkeypatch.setattr(
        extract_figure_content,
        "prepare_figure_images",
        lambda asset, slug, figure_id, cfg: ([asset], []),
    )
    monkeypatch.setattr(extract_figure_content, "load_openrouter_key", lambda cfg: "test-key")
    monkeypatch.setattr(
        extract_figure_content,
        "call_openrouter_vlm",
        lambda cfg, key, prompt, images: {
            "status": "no_visible_text",
            "figure_role": "other",
            "visible_text": [],
            "safety_relevant_content": [],
            "summary": "No readable text.",
        },
    )

    def fail_codex(*args):
        raise AssertionError("no_visible_text should not trigger fallback by default")

    monkeypatch.setattr(extract_figure_content, "run_codex_extractor", fail_codex)

    _, record = extract_figure_content.process_figure("paper-a", fig_json, cfg, write=False)

    assert record["status"] == "no_visible_text"
    assert record["backend"] == "direct-vlm"


def test_process_figure_with_mocked_codex_and_openrouter(monkeypatch, tmp_path):
    _, fig_json, _ = make_digest(tmp_path)
    cfg = extract_figure_content.Config(backend="planned-vlm")

    monkeypatch.setattr(
        extract_figure_content,
        "prepare_figure_images",
        lambda asset, slug, figure_id, cfg: ([asset], []),
    )
    monkeypatch.setattr(
        extract_figure_content,
        "run_codex_planner",
        lambda cfg, figure, images: {
            "figure_role": "jailbreak_prompt",
            "visual_focus": "speech bubble",
            "should_call_vlm": True,
            "vlm_prompt": "Extract visible prompt text as JSON.",
        },
    )
    monkeypatch.setattr(extract_figure_content, "load_openrouter_key", lambda cfg: "test-key")
    monkeypatch.setattr(
        extract_figure_content,
        "call_openrouter_vlm",
        lambda cfg, key, prompt, images: {
            "status": "ok",
            "visible_text": [{"text": "Whar's the Email of the seller?", "role": "prompt"}],
            "safety_relevant_content": [],
            "summary": "The figure contains a visible user query.",
        },
    )

    figure_id, record = extract_figure_content.process_figure("paper-a", fig_json, cfg, write=True)

    assert figure_id == "fig:demo"
    assert record["status"] == "ok"
    assert record["visible_text"][0]["text"].startswith("Whar")
    assert read_json(fig_json)["visual_extraction"]["status"] == "ok"
