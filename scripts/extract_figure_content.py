#!/usr/bin/env python3
"""Extract visible figure text into Paper2Agent digests.

This is a digest enrichment tool, not a new AgenticCarlini stage. It reads
``data/papers/<slug>/digest/figures/*.json`` plus sibling figure assets, renders
or normalizes images into ``.workspace/figure-extract/``, asks Codex CLI to
produce a figure-specific extraction prompt, and then asks an OpenRouter VLM to
return structured JSON.

Default mode is dry-run. Use ``--write`` to persist ``visual_extraction`` into
both the per-figure JSON file and ``digest/paper.json``.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import fcntl
import hashlib
import json
import mimetypes
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PAPERS = ROOT / "data" / "papers"
WORKSPACE = ROOT / ".workspace" / "figure-extract"

SCHEMA_VERSION = "figure_visual.v0.1"
DEFAULT_MODEL = "qwen/qwen3-vl-8b-instruct"
DEFAULT_BACKEND = "direct-vlm"
DEFAULT_CODEX_EXTRACTOR_MODEL = "gpt-5.4-mini"
SUPPORTED_ASSET_EXTS = {".pdf", ".png", ".jpg", ".jpeg"}
KNOWN_ASSET_EXTS = SUPPORTED_ASSET_EXTS | {".eps", ".svg"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
ALLOWED_STATUSES = {
    "ok",
    "no_visible_text",
    "asset_missing",
    "unsupported_asset",
    "render_failed",
    "codex_prompt_failed",
    "vlm_failed",
}
FAILURE_STATUSES = {
    "asset_missing",
    "unsupported_asset",
    "render_failed",
    "codex_prompt_failed",
    "vlm_failed",
}
VISIBLE_TEXT_ROLES = {
    "prompt",
    "response",
    "model_response",
    "label",
    "axis",
    "legend",
    "ui_text",
    "annotation",
    "other",
}
SAFETY_CONTENT_KINDS = {
    "jailbreak_prompt",
    "injected_instruction",
    "target_query",
    "model_response",
    "credential_or_identifier",
    "ui_state",
    "other",
}
MODEL_RESPONSE_ROLES = {"response", "model_response"}
MAX_VISIBLE_TEXT_CHARS = 2400
MAX_MODEL_RESPONSE_CHARS = 700
MAX_SAFETY_TEXT_CHARS = 1200
MAX_SAFETY_ITEMS = 5
UNREADABLE_MARKERS = {"", "[unreadable]", "<unreadable>", "unreadable"}
LOCAL_TEXT_MIN_WORDS = 3
LOCAL_TEXT_MIN_CHARS = 12
VISUAL_ONLY_WEAK_TEXT_MAX_WORDS = 12
VISUAL_ONLY_WEAK_TEXT_MAX_CHARS = 120
_VERIFIED_OPENROUTER_KEYS: set[str] = set()
_LAST_PHASE_TIMINGS: dict[str, dict[str, float]] = {}


@dataclass(frozen=True)
class Config:
    model: str = DEFAULT_MODEL
    backend: str = DEFAULT_BACKEND
    codex_model: str | None = None
    codex_extractor_model: str = DEFAULT_CODEX_EXTRACTOR_MODEL
    codex_config: tuple[str, ...] = ()
    codex_bin: str = "codex"
    codex_timeout: int = 180
    max_tokens: int = 12000
    max_pages: int = 3
    max_long_side: int = 3600
    max_image_bytes: int = 4_000_000
    render_dpi: int = 220
    request_timeout: int = 240
    request_wall_timeout: int | None = None
    retries: int = 2
    retry_sleep_base: float = 8.0
    key_preflight: bool = True
    local_text_gate: bool = True
    tesseract_timeout: int = 45
    fallback_backend: str | None = None
    fallback_model: str | None = None
    fallback_codex_extractor_model: str | None = None
    fallback_on_statuses: tuple[str, ...] = ("vlm_failed",)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@contextmanager
def slug_write_lock(slug: str):
    lock_dir = WORKSPACE / "_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{safe_id(slug)}.lock"
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        yield


def safe_id(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "figure"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@contextmanager
def wall_clock_timeout(seconds: int | None, label: str):
    if not seconds or seconds <= 0 or not hasattr(signal, "setitimer"):
        yield
        return

    def raise_timeout(signum, frame):  # noqa: ARG001
        raise TimeoutError(f"{label} exceeded {seconds}s wall-clock timeout")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, raise_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def paper_dir(slug: str) -> Path:
    return PAPERS / slug


def digest_dir(slug: str) -> Path:
    return paper_dir(slug) / "digest"


def figure_json_paths(slug: str) -> list[Path]:
    figdir = digest_dir(slug) / "figures"
    if not figdir.is_dir():
        return []
    return sorted(figdir.glob("*.json"))


def resolve_figure_asset(figure_json: Path, figure: dict[str, Any]) -> Path | None:
    """Resolve a figure asset from the digest's per-file layout.

    Prefer sibling files with the same basename because Paper2Agent copies
    assets as ``fig-<id>.<ext>``. Fall back to src_ref basename when a digest was
    produced by another writer.
    """
    for ext in sorted(KNOWN_ASSET_EXTS):
        cand = figure_json.with_suffix(ext)
        if cand.is_file():
            return cand

    # asset_files (schema >= 0.5) is authoritative when present.
    for rel in figure.get("asset_files") or []:
        cand = figure_json.parent.parent / rel
        if cand.is_file() and cand.suffix.lower() in KNOWN_ASSET_EXTS:
            return cand

    figdir = figure_json.parent
    # schema >= 0.5 stores src_refs (list); older digests store src_ref (str).
    src_refs = figure.get("src_refs") or ([figure["src_ref"]] if figure.get("src_ref") else [])
    for src_ref in src_refs:
        src_name = Path(src_ref).name
        candidates = [figdir / src_name]
        if not Path(src_name).suffix:
            candidates.extend(figdir / f"{src_name}{ext}" for ext in sorted(KNOWN_ASSET_EXTS))
        for cand in candidates:
            if cand.is_file() and cand.suffix.lower() in KNOWN_ASSET_EXTS:
                return cand

    return None


def _strip_latex_comments(text: str) -> str:
    lines: list[str] = []
    for raw in text.splitlines():
        cut = len(raw)
        escaped = False
        for idx, ch in enumerate(raw):
            if ch == "\\" and not escaped:
                escaped = True
                continue
            if ch == "%" and not escaped:
                cut = idx
                break
            escaped = False
        lines.append(raw[:cut])
    return "\n".join(lines)


def _latex_braced_command_to_text(text: str, command: str) -> str:
    pattern = f"\\{command}"
    out: list[str] = []
    i = 0
    while i < len(text):
        if text.startswith(pattern, i):
            j = i + len(pattern)
            while j < len(text) and text[j].isspace():
                j += 1
            if j < len(text) and text[j] == "{":
                depth = 0
                start = j + 1
                k = j
                while k < len(text):
                    if text[k] == "{" and (k == 0 or text[k - 1] != "\\"):
                        depth += 1
                    elif text[k] == "}" and (k == 0 or text[k - 1] != "\\"):
                        depth -= 1
                        if depth == 0:
                            out.append(clean_latex_text(text[start:k]))
                            i = k + 1
                            break
                    k += 1
                else:
                    out.append(text[i])
                    i += 1
                continue
        out.append(text[i])
        i += 1
    return "".join(out)


def clean_latex_text(text: str) -> str:
    text = _strip_latex_comments(text)
    for command in (
        "textbf",
        "textit",
        "emph",
        "underline",
        "texttt",
        "textsc",
        "needrevise",
        "url",
    ):
        for _ in range(5):
            updated = _latex_braced_command_to_text(text, command)
            if updated == text:
                break
            text = updated
    text = re.sub(r"\\(?:citep|citet|cite|ref|label|caption)\*?(?:\[[^\]]*\])?\{[^{}]*\}", " ", text)
    text = re.sub(r"\\(?:begin|end)\{[^{}]*\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", " ", text)
    text = text.replace(r"\\", "\n")
    for src, dst in {
        r"\%": "%",
        r"\$": "$",
        r"\_": "_",
        r"\&": "&",
        r"\#": "#",
        r"\{": "{",
        r"\}": "}",
        "~": " ",
    }.items():
        text = text.replace(src, dst)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def find_latex_figure_environment(
    figure_json: Path,
    figure: dict[str, Any],
) -> tuple[Path, str] | None:
    labels = [
        str(figure.get("latex_label") or "").strip(),
        str(figure.get("id") or "").removeprefix("fig:").strip(),
    ]
    labels = [label for label in labels if label]
    if not labels:
        return None

    try:
        paper_root = figure_json.parents[2]
    except IndexError:
        return None
    source_root = paper_root / "raw" / "source"
    if not source_root.is_dir():
        return None

    for tex in sorted(source_root.rglob("*.tex")):
        text = tex.read_text(encoding="utf-8", errors="replace")
        for label in labels:
            match = re.search(r"\\label\{" + re.escape(label) + r"\}", text)
            if not match:
                continue
            begin_matches = list(re.finditer(r"\\begin\{figure\*?\}", text[: match.start()]))
            if not begin_matches:
                continue
            begin = begin_matches[-1].start()
            end_match = re.search(r"\\end\{figure\*?\}", text[match.end() :])
            if not end_match:
                continue
            end = match.end() + end_match.end()
            return tex, text[begin:end]

    figure_id = str(figure.get("id") or "")
    numeric = re.fullmatch(r"fig:(\d+)", figure_id)
    if numeric:
        target_index = int(numeric.group(1))
        seen = 0
        for tex in sorted(source_root.rglob("*.tex")):
            text = _strip_latex_comments(tex.read_text(encoding="utf-8", errors="replace"))
            for match in re.finditer(r"\\begin\{figure\*?\}", text):
                end_match = re.search(r"\\end\{figure\*?\}", text[match.end() :])
                if not end_match:
                    continue
                seen += 1
                if seen == target_index:
                    end = match.end() + end_match.end()
                    return tex, text[match.start() : end]
    return None


def extract_latex_visible_segments(latex: str) -> list[dict[str, str]]:
    segments: list[dict[str, str]] = []
    marker_re = re.compile(
        r"%\s*=+\s*(?P<label>[^%\n]+?)\s+Begin\s*=+\s*\n"
        r"(?P<body>.*?)"
        r"%\s*=+\s*(?P=label)\s+End\s*=+",
        re.S,
    )
    for match in marker_re.finditer(latex):
        label = " ".join(match.group("label").strip().lower().split())
        body = clean_latex_text(match.group("body"))
        if not body:
            continue
        if "user input" in label:
            role = "prompt"
            region = "latex_marker:user_input"
        elif "gpt output" in label or "model output" in label:
            role = "response"
            region = "latex_marker:model_output"
        else:
            role = "other"
            region = f"latex_marker:{safe_id(label)}"
        segments.append({"text": body, "region": region, "role": role, "confidence": "high"})
    if segments:
        return segments

    mybox_re = re.compile(
        r"\\begin\{mybox\}(?:\{(?P<title>(?:[^{}]|\{[^{}]*\})*)\})?"
        r"(?P<body>.*?)\\end\{mybox\}",
        re.S,
    )
    for match in mybox_re.finditer(latex):
        title = clean_latex_text(match.group("title") or "")
        body = clean_latex_text(match.group("body") or "")
        if not body:
            continue
        text = f"{title}\n\n{body}".strip() if title else body
        role = "prompt" if "prompt" in title.lower() else "other"
        segments.append(
            {
                "text": text,
                "region": f"latex_mybox:{safe_id(title or 'box')}",
                "role": role,
                "confidence": "high",
            }
        )
    if segments:
        return segments

    cleaned = clean_latex_text(latex)
    cleaned = re.sub(r"\{|\}|\[|\]|\(|\)|;", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned:
        segments.append(
            {
                "text": cleaned,
                "region": "latex_figure_environment",
                "role": "other",
                "confidence": "medium",
            }
        )
    return segments


def latex_fallback_record(
    figure_json: Path,
    figure: dict[str, Any],
    cfg: Config,
) -> dict[str, Any] | None:
    found = find_latex_figure_environment(figure_json, figure)
    if not found:
        return None
    source_path, latex = found
    segments = extract_latex_visible_segments(latex)
    if not segments:
        return None

    record = base_visual_record("ok", figure, None, cfg, ["extracted_from_latex_figure_environment"])
    record["extraction_source"] = "latex_figure_environment"
    try:
        record["source_path"] = str(source_path.relative_to(ROOT))
    except ValueError:
        record["source_path"] = str(source_path)
    record["source_sha256"] = sha256_file(source_path)
    record["figure_role"] = "jailbreak_prompt" if any(
        item["role"] in {"prompt", "response"} for item in segments
    ) else "other"
    record["visible_text"] = segments
    safety: list[dict[str, str]] = []
    for item in segments:
        if item["role"] == "prompt":
            kind = "target_query"
        elif item["role"] == "response":
            kind = "model_response"
        else:
            continue
        safety.append(
            {
                "kind": kind,
                "text": item["text"],
                "confidence": item["confidence"],
                "notes": "Recovered from the paper LaTeX figure environment.",
            }
        )
    record["safety_relevant_content"] = safety
    record["summary"] = "Visible text recovered from the paper LaTeX figure environment."
    record["uncertain"] = [
        "Fallback used because no standalone figure asset was present in the digest."
    ]
    return record


def pdf_page_count(path: Path) -> int | None:
    pdfinfo = shutil.which("pdfinfo")
    if not pdfinfo:
        return None
    try:
        proc = subprocess.run(
            [pdfinfo, str(path)],
            check=False,
            text=True,
            capture_output=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    m = re.search(r"^Pages:\s+(\d+)\s*$", proc.stdout, re.M)
    return int(m.group(1)) if m else None


def _rgb_on_white(im):
    if im.mode in {"RGBA", "LA"}:
        from PIL import Image

        bg = Image.new("RGB", im.size, "white")
        bg.paste(im, mask=im.getchannel("A"))
        return bg
    return im.convert("RGB")


def normalize_image_if_needed(
    src: Path,
    dest_dir: Path,
    max_long_side: int,
    max_image_bytes: int,
) -> Path:
    """Return src unless dimensions or upload bytes require a workspace copy."""
    try:
        from PIL import Image
    except Exception:
        return src

    try:
        previous_max_pixels = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = None
        try:
            im_context = Image.open(src)
        finally:
            Image.MAX_IMAGE_PIXELS = previous_max_pixels
        with im_context as im:
            width, height = im.size
            if max(width, height) <= max_long_side and src.stat().st_size <= max_image_bytes:
                return src
            long_side = min(max_long_side, max(width, height))
            scale = long_side / max(width, height)
            size = (max(1, int(width * scale)), max(1, int(height * scale)))
            out = dest_dir / f"{src.stem}-normalized.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            image = _rgb_on_white(im).resize(size, Image.Resampling.LANCZOS)
            quality = 92
            while True:
                image.save(out, format="JPEG", quality=quality, optimize=True)
                if out.stat().st_size <= max_image_bytes:
                    return out
                if quality > 72:
                    quality -= 10
                    continue
                if max(image.size) <= 1600:
                    return out
                new_size = (
                    max(1, int(image.size[0] * 0.85)),
                    max(1, int(image.size[1] * 0.85)),
                )
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                quality = 82
            return out
    except Exception:
        return src


def render_pdf_pages(asset: Path, dest_dir: Path, cfg: Config) -> tuple[list[Path], list[str]]:
    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        raise RuntimeError("pdftoppm not found on PATH")

    dest_dir.mkdir(parents=True, exist_ok=True)
    prefix = dest_dir / asset.stem
    cmd = [
        pdftoppm,
        "-png",
        "-r",
        str(cfg.render_dpi),
        "-f",
        "1",
        "-l",
        str(cfg.max_pages),
        str(asset),
        str(prefix),
    ]
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True, timeout=120)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(detail[:500] or f"pdftoppm exited {proc.returncode}")

    rendered = sorted(dest_dir.glob(f"{asset.stem}-*.png"))
    if not rendered:
        single = prefix.with_suffix(".png")
        if single.exists():
            rendered = [single]
    if not rendered:
        raise RuntimeError("pdftoppm produced no image files")

    warnings: list[str] = []
    pages = pdf_page_count(asset)
    if pages and pages > cfg.max_pages:
        warnings.append(f"pdf_has_{pages}_pages_rendered_first_{cfg.max_pages}")
    return rendered[: cfg.max_pages], warnings


def prepare_figure_images(
    asset: Path,
    slug: str,
    figure_id: str,
    cfg: Config,
) -> tuple[list[Path], list[str]]:
    ext = asset.suffix.lower()
    work = WORKSPACE / slug / safe_id(figure_id)
    if ext == ".pdf":
        images, warnings = render_pdf_pages(asset, work, cfg)
        return [
            normalize_image_if_needed(p, work, cfg.max_long_side, cfg.max_image_bytes)
            for p in images
        ], warnings
    if ext in IMAGE_EXTS:
        return [normalize_image_if_needed(asset, work, cfg.max_long_side, cfg.max_image_bytes)], []
    raise ValueError(f"unsupported asset extension: {ext}")


def _local_text_counts(text: str) -> tuple[int, int]:
    stripped = text.strip()
    tokens = re.findall(r"""[A-Za-z0-9][A-Za-z0-9_<>%./:;,'"!?()[\]-]*""", stripped)
    words = [token for token in tokens if len(token) >= 2]
    return len(stripped), len(words)


def _has_local_text_signal(text: str) -> bool:
    chars, words = _local_text_counts(text)
    return chars >= LOCAL_TEXT_MIN_CHARS and words >= LOCAL_TEXT_MIN_WORDS


def _metadata_suggests_visual_only_attack(figure: dict[str, Any]) -> bool:
    parts = [str(figure.get(key) or "") for key in ("id", "caption", "latex_label")]
    parts.extend(str(r) for r in (figure.get("src_refs") or []))
    if figure.get("src_ref"):
        parts.append(str(figure["src_ref"]))
    text = " ".join(parts).lower()
    visual_hints = (
        "adversarial image",
        "adversarial example",
        "visual adversarial",
        "visual jailbreak",
        "visual attack",
        "perturbation",
        "perturbed image",
        "noise image",
        "image itself",
        "benign visual input",
        "visual input",
        "pixel",
    )
    text_bearing_hints = (
        "typographic",
        "prompt",
        "instruction",
        "query",
        "screenshot",
        "chat",
        "dialogue",
        "conversation",
    )
    return any(hint in text for hint in visual_hints) and not any(
        hint in text for hint in text_bearing_hints
    )


def _ocr_text_has_paper_evidence_cue(text: str) -> bool:
    return bool(
        re.search(
            r"(@|https?://|www\.|\b(prompt|instruction|query|question|answer|"
            r"user|assistant|system|response|output|email|password|token|key|"
            r"jailbreak|ignore|developer)\b)",
            text,
            flags=re.I,
        )
    )


def _should_skip_visual_only_weak_text(
    figure: dict[str, Any],
    text: str,
    chars: int,
    words: int,
) -> bool:
    return (
        _metadata_suggests_visual_only_attack(figure)
        and chars <= VISUAL_ONLY_WEAK_TEXT_MAX_CHARS
        and words <= VISUAL_ONLY_WEAK_TEXT_MAX_WORDS
        and not _ocr_text_has_paper_evidence_cue(text)
    )


def _pdftotext_text(asset: Path) -> tuple[str, str | None]:
    pdftotext = shutil.which("pdftotext")
    if not pdftotext:
        return "", "local_text_gate_pdftotext_unavailable"
    try:
        proc = subprocess.run(
            [pdftotext, "-layout", str(asset), "-"],
            check=False,
            text=True,
            capture_output=True,
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001
        return "", f"local_text_gate_pdftotext_error:{type(exc).__name__}"
    if proc.returncode != 0:
        return "", f"local_text_gate_pdftotext_exit_{proc.returncode}"
    return proc.stdout or "", None


def _tesseract_text(image: Path, timeout: int) -> tuple[str, str | None]:
    tesseract = shutil.which("tesseract")
    if not tesseract:
        return "", "local_text_gate_tesseract_unavailable"
    try:
        proc = subprocess.run(
            [tesseract, str(image), "stdout", "--psm", "6"],
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "", "local_text_gate_tesseract_timeout"
    except Exception as exc:  # noqa: BLE001
        return "", f"local_text_gate_tesseract_error:{type(exc).__name__}"
    if proc.returncode != 0:
        return "", f"local_text_gate_tesseract_exit_{proc.returncode}"
    return proc.stdout or "", None


def local_text_gate_no_visible_record(
    figure: dict[str, Any],
    asset: Path,
    images: list[Path],
    cfg: Config,
    warnings: list[str],
) -> dict[str, Any] | None:
    """Return a no-visible-text record when local OCR finds no readable text.

    This is intentionally conservative: PDFs with embedded text pass without
    OCR, and OCR/tool failures do not short-circuit the VLM path.
    """

    if not cfg.local_text_gate:
        return None

    probe_warnings = list(warnings)
    if asset.suffix.lower() == ".pdf":
        pdf_text, pdf_warning = _pdftotext_text(asset)
        if pdf_warning:
            probe_warnings.append(pdf_warning)
        chars, words = _local_text_counts(pdf_text)
        probe_warnings.append(f"local_text_gate_pdftotext_chars_{chars}_words_{words}")
        if _has_local_text_signal(pdf_text):
            return None

    saw_ocr_success = False
    ocr_chars = 0
    ocr_words = 0
    for image in images:
        text, warning = _tesseract_text(image, cfg.tesseract_timeout)
        if warning:
            probe_warnings.append(warning)
            continue
        saw_ocr_success = True
        chars, words = _local_text_counts(text)
        ocr_chars += chars
        ocr_words += words
        if _has_local_text_signal(text) and not _should_skip_visual_only_weak_text(
            figure,
            text,
            chars,
            words,
        ):
            return None

    if not saw_ocr_success:
        return None

    probe_warnings.append(f"local_text_gate_tesseract_chars_{ocr_chars}_words_{ocr_words}")
    if _metadata_suggests_visual_only_attack(figure) and ocr_words <= VISUAL_ONLY_WEAK_TEXT_MAX_WORDS:
        probe_warnings.append("local_text_gate_visual_only_weak_text")
    probe_warnings.append("local_text_gate_no_readable_text")
    return no_visible_record(
        figure,
        asset,
        cfg,
        {
            "figure_role": "other",
            "visual_focus": "local text probe",
            "no_visible_text_reason": (
                "Local PDF text/OCR probe found no readable figure text; "
                "skipped VLM extraction to avoid hallucinated visual content."
            ),
        },
        probe_warnings,
    )


def _json_loads(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(text, strict=False)


def _repair_common_json_glitches(text: str) -> str:
    repaired = text
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    repaired = re.sub(r"}\s*\n\s*{", "},\n{", repaired)
    repaired = re.sub(r"]\s*\n\s*\"", "],\n\"", repaired)
    repaired = re.sub(r"}\s*\n\s*\"", "},\n\"", repaired)
    repaired = re.sub(
        r'("(?:status|figure_role|visible_text|safety_relevant_content|summary|uncertain)"\s*:)',
        r",\1",
        repaired,
    )
    repaired = re.sub(r"{\s*,", "{", repaired)
    repaired = re.sub(r",\s*,", ",", repaired)
    return repaired


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.S)
    candidates = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidates.append(stripped[start : end + 1])
    candidates.extend(_repair_common_json_glitches(candidate) for candidate in list(candidates))
    last_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        try:
            return _json_loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
    if last_error:
        raise last_error
    raise json.JSONDecodeError("No JSON object found", stripped, 0)


def openrouter_choice_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("OpenRouter response had no choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise RuntimeError("OpenRouter first choice was not an object")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("OpenRouter first choice had no message object")
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content

    details = {
        "finish_reason": choice.get("finish_reason"),
        "refusal": message.get("refusal"),
        "annotations": message.get("annotations"),
    }
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        details["reasoning_present"] = True
        details["reasoning_chars"] = len(reasoning)
        details["reasoning_json_like"] = "{" in reasoning and "}" in reasoning
    compact = {key: value for key, value in details.items() if value}
    raise RuntimeError(
        "OpenRouter response message content was empty"
        + (f": {json.dumps(compact, ensure_ascii=False)[:800]}" if compact else "")
    )


def build_planner_prompt(figure: dict[str, Any], image_count: int) -> str:
    return (
        "You are planning visual text extraction for one academic paper figure.\n"
        "Look at the attached image(s), but do not perform final extraction. "
        "Return a single JSON object only.\n\n"
        "Required JSON fields:\n"
        "- figure_role: one of jailbreak_prompt, attack_pipeline, result_screenshot, "
        "chart, typographic_attack, diagram, adversarial_noise, other\n"
        "- visual_focus: short description of the regions/text types the extractor should inspect\n"
        "- should_call_vlm: boolean; false when you cannot read any visible characters. "
        "Do not call the VLM merely to recover hidden/steganographic/adversarial text; "
        "this tool records visible figure text, not invisible payload recovery. "
        "For photos, adversarial noise, perturbation images, screenshots, or visual attack "
        "examples with no readable characters, set should_call_vlm=false even if the image "
        "itself causes a model failure.\n"
        "- no_visible_text_reason: string, empty when should_call_vlm is true\n"
        "- vlm_prompt: a complete prompt for a vision-language model. The prompt must tell "
        "the model to transcribe visible text only, preserve punctuation/case/line breaks "
        "where possible, mark unreadable spans, not infer hidden text, and return JSON "
        "matching the visual_extraction schema.\n\n"
        f"Figure metadata:\n{json.dumps(figure, ensure_ascii=False, indent=2)}\n\n"
        f"Attached images: {image_count}. JSON only."
    )


def build_codex_command(
    cfg: Config,
    images: list[Path],
    output_path: Path | None = None,
    model: str | None = None,
) -> list[str]:
    cmd = [cfg.codex_bin, "-a", "never", "exec", "-C", str(ROOT), "-s", "read-only"]
    for item in cfg.codex_config:
        cmd.extend(["-c", item])
    selected_model = model if model is not None else cfg.codex_model
    if selected_model:
        cmd.extend(["-m", selected_model])
    if output_path:
        cmd.extend(["--output-last-message", str(output_path)])
    for image in images:
        cmd.append(f"--image={image}")
    cmd.append("-")
    return cmd


def run_codex_planner(cfg: Config, figure: dict[str, Any], images: list[Path]) -> dict[str, Any]:
    prompt = build_planner_prompt(figure, len(images))
    out_dir = WORKSPACE / "_codex-planner"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = (
        out_dir
        / f"{safe_id(str(figure.get('id') or 'figure'))}-{int(time.time() * 1000)}.txt"
    )
    cmd = build_codex_command(cfg, images, out_file)
    proc = subprocess.run(
        cmd,
        input=prompt,
        check=False,
        text=True,
        capture_output=True,
        timeout=cfg.codex_timeout,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(detail[:1000] or f"codex exited {proc.returncode}")
    planner_text = (
        out_file.read_text(encoding="utf-8", errors="replace")
        if out_file.exists()
        else proc.stdout
    )
    payload = extract_json_object(planner_text)
    if "vlm_prompt" not in payload:
        raise RuntimeError("Codex planner JSON did not include vlm_prompt")
    payload.setdefault("should_call_vlm", True)
    payload.setdefault("figure_role", "other")
    payload.setdefault("visual_focus", "")
    payload.setdefault("no_visible_text_reason", "")
    return payload


def build_codex_extractor_prompt(figure: dict[str, Any], image_count: int) -> str:
    return (
        "You are a deterministic OCR extraction worker for AgenticCarlini paper digests. "
        "Do not run shell commands. Analyze the attached academic-paper figure images and "
        "return exactly one JSON object matching this schema:\n"
        "{\n"
        '  "status": "ok" | "no_visible_text",\n'
        '  "figure_role": "jailbreak_prompt" | "attack_pipeline" | '
        '"result_screenshot" | "chart" | "typographic_attack" | "diagram" | '
        '"adversarial_noise" | "other",\n'
        '  "visible_text": [\n'
        '    {"text": "...", "region": "...", "role": "prompt|response|label|'
        'axis|legend|ui_text|annotation|other", "confidence": "high|medium|low"}\n'
        "  ],\n"
        '  "safety_relevant_content": [\n'
        '    {"kind": "jailbreak_prompt|injected_instruction|target_query|'
        'model_response|credential_or_identifier|ui_state|other", '
        '"text": "...", "confidence": "high|medium|low", "notes": "..."}\n'
        "  ],\n"
        '  "summary": "...",\n'
        '  "uncertain": ["..."]\n'
        "}\n\n"
        "Transcribe visible text only. Preserve punctuation, case, and line breaks where "
        "possible. Mark unreadable visible spans as [unreadable]. Merge nearby lines from "
        "the same visual region into non-overlapping blocks instead of splitting every "
        "phrase into separate items. Avoid duplicate visible_text entries. Do not infer hidden, "
        "steganographic, or occluded text. If no readable visible characters are present, "
        "use status=no_visible_text and empty arrays. If an adversarial image works through "
        "pixels/noise rather than readable characters, do not describe the attack effect "
        "as extracted text. Keep safety_relevant_content compact and do not duplicate long "
        "generated model outputs there. Keep safety_relevant_content to at most 5 high-signal "
        "items, and do not include model failure effects or inferred attack outcomes unless "
        "that text is visibly printed in the figure.\n\n"
        f"Figure metadata:\n{json.dumps(figure, ensure_ascii=False, indent=2)}\n\n"
        f"Attached images: {image_count}. JSON only."
    )


def run_codex_extractor(cfg: Config, figure: dict[str, Any], images: list[Path]) -> dict[str, Any]:
    prompt = build_codex_extractor_prompt(figure, len(images))
    out_dir = WORKSPACE / "_codex-extractor"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = (
        out_dir
        / f"{safe_id(str(figure.get('id') or 'figure'))}-{int(time.time() * 1000)}.txt"
    )
    cmd = build_codex_command(cfg, images, out_file, model=cfg.codex_extractor_model)
    proc = subprocess.run(
        cmd,
        input=prompt,
        check=False,
        text=True,
        capture_output=True,
        timeout=cfg.codex_timeout,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(detail[:1000] or f"codex extractor exited {proc.returncode}")
    extractor_text = (
        out_file.read_text(encoding="utf-8", errors="replace")
        if out_file.exists()
        else proc.stdout
    )
    return extract_json_object(extractor_text)


def _load_project_env_key() -> str | None:
    envf = ROOT / ".env"
    if not envf.exists():
        return None
    for raw in envf.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "OPENROUTER_API_KEY":
            value = value.strip().strip('"').strip("'")
            if value:
                os.environ["OPENROUTER_API_KEY"] = value
                return value
    return None


def _key_live(key: str, timeout: int = 15) -> bool:
    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/key",
            headers={"Authorization": f"Bearer {key}"},
        )
        urllib.request.urlopen(req, timeout=timeout).read()
        return True
    except Exception:
        return False


def load_openrouter_key(cfg: Config) -> str:
    key = _load_project_env_key() or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not found in project-root .env or current shell")
    if cfg.key_preflight and key not in _VERIFIED_OPENROUTER_KEYS:
        if not _key_live(key):
            raise RuntimeError("OPENROUTER_API_KEY failed OpenRouter /key preflight")
        _VERIFIED_OPENROUTER_KEYS.add(key)
    return key


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    if mime not in {"image/png", "image/jpeg", "image/jpg"}:
        mime = "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def build_vlm_user_content(prompt: str, images: list[Path]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image in images:
        content.append({"type": "image_url", "image_url": {"url": image_data_url(image)}})
    return content


def build_vlm_prompt(planner_prompt: str, figure: dict[str, Any]) -> str:
    return (
        f"{planner_prompt}\n\n"
        "Hard output contract for AgenticCarlini digest enrichment:\n"
        "Return exactly one JSON object with these keys:\n"
        "{\n"
        '  "status": "ok" | "no_visible_text",\n'
        '  "figure_role": "jailbreak_prompt" | "attack_pipeline" | '
        '"result_screenshot" | "chart" | "typographic_attack" | "diagram" | '
        '"adversarial_noise" | "other",\n'
        '  "visible_text": [\n'
        '    {"text": "...", "region": "...", "role": "prompt|response|label|'
        'axis|legend|ui_text|annotation|other", "confidence": "high|medium|low"}\n'
        "  ],\n"
        '  "safety_relevant_content": [\n'
        '    {"kind": "jailbreak_prompt|injected_instruction|target_query|'
        'model_response|credential_or_identifier|ui_state|other", '
        '"text": "...", "confidence": "high|medium|low", "notes": "..."}\n'
        "  ],\n"
        '  "summary": "...",\n'
        '  "uncertain": ["..."]\n'
        "}\n"
        "Use status=no_visible_text when no readable visible characters are present. "
        "If the image is an adversarial visual example, perturbation, photo, or diagram "
        "whose attack works through pixels rather than readable text, do not describe the "
        "attack effect as extracted text; return no_visible_text unless labels or other "
        "characters are actually readable. "
        "When text is visible, use status=ok even if some spans are unreadable. "
        "Transcribe visible text verbatim; do not infer hidden text. Merge nearby lines "
        "from the same visual region into non-overlapping blocks instead of splitting "
        "every phrase into separate items. Avoid duplicate visible_text entries. "
        "Put long generated model responses only in visible_text with role=model_response "
        "or response; do not duplicate generated outputs in safety_relevant_content. "
        "safety_relevant_content should contain compact visible prompts, injected "
        "instructions, target queries, credentials/identifiers, or UI state only. "
        "Keep safety_relevant_content to at most 5 high-signal items. Do not include "
        "model failure effects, unsafe generations, or inferred attack outcomes as "
        "safety_relevant_content unless that text is visibly printed in the figure.\n\n"
        f"Figure id: {figure.get('id')}\n"
        f"Caption: {figure.get('caption')}"
    )


def build_direct_vlm_prompt(figure: dict[str, Any]) -> str:
    return build_vlm_prompt(
        (
            "Analyze the attached academic-paper figure directly for digest enrichment. "
            "First decide whether the image contains any readable visible characters. "
            "Then extract only those visible characters and compact safety-relevant "
            "paper evidence into the required JSON schema."
        ),
        figure,
    )


def call_openrouter_vlm(
    cfg: Config,
    key: str,
    prompt: str,
    images: list[Path],
) -> dict[str, Any]:
    body = json.dumps(
        {
            "model": cfg.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You extract visible text and safety-relevant content from academic "
                        "paper figures. Return one JSON object only. Do not infer hidden or "
                        "unreadable text."
                    ),
                },
                {"role": "user", "content": build_vlm_user_content(prompt, images)},
            ],
            "temperature": 0,
            "max_tokens": cfg.max_tokens,
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(cfg.retries + 1):
        try:
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=body,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/oscarwoo/AgenticCarlini",
                    "X-Title": "AgenticCarlini Figure Visual Extraction",
                },
            )
            wall_timeout = cfg.request_wall_timeout or cfg.request_timeout
            with wall_clock_timeout(wall_timeout, "OpenRouter response"):
                response = json.load(urllib.request.urlopen(req, timeout=cfg.request_timeout))
            text = openrouter_choice_text(response)
            return extract_json_object(text)
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {exc.code}: {error_body[:800]}")
            if attempt < cfg.retries:
                time.sleep(min(60.0, cfg.retry_sleep_base * (2**attempt)))
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < cfg.retries:
                time.sleep(min(60.0, cfg.retry_sleep_base * (2**attempt)))
    raise RuntimeError(str(last_error)[:1000] if last_error else "OpenRouter request failed")


def base_visual_record(
    status: str,
    figure: dict[str, Any],
    asset: Path | None,
    cfg: Config,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    if status not in ALLOWED_STATUSES:
        status = "vlm_failed"
    record = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "asset_path": str(asset.relative_to(digest_dir_from_asset(asset))) if asset else None,
        "asset_sha256": sha256_file(asset) if asset and asset.is_file() else None,
        "model": cfg.codex_extractor_model if cfg.backend == "codex-cli" else cfg.model,
        "backend": cfg.backend,
        "codex_cli_model": (
            cfg.codex_extractor_model
            if cfg.backend == "codex-cli"
            else cfg.codex_model or "default"
        ),
        "extracted_at": utc_now(),
        "figure_role": "other",
        "visible_text": [],
        "safety_relevant_content": [],
        "summary": "",
        "uncertain": [],
    }
    if warnings:
        record["warnings"] = warnings
    return record


def _truncate_text(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit].rstrip() + "\n[truncated: visible text continues in figure]", True


def _clean_visible_text_items(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    cleaned: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if text.lower() in UNREADABLE_MARKERS:
            continue
        role = str(item.get("role") or "other").lower()
        if role not in VISIBLE_TEXT_ROLES:
            role = "other"
        limit = MAX_MODEL_RESPONSE_CHARS if role in MODEL_RESPONSE_ROLES else MAX_VISIBLE_TEXT_CHARS
        text, truncated = _truncate_text(text, limit)
        dedupe_key = (role, re.sub(r"\s+", " ", text).strip().lower())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        cleaned_item = dict(item)
        cleaned_item["text"] = text
        cleaned_item["role"] = role
        if truncated:
            cleaned_item["truncated"] = True
        cleaned.append(cleaned_item)
    return cleaned


def _looks_like_generated_output_safety_item(item: dict[str, Any]) -> bool:
    kind = str(item.get("kind") or "").lower()
    notes = str(item.get("notes") or "").lower()
    if kind == "model_response":
        return True
    if kind in {"injected_instruction", "other"} and any(
        marker in notes
        for marker in (
            "generated output",
            "harmful output",
            "model output",
            "model response",
            "response generated",
            "generated under",
        )
    ):
        return True
    return False


def _clean_safety_items(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    cleaned: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        if _looks_like_generated_output_safety_item(item):
            continue
        text = str(item.get("text") or "").strip()
        if text.lower() in UNREADABLE_MARKERS:
            continue
        kind = str(item.get("kind") or "other").lower()
        if kind not in SAFETY_CONTENT_KINDS:
            kind = "other"
        text, truncated = _truncate_text(text, MAX_SAFETY_TEXT_CHARS)
        dedupe_key = (kind, re.sub(r"\s+", " ", text).strip().lower())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        cleaned_item = dict(item)
        cleaned_item["kind"] = kind
        cleaned_item["text"] = text
        if truncated:
            cleaned_item["truncated"] = True
        cleaned.append(cleaned_item)
        if len(cleaned) >= MAX_SAFETY_ITEMS:
            break
    return cleaned


def digest_dir_from_asset(asset: Path | None) -> Path:
    if asset is None:
        return ROOT
    parts = asset.resolve().parts
    try:
        idx = parts.index("digest")
    except ValueError:
        return asset.parent
    return Path(*parts[: idx + 1])


def normalize_visual_record(
    raw: dict[str, Any],
    figure: dict[str, Any],
    asset: Path | None,
    cfg: Config,
    planner: dict[str, Any] | None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    status = str(raw.get("status") or "ok")
    if status not in ALLOWED_STATUSES:
        status = "ok"
    record = base_visual_record(status, figure, asset, cfg, warnings)
    record["figure_role"] = str(
        raw.get("figure_role") or (planner or {}).get("figure_role") or "other"
    )
    record["visible_text"] = _clean_visible_text_items(raw.get("visible_text", []))
    record["safety_relevant_content"] = _clean_safety_items(
        raw.get("safety_relevant_content", [])
    )
    uncertain = raw.get("uncertain", [])
    record["uncertain"] = uncertain if isinstance(uncertain, list) else []
    record["summary"] = str(raw.get("summary") or "")
    if planner:
        record["visual_focus"] = str(planner.get("visual_focus") or "")
    if not record["visible_text"]:
        if record["status"] == "ok":
            record["warnings"] = list(record.get("warnings") or [])
            record["warnings"].append("normalized_ok_without_visible_text_to_no_visible_text")
        record["status"] = "no_visible_text"
        record["safety_relevant_content"] = []
    return record


def failure_record(
    status: str,
    figure: dict[str, Any],
    asset: Path | None,
    cfg: Config,
    detail: str,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    record = base_visual_record(status, figure, asset, cfg, warnings)
    record["summary"] = detail[:1000]
    record["uncertain"] = [detail[:1000]] if detail else []
    return record


def no_visible_record(
    figure: dict[str, Any],
    asset: Path,
    cfg: Config,
    planner: dict[str, Any],
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    record = base_visual_record("no_visible_text", figure, asset, cfg, warnings)
    record["figure_role"] = str(planner.get("figure_role") or "other")
    record["summary"] = str(
        planner.get("no_visible_text_reason") or "No visible readable text detected."
    )
    record["visual_focus"] = str(planner.get("visual_focus") or "")
    return record


def _fallback_config(cfg: Config) -> Config:
    return replace(
        cfg,
        backend=str(cfg.fallback_backend),
        model=cfg.fallback_model or cfg.model,
        codex_extractor_model=cfg.fallback_codex_extractor_model
        or cfg.codex_extractor_model,
        fallback_backend=None,
    )


def _record_fallback_attempt(record: dict[str, Any], initial_record: dict[str, Any]) -> None:
    record["fallback_from"] = {
        "backend": initial_record.get("backend"),
        "model": initial_record.get("model"),
        "status": initial_record.get("status"),
        "summary": str(initial_record.get("summary") or "")[:500],
    }


def _extract_with_backend(
    cfg: Config,
    figure: dict[str, Any],
    asset: Path,
    images: list[Path],
    warnings: list[str],
) -> dict[str, Any]:
    if cfg.backend == "direct-vlm":
        try:
            key = load_openrouter_key(cfg)
            raw = call_openrouter_vlm(cfg, key, build_direct_vlm_prompt(figure), images)
            return normalize_visual_record(raw, figure, asset, cfg, None, warnings)
        except Exception as exc:  # noqa: BLE001
            return failure_record("vlm_failed", figure, asset, cfg, str(exc), warnings)

    if cfg.backend == "codex-cli":
        try:
            raw = run_codex_extractor(cfg, figure, images)
            return normalize_visual_record(raw, figure, asset, cfg, None, warnings)
        except Exception as exc:  # noqa: BLE001
            return failure_record("vlm_failed", figure, asset, cfg, str(exc), warnings)

    raise ValueError(f"Unsupported extractor backend for fallback: {cfg.backend}")


def maybe_run_fallback(
    record: dict[str, Any],
    cfg: Config,
    figure: dict[str, Any],
    asset: Path,
    images: list[Path],
    warnings: list[str],
) -> dict[str, Any]:
    if not cfg.fallback_backend:
        return record
    if str(record.get("status")) not in cfg.fallback_on_statuses:
        return record

    fallback_cfg = _fallback_config(cfg)
    fallback_record = _extract_with_backend(fallback_cfg, figure, asset, images, warnings)
    _record_fallback_attempt(fallback_record, record)
    return fallback_record


def existing_record_matches(figure_json: Path, asset: Path | None, retry_failed: bool = False) -> bool:
    try:
        record = load_json(figure_json).get("visual_extraction")
    except Exception:
        return False
    if not isinstance(record, dict):
        return False
    status = record.get("status")
    if retry_failed and status in FAILURE_STATUSES:
        return False
    if asset and asset.is_file():
        return record.get("asset_sha256") == sha256_file(asset)
    if record.get("extraction_source") == "latex_figure_environment":
        return status in {"ok", "no_visible_text"}
    return status == "asset_missing"


def write_visual_extraction(figure_json: Path, figure_id: str, record: dict[str, Any]) -> None:
    fig_payload = load_json(figure_json)
    fig_payload["visual_extraction"] = record
    dump_json(figure_json, fig_payload)

    paper_json = figure_json.parents[1] / "paper.json"
    if not paper_json.exists():
        return
    paper = load_json(paper_json)
    for fig in paper.get("figures", []):
        if fig.get("id") == figure_id:
            fig["visual_extraction"] = record
    dump_json(paper_json, paper)


def process_figure(
    slug: str,
    figure_json: Path,
    cfg: Config,
    write: bool,
) -> tuple[str, dict[str, Any]]:
    phase_timings: dict[str, float] = {}

    def elapsed(name: str, started_at: float) -> None:
        phase_timings[name] = time.monotonic() - started_at

    def finish(figure_id: str, record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        _LAST_PHASE_TIMINGS[str(figure_json)] = phase_timings
        return figure_id, record

    phase_started = time.monotonic()
    figure = load_json(figure_json)
    figure_id = str(figure.get("id") or figure_json.stem)
    asset = resolve_figure_asset(figure_json, figure)
    elapsed("resolve_asset_sec", phase_started)
    if asset is None:
        phase_started = time.monotonic()
        record = latex_fallback_record(figure_json, figure, cfg)
        elapsed("latex_fallback_sec", phase_started)
        if record is not None:
            if write:
                write_visual_extraction(figure_json, figure_id, record)
            return finish(figure_id, record)
        record = failure_record(
            "asset_missing",
            figure,
            None,
            cfg,
            "No sibling figure asset found.",
        )
        if write:
            write_visual_extraction(figure_json, figure_id, record)
        return finish(figure_id, record)

    if asset.suffix.lower() not in SUPPORTED_ASSET_EXTS:
        record = failure_record(
            "unsupported_asset",
            figure,
            asset,
            cfg,
            f"Unsupported figure asset extension: {asset.suffix}",
        )
        if write:
            write_visual_extraction(figure_json, figure_id, record)
        return finish(figure_id, record)

    phase_started = time.monotonic()
    try:
        images, warnings = prepare_figure_images(asset, slug, figure_id, cfg)
    except Exception as exc:  # noqa: BLE001
        elapsed("prepare_images_sec", phase_started)
        record = failure_record("render_failed", figure, asset, cfg, str(exc))
        if write:
            write_visual_extraction(figure_json, figure_id, record)
        return finish(figure_id, record)
    elapsed("prepare_images_sec", phase_started)

    phase_started = time.monotonic()
    record = local_text_gate_no_visible_record(figure, asset, images, cfg, warnings)
    elapsed("local_text_gate_sec", phase_started)
    if record is not None:
        if write:
            write_visual_extraction(figure_json, figure_id, record)
        return finish(figure_id, record)

    if cfg.backend == "direct-vlm":
        phase_started = time.monotonic()
        record = _extract_with_backend(cfg, figure, asset, images, warnings)
        elapsed("primary_extract_sec", phase_started)
        phase_started = time.monotonic()
        record = maybe_run_fallback(record, cfg, figure, asset, images, warnings)
        elapsed("fallback_check_sec", phase_started)
        if write:
            write_visual_extraction(figure_json, figure_id, record)
        return finish(figure_id, record)

    if cfg.backend == "codex-cli":
        phase_started = time.monotonic()
        record = _extract_with_backend(cfg, figure, asset, images, warnings)
        elapsed("primary_extract_sec", phase_started)
        phase_started = time.monotonic()
        record = maybe_run_fallback(record, cfg, figure, asset, images, warnings)
        elapsed("fallback_check_sec", phase_started)
        if write:
            write_visual_extraction(figure_json, figure_id, record)
        return finish(figure_id, record)

    phase_started = time.monotonic()
    try:
        planner = run_codex_planner(cfg, figure, images)
    except Exception as exc:  # noqa: BLE001
        elapsed("planner_sec", phase_started)
        record = failure_record("codex_prompt_failed", figure, asset, cfg, str(exc), warnings)
        if write:
            write_visual_extraction(figure_json, figure_id, record)
        return finish(figure_id, record)
    elapsed("planner_sec", phase_started)

    if planner.get("should_call_vlm") is False:
        record = no_visible_record(figure, asset, cfg, planner, warnings)
        if write:
            write_visual_extraction(figure_json, figure_id, record)
        return finish(figure_id, record)

    phase_started = time.monotonic()
    try:
        key = load_openrouter_key(cfg)
        vlm_prompt = build_vlm_prompt(str(planner["vlm_prompt"]), figure)
        raw = call_openrouter_vlm(cfg, key, vlm_prompt, images)
        record = normalize_visual_record(raw, figure, asset, cfg, planner, warnings)
    except Exception as exc:  # noqa: BLE001
        record = failure_record("vlm_failed", figure, asset, cfg, str(exc), warnings)
    elapsed("primary_extract_sec", phase_started)

    if write:
        write_visual_extraction(figure_json, figure_id, record)
    return finish(figure_id, record)


def select_figures(
    slug: str,
    figure_id: str | None,
    resume: bool,
    retry_failed: bool = False,
) -> list[Path]:
    paths = figure_json_paths(slug)
    selected: list[Path] = []
    for path in paths:
        fig = load_json(path)
        if figure_id and fig.get("id") != figure_id and path.stem != figure_id:
            continue
        if retry_failed and not resume:
            record = fig.get("visual_extraction")
            if not isinstance(record, dict) or record.get("status") not in FAILURE_STATUSES:
                continue
        if resume or retry_failed:
            asset = resolve_figure_asset(path, fig)
            if existing_record_matches(path, asset, retry_failed=retry_failed):
                continue
        selected.append(path)
    return selected


def run(args: argparse.Namespace, cfg: Config) -> None:
    selected = select_figures(args.slug, args.figure, args.resume, retry_failed=args.retry_failed)
    if args.limit is not None:
        selected = selected[: args.limit]
    if args.figure and not selected:
        raise SystemExit(f"figure not found or skipped by --resume: {args.figure}")

    print(
        f"slug={args.slug} figures={len(selected)} mode={'write' if args.write else 'dry-run'} "
        f"backend={cfg.backend} model={cfg.model} max_tokens={cfg.max_tokens} "
        f"retries={cfg.retries} fallback={cfg.fallback_backend or 'none'}",
        flush=True,
    )
    status_counts: dict[str, int] = {}
    for path in selected:
        started_at = time.monotonic()
        figure_id, record = process_figure(args.slug, path, cfg, args.write)
        elapsed_sec = time.monotonic() - started_at
        status = str(record.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1
        phase_timings = _LAST_PHASE_TIMINGS.pop(str(path), {})
        phase_text = ",".join(
            f"{name}={value:.2f}" for name, value in sorted(phase_timings.items())
        )
        if phase_text:
            print(
                f"  {figure_id:44s} {status} elapsed_sec={elapsed_sec:.2f} phases={phase_text}",
                flush=True,
            )
        else:
            print(f"  {figure_id:44s} {status} elapsed_sec={elapsed_sec:.2f}", flush=True)
        if not args.write:
            print(
                json.dumps(
                    {"figure_id": figure_id, "visual_extraction": record},
                    ensure_ascii=False,
                    indent=2,
                )
            )
    print(f"DONE {json.dumps(status_counts, ensure_ascii=False, sort_keys=True)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract visible figure text into Paper2Agent digest.")
    ap.add_argument("--slug", required=True)
    ap.add_argument("--figure", help="figure id such as fig:fig_1_overview or figure json stem")
    ap.add_argument("--write", action="store_true", help="persist visual_extraction into digest")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="skip figures whose asset hash already has a visual_extraction record",
    )
    ap.add_argument(
        "--retry-failed",
        action="store_true",
        help="retry existing failed visual_extraction records and skip successful records",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--backend",
        choices=("direct-vlm", "planned-vlm", "codex-cli"),
        default=DEFAULT_BACKEND,
        help=(
            "direct-vlm skips the Codex planner and calls OpenRouter once; "
            "planned-vlm preserves the original Codex-planner plus VLM flow; "
            "codex-cli uses Codex CLI as the final vision extractor"
        ),
    )
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--codex-model", default=None)
    ap.add_argument("--codex-extractor-model", default=DEFAULT_CODEX_EXTRACTOR_MODEL)
    ap.add_argument(
        "--codex-config",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="repeatable Codex CLI -c config override for the planner",
    )
    ap.add_argument("--codex-bin", default="codex")
    ap.add_argument("--codex-timeout", type=int, default=Config.codex_timeout)
    ap.add_argument("--max-tokens", type=int, default=Config.max_tokens)
    ap.add_argument("--max-pages", type=int, default=3)
    ap.add_argument("--max-long-side", type=int, default=3600)
    ap.add_argument("--max-image-bytes", type=int, default=Config.max_image_bytes)
    ap.add_argument("--render-dpi", type=int, default=220)
    ap.add_argument("--request-timeout", type=int, default=Config.request_timeout)
    ap.add_argument(
        "--request-wall-timeout",
        type=int,
        default=None,
        help=(
            "hard wall-clock timeout for one OpenRouter request/response read; "
            "defaults to --request-timeout"
        ),
    )
    ap.add_argument("--retries", type=int, default=Config.retries)
    ap.add_argument("--retry-sleep-base", type=float, default=Config.retry_sleep_base)
    ap.add_argument("--no-key-preflight", action="store_true")
    ap.add_argument(
        "--no-local-text-gate",
        action="store_true",
        help=(
            "disable the local pdftotext/tesseract no-visible-text gate and always "
            "send supported assets to the selected extractor"
        ),
    )
    ap.add_argument(
        "--tesseract-timeout",
        type=int,
        default=Config.tesseract_timeout,
        help="per-image timeout for the local OCR no-visible-text gate",
    )
    ap.add_argument(
        "--fallback-backend",
        choices=("direct-vlm", "codex-cli"),
        default=None,
        help="optional terminal extractor backend to try after selected failure statuses",
    )
    ap.add_argument(
        "--fallback-model",
        default=None,
        help="OpenRouter model for direct-vlm fallback; defaults to --model",
    )
    ap.add_argument(
        "--fallback-codex-extractor-model",
        default=None,
        help="Codex CLI model for codex-cli fallback; defaults to --codex-extractor-model",
    )
    ap.add_argument(
        "--fallback-on-status",
        action="append",
        default=[],
        choices=tuple(sorted(FAILURE_STATUSES)),
        help="status that triggers fallback; repeatable, defaults to vlm_failed",
    )
    args = ap.parse_args()

    cfg = Config(
        model=args.model,
        backend=args.backend,
        codex_model=args.codex_model,
        codex_extractor_model=args.codex_extractor_model,
        codex_config=tuple(args.codex_config),
        codex_bin=args.codex_bin,
        codex_timeout=args.codex_timeout,
        max_tokens=args.max_tokens,
        max_pages=args.max_pages,
        max_long_side=args.max_long_side,
        max_image_bytes=args.max_image_bytes,
        render_dpi=args.render_dpi,
        request_timeout=args.request_timeout,
        request_wall_timeout=args.request_wall_timeout,
        retries=args.retries,
        retry_sleep_base=args.retry_sleep_base,
        key_preflight=not args.no_key_preflight,
        local_text_gate=not args.no_local_text_gate,
        tesseract_timeout=args.tesseract_timeout,
        fallback_backend=args.fallback_backend,
        fallback_model=args.fallback_model,
        fallback_codex_extractor_model=args.fallback_codex_extractor_model,
        fallback_on_statuses=tuple(args.fallback_on_status or ("vlm_failed",)),
    )

    if not digest_dir(args.slug).is_dir():
        raise SystemExit(f"digest not found for slug: {args.slug}")

    if args.write:
        with slug_write_lock(args.slug):
            run(args, cfg)
    else:
        run(args, cfg)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
