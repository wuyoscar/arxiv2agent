"""LaTeX-source download / flatten / denoise / section-tree-parse.

Vendored from `arxiv-to-prompt` v0.11.0 (MIT, © 2025 Takashi Ishida).
Original: https://github.com/takashiishida/arxiv-to-prompt

We vendor (rather than depend) for three reasons:
  1. We depend on internal data structures (`SectionNode`), not just public CLI.
  2. We want to make small targeted improvements (filter template-pollution
     section titles, drop unused CLI-only code paths like clipboard / abstract-
     only / figure-paths-only).
  3. Open-source PR contributors should see a complete, self-contained codebase.

Attribution: see /NOTICE.md (includes the full upstream MIT license text).

Modifications from upstream v0.11.0:
  - Removed `figure_paths_only` and `abstract_only` paths from
    `process_latex_source` (we have our own extract.py for these).
  - Removed pyperclip clipboard support (CLI-only feature).
  - `parse_section_tree`: skip section nodes whose title contains template
    pollution (`\\@`, `\\@mkboth`, balance-of-braces issues).
  - Renamed module-private helpers to `_underscore` form for clarity.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import tarfile
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import requests
from filelock import FileLock, Timeout

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_CACHE_COMPLETE_MARKER = ".arxiv_cache_complete"


# ─────────────────────────────────────────────────────────────────────────────
# Cache directory & file locks
# ─────────────────────────────────────────────────────────────────────────────


def get_default_cache_dir() -> Path:
    """OS-standard cache directory for downloaded arXiv sources."""
    if os.name == 'nt':
        base_dir = Path(os.environ.get('LOCALAPPDATA', '~'))
    else:
        base_dir = Path(os.environ.get('XDG_CACHE_HOME', '~/.cache'))
    return base_dir.expanduser() / 'arxiv-to-prompt'


def _cache_has_tex_files(directory: Path) -> bool:
    return any(directory.rglob("*.tex"))


def _is_valid_cache_dir(directory: Path) -> bool:
    if not directory.exists() or not directory.is_dir():
        return False
    marker_path = directory / _CACHE_COMPLETE_MARKER
    return marker_path.is_file() and _cache_has_tex_files(directory)


def _safe_rmtree(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    except Exception as exc:
        logging.warning(f"Failed to remove directory {path}: {exc}")


def _get_lock_path(base_dir: Path, arxiv_id: str) -> Path:
    lock_key = hashlib.sha256(arxiv_id.encode("utf-8")).hexdigest()
    return base_dir / ".locks" / f"{lock_key}.lock"


def _extract_tar_safely(tar_path: Path, extract_to: Path) -> None:
    with tarfile.open(tar_path) as tar:
        for member in tar.getmembers():
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe path in tar archive: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"Link entry in tar archive is not allowed: {member.name}")
        try:
            tar.extractall(path=extract_to, filter="data")
        except TypeError:
            tar.extractall(path=extract_to)


# ─────────────────────────────────────────────────────────────────────────────
# Download
# ─────────────────────────────────────────────────────────────────────────────


def check_source_available(arxiv_id: str) -> bool:
    url = f'https://arxiv.org/format/{arxiv_id}'
    headers = {'User-Agent': 'Mozilla/5.0'}
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=3)
    session.mount('https://', adapter)
    try:
        response = session.get(url, headers=headers, timeout=(5, 30))
        response.raise_for_status()
        return 'Download source' in response.text
    except requests.exceptions.RequestException as exc:
        logging.error(f"Error checking source availability: {exc}")
        return False
    finally:
        session.close()


def download_arxiv_source(
    arxiv_id: str,
    cache_dir: Optional[str] = None,
    use_cache: bool = False,
    lock_timeout_seconds: float = 120.0,
    stale_cache_repair: bool = True,
) -> bool:
    """Download (and cache) the .tar source for an arXiv paper."""
    base_dir = Path(cache_dir) if cache_dir else get_default_cache_dir()
    directory = base_dir / arxiv_id
    lock_path = _get_lock_path(base_dir, arxiv_id)
    staging_root = base_dir / ".staging"

    try:
        base_dir.mkdir(parents=True, exist_ok=True)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        staging_root.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logging.error(f"Failed to initialize cache directories in {base_dir}: {exc}")
        return False

    try:
        with FileLock(str(lock_path), timeout=lock_timeout_seconds):
            if directory.exists():
                if use_cache and _is_valid_cache_dir(directory):
                    logging.info(f"Directory {directory} already exists, using cached version.")
                    return True
                if use_cache and not stale_cache_repair:
                    logging.error(
                        f"Cached directory {directory} is incomplete and stale cache "
                        f"repair is disabled."
                    )
                    return False
                if not _is_valid_cache_dir(directory):
                    logging.warning(f"Found incomplete cache at {directory}; rebuilding.")

            if not check_source_available(arxiv_id):
                logging.warning(f"TeX source files not available for {arxiv_id}")
                return False

            url = f'https://arxiv.org/e-print/{arxiv_id}'
            logging.info(f"Downloading source from {url}")
            headers = {'User-Agent': 'Mozilla/5.0'}

            staging_dir = Path(tempfile.mkdtemp(prefix=f"{arxiv_id}.", dir=staging_root))
            extracted_dir = staging_dir / "extracted"
            tar_path = staging_dir / "source.tar"

            try:
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                with open(tar_path, 'wb') as fh:
                    fh.write(response.content)

                extracted_dir.mkdir(parents=True, exist_ok=True)
                _extract_tar_safely(tar_path, extracted_dir)

                if not _cache_has_tex_files(extracted_dir):
                    raise ValueError("Downloaded archive does not contain any .tex files")

                (extracted_dir / _CACHE_COMPLETE_MARKER).write_text("ok\n", encoding="utf-8")

                backup_dir = None
                published = False
                try:
                    if directory.exists():
                        backup_dir = directory.parent / f"{directory.name}.old.{uuid.uuid4().hex}"
                        os.replace(str(directory), str(backup_dir))
                    os.replace(str(extracted_dir), str(directory))
                    published = True
                except Exception as publish_error:
                    if backup_dir and backup_dir.exists() and not directory.exists():
                        try:
                            os.replace(str(backup_dir), str(directory))
                            raise RuntimeError(
                                "Failed to publish new cache; rolled back to previous cache."
                            ) from publish_error
                        except Exception as rollback_exc:
                            raise RuntimeError(
                                f"Failed to publish new cache and rollback failed: {rollback_exc}"
                            ) from publish_error
                    raise RuntimeError("Failed to publish new cache.") from publish_error
                finally:
                    if published and backup_dir and backup_dir.exists():
                        _safe_rmtree(backup_dir)

                logging.info(f"Source files downloaded and extracted to {directory}/")
                return True
            finally:
                _safe_rmtree(staging_dir)

    except Timeout:
        logging.error(
            f"Timed out waiting for download lock for {arxiv_id} after "
            f"{lock_timeout_seconds} seconds"
        )
        return False
    except Exception as exc:
        logging.error(f"Error downloading/extracting source: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# TeX discovery and flattening
# ─────────────────────────────────────────────────────────────────────────────


def find_main_tex(directory: str) -> Optional[str]:
    """Find the master .tex file — the one that holds the document body.

    The master is identified by ``\\begin{document}``, NOT ``\\documentclass``:
    some manuscripts keep main.tex (with the body) but \\input a preamble file
    that holds ``\\documentclass`` — keying on \\documentclass then wrongly picks
    the subdir preamble and the body is never flattened (observed: papers whose
    body lives in Sections/ , tex/ , 0_contents/ came out with 0 sections).

    Ranking among candidates: a file with ``\\begin{document}`` always beats one
    with only ``\\documentclass``; then top-level beats a subdirectory; then a
    conventional name (main/paper/index); then the longest file.
    """
    common_names = {'main.tex', 'paper.tex', 'index.tex'}
    # each candidate: (has_begin_doc, is_toplevel, is_common_name, line_count, relpath)
    candidates: list[tuple] = []
    for root, _dirs, files in os.walk(directory):
        rel_root = os.path.relpath(root, directory)
        for file_name in files:
            if not file_name.endswith('.tex'):
                continue
            file_path = os.path.join(root, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as fh:
                    text = fh.read()
            except Exception as exc:
                logging.warning(f"Could not read file {file_path}: {exc}")
                continue
            has_begin = '\\begin{document}' in text
            has_class = '\\documentclass' in text
            if not (has_begin or has_class):
                continue
            relpath = file_name if rel_root == '.' else os.path.join(rel_root, file_name)
            candidates.append((
                has_begin,
                rel_root == '.',
                file_name in common_names,
                text.count('\n'),
                relpath,
            ))
    if not candidates:
        return None
    # max() on the tuple ranks by: begin-document > toplevel > common-name > length
    best = max(candidates, key=lambda c: (c[0], c[1], c[2], c[3]))
    return best[4]


def flatten_tex(directory: str, main_file: str) -> str:
    """Recursively expand \\input / \\include directives into one big string.

    Comment-aware: \\input statements on commented lines are left untouched.
    """
    def process_file(file_path: str, processed: set) -> str:
        if file_path in processed:
            return ""
        processed.add(file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as fh:
                content = fh.read()
        except Exception as exc:
            logging.warning(f"Error processing file {file_path}: {exc}")
            return ""

        def replace_input(match: re.Match) -> str:
            # Skip commented-out \input lines
            line_start = content.rfind('\n', 0, match.start()) + 1
            line_prefix = content[line_start:match.start()]
            comment_pos = -1
            i = 0
            while i < len(line_prefix):
                if line_prefix[i] == '%':
                    if i > 0 and line_prefix[i - 1] == '\\':
                        # Count backslashes for escape handling
                        bs_count = 0
                        j = i - 1
                        while j >= 0 and line_prefix[j] == '\\':
                            bs_count += 1
                            j -= 1
                        if bs_count % 2 == 1:
                            i += 1
                            continue
                    comment_pos = i
                    break
                i += 1
            if comment_pos != -1:
                return match.group(0)

            input_file = match.group(1)
            if not input_file.endswith('.tex'):
                tex_path = os.path.join(directory, input_file + '.tex')
                input_path = tex_path if os.path.isfile(tex_path) else os.path.join(directory, input_file)
            else:
                input_path = os.path.join(directory, input_file)
            return process_file(input_path, processed)

        return re.sub(r'\\(?:input|include){([^}]+)}', replace_input, content)

    main_path = os.path.join(directory, main_file)
    return process_file(main_path, set())


# ─────────────────────────────────────────────────────────────────────────────
# Comments, appendix, abstract, macros
# ─────────────────────────────────────────────────────────────────────────────


# BUG-3: a literal `%` inside verbatim/lstlisting/minted/\verb is NOT a comment,
# but the line-stripper below treats every unescaped `%` as a comment start and
# truncates the line — silently chopping code listings, URLs, XML/templates that
# contain `%`. Mask these regions before stripping, restore after.
_VERBATIM_BLOCK_RE = re.compile(
    r'\\begin\{(verbatim|verbatim\*|lstlisting|minted|Verbatim|alltt|comment)\}'
    r'.*?\\end\{\1\}',
    re.DOTALL,
)
_VERB_INLINE_RE = re.compile(r'\\(?:verb|lstinline)\*?(\S).*?\1')
_VERBATIM_SENTINEL = "\x00VERB{}\x00"


def remove_comments_from_lines(text: str) -> str:
    text = re.sub(r'\\iffalse\b.*?\\fi\b', '', text, flags=re.DOTALL)

    # mask verbatim-family regions (block + inline) so their `%` survives
    stash: list[str] = []

    def _grab(m: "re.Match[str]") -> str:
        stash.append(m.group(0))
        return _VERBATIM_SENTINEL.format(len(stash) - 1)

    text = _VERBATIM_BLOCK_RE.sub(_grab, text)
    text = _VERB_INLINE_RE.sub(_grab, text)

    lines = text.split('\n')
    result = []
    for line in lines:
        if line.lstrip().startswith('%'):
            continue
        in_command = False
        cleaned = []
        for char in line:
            if char == '\\':
                in_command = True
                cleaned.append(char)
            elif in_command:
                in_command = False
                cleaned.append(char)
            elif char == '%' and not in_command:
                break
            else:
                cleaned.append(char)
        result.append(''.join(cleaned).rstrip())
    out = '\n'.join(result)

    for idx, span in enumerate(stash):  # restore verbatim verbatim
        out = out.replace(_VERBATIM_SENTINEL.format(idx), span)
    return out


def remove_appendix(text: str) -> str:
    m = re.search(r'\\appendix\b', text)
    if m:
        return text[:m.start()].rstrip()
    return text


def extract_abstract(text: str) -> Optional[str]:
    m = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', text, re.DOTALL)
    return m.group(1).strip() if m else None


# ── Macro expansion ─────────────────────────────────────────────────────────


@dataclass
class MacroDefinition:
    name: str
    num_args: int
    optional_default: Optional[str]
    body: str
    is_math_operator: bool = False
    starred: bool = False


def _find_matching_brace(text: str, pos: int) -> int:
    if pos >= len(text) or text[pos] != '{':
        return -1
    depth = 1
    i = pos + 1
    while i < len(text):
        if text[i] == '\\' and i + 1 < len(text) and text[i + 1] in ('{', '}'):
            i += 2
            continue
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _find_matching_bracket(text: str, pos: int) -> int:
    if pos >= len(text) or text[pos] != '[':
        return -1
    depth = 1
    i = pos + 1
    while i < len(text):
        if text[i] == '\\' and i + 1 < len(text) and text[i + 1] in ('[', ']'):
            i += 2
            continue
        if text[i] == '[':
            depth += 1
        elif text[i] == ']':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


# Structural commands the downstream extractors scan for LITERALLY (section
# tree, title/author/abstract, cite/ref/label/caption/footnote, graphics).
# A paper that redefines any of these via \def / \renewcommand (common in
# journal/article style files) must NOT have the redefinition expanded — that
# would turn `\section{X}` into `\@startsection{...}{X}` and the section parser
# would find nothing (observed: 11 papers with 0 sections / empty title).
_PROTECTED_MACRO_EXACT = frozenset({
    "section", "subsection", "subsubsection", "subsubsubsection",
    "paragraph", "subparagraph", "chapter", "part", "appendix",
    "title", "author", "abstract", "maketitle",
    "ref", "cref", "Cref", "eqref", "autoref", "nameref", "pageref", "vref",
    "label", "caption", "footnote", "footnotetext",
    "includegraphics", "input", "include",
})


def _is_protected_macro(cmd_name: str) -> bool:
    """True if a macro name must stay literal (never expand its redefinition)."""
    base = cmd_name.lstrip("\\")
    if base in _PROTECTED_MACRO_EXACT:
        return True
    low = base.lower()
    # cite-like (\citep \citet \parencite \textcite …) and sectioning/title-like
    return "cite" in low or low.endswith("section") or low.endswith("title")


def _parse_macro_definitions(text: str) -> tuple:
    macros: dict[str, MacroDefinition] = {}
    regions_to_remove: list[tuple[int, int]] = []

    # \newcommand / \renewcommand / \providecommand
    for match in re.finditer(r'\\(newcommand|renewcommand|providecommand)\*?\s*', text):
        start = match.start()
        pos = match.end()
        while pos < len(text) and text[pos] in ' \t':
            pos += 1
        if pos < len(text) and text[pos] == '{':
            close = _find_matching_brace(text, pos)
            if close == -1:
                continue
            cmd_name = text[pos + 1:close].strip()
            pos = close + 1
        elif pos < len(text) and text[pos] == '\\':
            nm = re.match(r'\\([a-zA-Z@]+)', text[pos:])
            if not nm:
                continue
            cmd_name = '\\' + nm.group(1)
            pos += nm.end()
        else:
            continue
        if not cmd_name.startswith('\\'):
            cmd_name = '\\' + cmd_name
        while pos < len(text) and text[pos] in ' \t':
            pos += 1
        num_args = 0
        if pos < len(text) and text[pos] == '[':
            bc = _find_matching_bracket(text, pos)
            if bc == -1:
                continue
            try:
                num_args = int(text[pos + 1:bc].strip())
            except ValueError:
                continue
            pos = bc + 1
        while pos < len(text) and text[pos] in ' \t':
            pos += 1
        optional_default = None
        if pos < len(text) and text[pos] == '[':
            bc = _find_matching_bracket(text, pos)
            if bc == -1:
                continue
            optional_default = text[pos + 1:bc]
            pos = bc + 1
        while pos < len(text) and text[pos] in ' \t':
            pos += 1
        if pos >= len(text) or text[pos] != '{':
            continue
        body_close = _find_matching_brace(text, pos)
        if body_close == -1:
            continue
        body = text[pos + 1:body_close]
        starred = '*' in match.group(0)
        # strip the definition line regardless, but never register a protected
        # structural command for expansion (keep `\section{X}` literal).
        if not _is_protected_macro(cmd_name):
            macros[cmd_name] = MacroDefinition(
                name=cmd_name, num_args=num_args, optional_default=optional_default,
                body=body, starred=starred,
            )
        end = body_close + 1
        if end < len(text) and text[end] == '\n':
            end += 1
        regions_to_remove.append((start, end))

    # \DeclareMathOperator
    for match in re.finditer(r'\\DeclareMathOperator(\*?)\s*', text):
        start = match.start()
        starred = match.group(1) == '*'
        pos = match.end()
        while pos < len(text) and text[pos] in ' \t':
            pos += 1
        if pos >= len(text) or text[pos] != '{':
            continue
        close = _find_matching_brace(text, pos)
        if close == -1:
            continue
        cmd_name = text[pos + 1:close].strip()
        if not cmd_name.startswith('\\'):
            cmd_name = '\\' + cmd_name
        pos = close + 1
        while pos < len(text) and text[pos] in ' \t':
            pos += 1
        if pos >= len(text) or text[pos] != '{':
            continue
        body_close = _find_matching_brace(text, pos)
        if body_close == -1:
            continue
        op_text = text[pos + 1:body_close]
        body = (f'\\operatorname*{{{op_text}}}' if starred else f'\\operatorname{{{op_text}}}')
        macros[cmd_name] = MacroDefinition(
            name=cmd_name, num_args=0, optional_default=None,
            body=body, is_math_operator=True, starred=starred,
        )
        end = body_close + 1
        if end < len(text) and text[end] == '\n':
            end += 1
        regions_to_remove.append((start, end))

    # \def\cmd{body}
    for match in re.finditer(r'\\def\s*(\\[a-zA-Z@]+)\s*', text):
        start = match.start()
        cmd_name = match.group(1)
        pos = match.end()
        while pos < len(text) and text[pos] in ' \t':
            pos += 1
        if pos >= len(text) or text[pos] != '{':
            continue
        body_close = _find_matching_brace(text, pos)
        if body_close == -1:
            continue
        body = text[pos + 1:body_close]
        if not _is_protected_macro(cmd_name):
            macros[cmd_name] = MacroDefinition(
                name=cmd_name, num_args=0, optional_default=None, body=body,
            )
        end = body_close + 1
        if end < len(text) and text[end] == '\n':
            end += 1
        regions_to_remove.append((start, end))

    # Remove regions (merge overlap then reverse-replace)
    regions_to_remove.sort()
    merged: list[tuple[int, int]] = []
    for s, e in regions_to_remove:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    cleaned = text
    for s, e in reversed(merged):
        cleaned = cleaned[:s] + cleaned[e:]
    return macros, cleaned


def _expand_single_macro(text: str, macro: MacroDefinition) -> str:
    name_escaped = re.escape(macro.name)
    if macro.num_args == 0:
        pat = re.compile(name_escaped + r'(?![a-zA-Z@])')
        return pat.sub(lambda _m: macro.body, text)
    pat = re.compile(name_escaped + r'(?![a-zA-Z@])')
    replacements: list[tuple[int, int, str]] = []
    for match in pat.finditer(text):
        start = match.start()
        pos = match.end()
        while pos < len(text) and text[pos] in ' \t\n':
            pos += 1
        args: list[str] = []
        has_optional = macro.optional_default is not None
        if has_optional:
            if pos < len(text) and text[pos] == '[':
                bc = _find_matching_bracket(text, pos)
                if bc == -1:
                    continue
                args.append(text[pos + 1:bc])
                pos = bc + 1
            else:
                args.append(macro.optional_default)
            remaining = macro.num_args - 1
        else:
            remaining = macro.num_args
        success = True
        for _ in range(remaining):
            while pos < len(text) and text[pos] in ' \t\n':
                pos += 1
            if pos >= len(text) or text[pos] != '{':
                success = False
                break
            bc = _find_matching_brace(text, pos)
            if bc == -1:
                success = False
                break
            args.append(text[pos + 1:bc])
            pos = bc + 1
        if not success or len(args) != macro.num_args:
            continue
        result = macro.body
        for i, arg in enumerate(args, 1):
            result = result.replace(f'#{i}', arg)
        replacements.append((start, pos, result))
    for start, end, replacement in reversed(replacements):
        text = text[:start] + replacement + text[end:]
    return text


def expand_macros(text: str) -> str:
    """Expand custom `\\newcommand` / `\\def` / `\\DeclareMathOperator` macros inline."""
    macros, text = _parse_macro_definitions(text)
    if not macros:
        return text
    for _ in range(10):
        previous = text
        for macro in macros.values():
            text = _expand_single_macro(text, macro)
        if text == previous:
            break
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Section tree
# ─────────────────────────────────────────────────────────────────────────────


# Template-pollution markers — these are LaTeX header machinery, not real sections.
_POLLUTION_RE = re.compile(r'\\@|@mkboth|@startsection')


@dataclass
class SectionNode:
    level: int                                 # 0=section, 1=subsection, 2=subsubsection
    name: str
    start_pos: int
    end_pos: int = -1
    children: List['SectionNode'] = field(default_factory=list)
    parent: Optional['SectionNode'] = None


def parse_section_tree(text: str) -> List[SectionNode]:
    """Build hierarchical tree from \\section / \\subsection / \\subsubsection commands.

    Uses balanced-brace title extraction so titles containing nested commands
    (e.g. `\\subsection{Proof of \\Cref{lem:reduction}}`) are captured fully —
    not truncated at the first `}`.

    Filters out template-pollution titles (e.g. `\\section*{References\\@mkboth...}`
    which match the regex but are actually template machinery, not document content).
    """
    pattern = re.compile(r'\\(section|subsection|subsubsection)\*?\s*\{')
    level_map = {'section': 0, 'subsection': 1, 'subsubsection': 2}

    all_nodes: list[SectionNode] = []
    for match in pattern.finditer(text):
        # The match ends just past the opening `{`. Find its matching close.
        open_brace_pos = match.end() - 1
        close = _find_matching_brace(text, open_brace_pos)
        if close == -1:
            continue
        name = text[open_brace_pos + 1:close]
        if _POLLUTION_RE.search(name):
            continue   # skip template-internal section commands
        all_nodes.append(SectionNode(
            level=level_map[match.group(1)],
            name=name,
            start_pos=match.start(),
        ))

    # Set end positions: each section ends where the next same-or-higher level starts.
    for i, node in enumerate(all_nodes):
        for j in range(i + 1, len(all_nodes)):
            if all_nodes[j].level <= node.level:
                node.end_pos = all_nodes[j].start_pos
                break
        if node.end_pos == -1:
            node.end_pos = len(text)

    # Build tree
    root_nodes: list[SectionNode] = []
    stack: list[SectionNode] = []
    for node in all_nodes:
        while stack and stack[-1].level >= node.level:
            stack.pop()
        if stack:
            node.parent = stack[-1]
            stack[-1].children.append(node)
        else:
            root_nodes.append(node)
        stack.append(node)
    return root_nodes


def list_sections(text: str) -> list[str]:
    """Return all top-level \\section names (legacy convenience)."""
    return [n.name for n in parse_section_tree(text)]


# ─────────────────────────────────────────────────────────────────────────────
# High-level entry point
# ─────────────────────────────────────────────────────────────────────────────


def process_latex_source(
    arxiv_id: Optional[str] = None,
    keep_comments: bool = True,
    cache_dir: Optional[str] = None,
    use_cache: bool = False,
    remove_appendix_section: bool = False,
    local_folder: Optional[str] = None,
    lock_timeout_seconds: float = 120.0,
    expand_macros_flag: bool = False,
) -> Optional[str]:
    """Resolve a paper to its flattened, optionally-denoised TeX string.

    Either `arxiv_id` or `local_folder` must be provided.

    Returns the cleaned TeX content, or None if processing fails.
    """
    if local_folder:
        directory = Path(local_folder).expanduser().resolve()
        if not directory.exists():
            logging.error(f"Local folder does not exist: {directory}")
            return None
        if not directory.is_dir():
            logging.error(f"Path is not a directory: {directory}")
            return None
        logging.info(f"Processing local folder: {directory}")
    elif arxiv_id:
        base_dir = Path(cache_dir) if cache_dir else get_default_cache_dir()
        if not download_arxiv_source(
            arxiv_id, cache_dir, use_cache,
            lock_timeout_seconds=lock_timeout_seconds,
        ):
            return None
        directory = base_dir / arxiv_id
    else:
        logging.error("Either arxiv_id or local_folder must be provided")
        return None

    main_file = find_main_tex(str(directory))
    if not main_file:
        logging.error("Main .tex file not found.")
        return None

    content = flatten_tex(str(directory), main_file)

    if not keep_comments:
        content = remove_comments_from_lines(content)
    if expand_macros_flag:
        content = expand_macros(content)
    if remove_appendix_section:
        content = remove_appendix(content)

    return content
