# Third-Party Attributions

`arxiv2agent` vendors part of [`arxiv-to-prompt`](https://github.com/takashiishida/arxiv-to-prompt) (v0.11.0) — the LaTeX download / flatten / denoise / section-tree pipeline lives at `src/arxiv2agent/_tex.py`. We vendor (rather than depend) because we rely on internal data structures (`SectionNode`) and want to make targeted improvements without forking.

The vendored code is MIT-licensed:

> MIT License
>
> Copyright (c) 2025 Takashi Ishida
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

## Modifications

The vendored `_tex.py` differs from upstream `arxiv-to-prompt` v0.11.0 in:

1. Removed `figure_paths_only` and `abstract_only` paths from `process_latex_source` — we have our own `extract.py` for these.
2. Removed `pyperclip` clipboard support (CLI-only feature we don't use).
3. `parse_section_tree`: skips section nodes whose title contains template pollution markers (`\@`, `@mkboth`, `@startsection`) — these are LaTeX header machinery, not real document sections.
4. Minor code-style normalization (private helper underscores, type hints).

All modifications stay within the spirit of the original MIT license. We encourage upstream-friendly improvements to be contributed back to `arxiv-to-prompt` where applicable.
