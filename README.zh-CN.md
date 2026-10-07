# arxiv2agent

[English](README.md) | **简体中文**

几个小下载工具：让 agent 把任意 arXiv 论文以 **LaTeX 源码**的形式拉到本地，读真论文，而不是搜索摘要或有损的 PDF。

![find → fetch / digest](docs/flow.svg)

```bash
arxiv2agent find "Attention Is All You Need"   # 标题 → arXiv ID（arXiv API）
arxiv2agent fetch 1706.03762 -o papers/        # 原始源码树 + 一份展开后的单文件 .tex
arxiv2agent digest 1706.03762 -o papers/       # 结构化：paper.json、sections/、tables/ …
```

`fetch` 和 `digest` 可以一次传多个 ID。内部没有 LLM，不需要 API key；对 arXiv 的请求自动限速并缓存。

## 使用效果

```console
$ arxiv2agent find "Attention Is All You Need" -n 3
1706.03762	2017	Attention Is All You Need	Ashish Vaswani et al.
2104.04692	2021	Not All Attention Is All You Need	Hongqiu Wu et al.
2501.06425	2025	Tensor Product Attention Is All You Need	Yifan Zhang et al.

$ arxiv2agent fetch 1706.03762 -o papers/
$ ls papers/1706.03762/source
Figures/  background.tex  introduction.tex  ms.tex  nips_2017.sty  results.tex  …
```

**“Table 1 是怎么排的？”** agent grep 一下 `papers/1706.03762/1706.03762.tex`，拿到的是原样源码：`booktabs`、`\toprule`、用来撑行高的 `\rule{0pt}{2.0ex}`，还有作者注释掉的 `\scalebox`：

```latex
\begin{tabular}{lccc}
\toprule
Layer Type & Complexity per Layer & Sequential & Maximum Path Length  \\
\hline
\rule{0pt}{2.0ex}Self-Attention & $O(n^2 \cdot d)$ & $O(1)$ & $O(1)$ \\
```

**“这篇论文 3.2 节到底说了什么？”** agent 跑一次 `digest`，读 `sections/` 或 `paper.json`，基于全文回答；公式原样在 `equations/*.tex`，引用解析成 BibTeX 在 `references.json`。

**“对比这十篇论文的 Introduction。”** 一条 `digest` 传十个 ID，然后循环读 `paper.json`（每篇论文 schema 相同）。

## 安装

```bash
uv tool install git+https://github.com/wuyoscar/arxiv2agent
```

给 agent（Claude Code、Codex 等）用，直接告诉它：

> 用 `uv tool install` 安装 https://github.com/wuyoscar/arxiv2agent，然后读它的 `SKILL.md` 并注册成 skill。

[`SKILL.md`](SKILL.md) 里写了命令、digest 目录结构，以及什么时候用哪个。

## 致谢

LaTeX 下载与展开流程 vendored 自 **arxiv-to-prompt**（见 [NOTICE.md](NOTICE.md)），也受 **DeepXiv** 启发。

## License

MIT.
