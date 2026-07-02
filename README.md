# arxiv2agent

把 arXiv 论文变成 **Agent 方便检索的结构化多模态文件目录**。

全程不涉及任何大模型——纯规则解析 LaTeX 源码，确定、免费、快（实测：解析一篇 0.1–0.3 秒；10 篇新论文含下载端到端 42 秒；缓存后 10 篇重建 0.9 秒）。给 Agent 一个论文标题或 arXiv ID，几秒后整篇论文就在本地，结构化、可检索。

```
arxiv2agent 2305.13860 -o corpus/                    # 单篇
arxiv2agent 2305.13860 1706.03762 2005.14165 -o corpus/   # 批量：直接传 ID 列表
```

```
corpus/2305.13860/
├── paper.json                ← 所有论文统一 schema 的完整记录
├── README.md                 ← 导航入口
├── sections/01-introduction.md, 02-background.md, …
├── figures/fig-overview.pdf  ← 原始矢量图（子图全部保留）
│   figures/fig-prompt.txt    ← 文本型图（prompt 框、示例）
├── tables/tab-results.tex    ← 原始 LaTeX 表格
├── equations/eq-loss.tex     ← 原始 LaTeX 公式，排版一模一样
├── listings/lst-1.py         ← 可直接运行的代码
├── references.json           ← 引用解析成 BibTeX
└── footnotes.json
```

<!-- TODO: 换成 docs/digest-structure.png（标注版截图） -->

## Idea

我把 Agent 读论文当成一个**检索问题**来做：论文里的每个 section、图、表、公式、算法、代码、引用都变成一个有稳定 ID 的实体（`sec:3.2`、`fig:overview`、`eq:loss`、`[@xu2021]`），内容保留原生形态（矢量 PDF、原始 LaTeX、可运行代码、markdown 正文），外加一份全语料统一 schema 的 `paper.json`。Agent 大规模读论文时不再一篇篇搜索下载解析——一个 list comprehension 就能拿到多篇论文的某个章节。

## 使用场景

**1.「对比这十篇论文的 Introduction」** — 十篇论文，一个循环，秒级完成：

```python
import json
intros = {
    pid: [s["text"] for s in json.load(open(f"corpus/{pid}/paper.json"))["sections"]
          if "introduction" in s["title"].lower()]
    for pid in ids   # 十个 arXiv ID
}
```

**2.「参考这篇论文的 Fig 2 矢量图，生成一个风格类似的」** — `figures/fig-*.pdf` 是从源码拷出的原始矢量图（多子图全部保留），是真正可参考的资产，不是截图。

**3.「这篇论文 Related Work 引用了哪些论文？BibTeX 直接给我」** — `references.json` 里每个引用键都有解析好的标题/作者/年份 + 原文 `bib_raw`，还有 `cited_in` 标明在哪些章节被引。

**4.「论文的公式一，LaTeX 排版一模一样给我」** — `equations/eq-*.tex` 是未经改动的原始 LaTeX，贴进你的文档渲染结果完全一致。表格、算法同理。

同样的结构也支持论断核查（沿 `[@key]`、`[#fig:x]` 标记从正文跳到证据，`is_appendix` 区分正文与附录证据）和语料库构建（统一 schema + 逐字段溯源）。

## 安装

```bash
git clone https://github.com/wuyoscar/arxiv2agent && cd arxiv2agent
uv tool install .
```

### 作为 Agent Skill 安装（推荐）

这个工具是给 coding agent（Claude Code、Codex …）用的。把这句话发给你的 agent：

> Clone https://github.com/wuyoscar/arxiv2agent，用 `uv tool install .` 安装，然后阅读其中的 `SKILL.md` 并注册为 skill，之后帮我处理和检索 arXiv 论文。

[`SKILL.md`](SKILL.md) 教 agent 使用 CLI、导航 digest 目录，以及 ≥2 篇论文时改写脚本批量检索的模式。

## CLI

```
arxiv2agent ARXIV_ID [ARXIV_ID …] [-o OUT_DIR] [--local-folder PATH] [--include-source]
```

| 参数                 | 含义                                                          |
|----------------------|---------------------------------------------------------------|
| `ARXIV_ID …`         | 一个或多个 ID，如 `2305.13860 1706.03762`。用 `--local-folder` 时省略。 |
| `-o, --output`       | 父目录；每篇 digest 落在 `<output>/<arxiv_id>/`。默认当前目录。 |
| `--local-folder`     | 用本地 LaTeX 文件夹，不走下载。                               |
| `--include-source`   | 同时把原始 LaTeX 树镜像到 `<digest>/source/`。默认关闭。      |

批量时单篇失败不会中断整个任务，结尾汇总报告。

## 速度

实测（M 系列 MacBook，真实 arXiv 论文）：

| 场景 | 耗时 |
|---|---|
| 解析一篇（已缓存） | 0.13 s（GPT-3 这种超长论文 0.29 s） |
| 10 篇新论文，含下载，端到端 | 42 s |
| 10 篇重建（全缓存） | 0.9 s |

每篇论文一生只下载一次，之后完全离线；下载自带礼貌间隔与退避重试。

## 实体 ID 与行内标记

每个实体都有稳定的带类型前缀 ID，原始 `\label{}` 保留在 `latex_label`；无标签实体自动编号（`eq:1`、`eq:2`），绝不为 null。

| 实体      | 前缀      | 示例           |
|-----------|-----------|----------------|
| section   | `sec:`    | `sec:3.2.1`    |
| figure    | `fig:`    | `fig:overview` |
| table     | `tab:`    | `tab:results`  |
| equation  | `eq:`     | `eq:loss`      |
| algorithm | `alg:`    | `alg:tool`     |
| listing   | `lst:`    | `lst:1`        |
| footnote  | `fn:`     | `fn:1`         |
| citation  | (bib key) | `xu2021`       |

正文 markdown 中交叉引用保持可读：`[@key]`（引用 → `references.json`）、`[#fig:x]` / `[#tab:x]` / `[#eq:x]`（实体 → 对应文件）、`[^fn:N]`（脚注）、`**加粗**` / `*斜体*`（作者强调，保留）。

## 诚实与溯源

规则解析的原则：**提取出来的必须精确，提取不了的必须可见，绝不猜测。**

- `metadata.*_source` 记录每个字段的来源（`title_cmd` / `abstract_env` / `arxiv_api` / `none`）。
- 作者、投稿日期、`arxiv_version` 来自 arXiv 在论文 abs 页发布的元数据——每份 digest 都钉在具体论文版本上。
- 引用在源码带 `.bib` 时解析；只带编译后 `.bbl` 的论文保持 `title: null`，不编造。
- `paper.json.warnings` 逐篇报告残留的 LaTeX 命令，语料管线可据此过滤。

## Library API

```python
from arxiv2agent import digest, write_digest

paper = digest(arxiv_id="2305.13860")          # → dict（完整记录）
write_digest(paper, output_dir="./corpus/")
```

## 致谢

灵感来自 **arxiv-to-prompt**（本项目 vendor 了其 LaTeX 下载/展平管线，见 [NOTICE.md](NOTICE.md)）与 **DeepXiv**。

## License

MIT.
