---
name: math-modeling-paper-writer
description: "撰写数学建模相关内容的技能。触发条件：用户要求写数学建模内容、数学建模国赛论文、数学建模美赛论文、MCM/ICM论文、数学建模竞赛 solution report、基于某道赛题撰写论文、把建模结果写成论文，或需要将 md 转换为 docx。根据用户语言自动选择规范：中文走国赛（CUMCM）规范，英文走美赛（MCM/ICM）规范。支持完整论文撰写、部分内容撰写、以及 md→docx 格式转换。"
---

# 数学建模竞赛论文撰写

根据用户语言自动选择规范：中文使用国赛规范，英文使用美赛规范。识别用户意图后，分别进入「完整论文撰写」「部分内容撰写」或「单纯格式转换」三种模式。

---

## 本地文件访问规则

本 Skill 的全部参考文档、示例论文和脚本均已包含在 Skill 目录内部，**所有内容都通过本地相对路径读取，不需要也不允许联网访问 Gitee 或 GitHub**。

本文件（`SKILL.md`）所在目录即为 Skill 根目录，目录结构如下：

```text
math-modeling-paper-writer/
├── SKILL.md                  # 本文件
├── package.json              # Node 依赖声明
├── references/               # 全部参考文档与获奖论文示例
└── scripts/                  # 全部转换、渲染、预检脚本
```

访问本 Skill 的参考文档、脚本时，直接读取 Skill 根目录下的相对路径。不得仅依据本 Skill 中的概述、标题、模型记忆或文件名执行写作、转换或渲染任务，必须实际完整读取对应的本地文件。

例如，中文完整论文模板对应本地路径：

```text
references/cn_full_paper_template_guide.md
```

---

## 环境准备与安装

执行本 Skill 前，先确认环境：

```bash
node -v && npm -v
```

- 若返回版本号且 Node.js >= 18，直接执行第 2 步。
- 若未安装或版本过低，先执行第 1 步安装 Node.js。

1. **安装 Node.js**

   - macOS：`brew install node`
   - Ubuntu/Debian：`sudo apt update && sudo apt install -y nodejs npm`
   - CentOS/RHEL：`sudo yum install -y nodejs npm`
   - Windows：`winget install OpenJS.NodeJS` 或 `choco install nodejs`

2. **安装项目依赖**（在本 Skill 根目录下执行）

   ```bash
   npm install
   ```

3. **安装 Mermaid 渲染依赖（Chromium）**

   本 Skill 在理论部分需要绘制流程图，流程图通过本地脚本 `scripts/render_mermaid.js` 调用 Mermaid CLI 渲染，Mermaid CLI 依赖 Chromium。请按以下任一方式配置：

   **方式 A：安装时自动下载 Chromium（下载失败时设置镜像）**

   ```bash
   export PUPPETEER_DOWNLOAD_BASE_URL=https://registry.npmmirror.com/-/binary/chrome-for-testing
   npm install
   ```

   **方式 B：使用系统已安装的 Chrome**

   ```bash
   PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=true npm install
   export PUPPETEER_EXECUTABLE_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
   ```

4. **验证安装**（在本 Skill 根目录下执行）

   ```bash
   node scripts/preflight_md.js <paper.md>
   node scripts/convert_md_to_docx.js <paper.md> --lang cn --output test.docx
   ```

   对应脚本源码：`scripts/preflight_md.js`、`scripts/convert_md_to_docx.js`。

---

## 快速开始

**完整论文（中文示例）**
```
请根据下面这道国赛题和建模结果，帮我写完整论文。题目：……，结果：……
```
→ 进入完整论文模式，按国赛模板组织全文，每个问题分「理论内容 + 结果内容」撰写。

**完整论文（英文示例）**
```
Write an MCM paper for Problem C. Here is the problem and our results: ...
```
→ 进入完整论文模式，按 MCM/ICM 模板组织全文。

**部分内容**
```
帮我写一下摘要 / 帮我写问题一的理论部分 / 把问题二的求解结果整理成文字 / 帮我把当前已有结果整理成论文形式
```
→ 进入部分内容模式，阅读对应参考文档后撰写。

**格式转换**
```
把这份 markdown 论文转成 docx / convert this md to docx
```
→ 进入格式转换模式，直接调用本地脚本 `scripts/convert_md_to_docx.js`。

## 触发条件

### 触发关键词

**中文**：数学建模论文、国赛论文、美赛论文、建模论文、写数模论文、撰写竞赛论文、数学建模竞赛论文、根据题目写论文、问题一/问题二论文、md转docx、markdown转word。

**English**: math modeling paper, MCM paper, ICM paper, mathematical modeling competition paper, write solution report, modeling competition thesis, convert md to docx, markdown to docx.

### 不触发场景

| 场景 | 应使用 |
|------|--------|
| 仅进行文献综述或研究搜索 | `deep-research` / `academic-research-skills` |
| 仅对已有论文进行同行评审 | `academic-paper-reviewer` |
| 通用学术论文（非数学建模竞赛） | `academic-paper` |

---

## 第零步：语言识别与规范选择

1. **语言识别**：判断用户输入（以及用户提供的赛题、结果等材料）的主要语言。
   - 若主要语言为中文 → 采用 **国赛（CUMCM）规范**。
   - 若主要语言为英文 → 采用 **美赛（MCM/ICM）规范**。
2. **规范锁定**：一旦确定语言/规范，全文撰写、图表风格、排版、最终 docx 模板均遵循该规范。不允许混用（例如中文论文中出现英文模板元素，或英文论文中出现中文模板元素）。

## 第一步：意图识别

根据用户表述和提供材料判断进入哪种模式：

| 模式 | 判断标准 |
|------|----------|
| **撰写完整论文** | 用户提供了较完整的赛题与建模结果，并明确要求撰写「完整论文」「全文」「solution report」等。 |
| **部分内容撰写** | 用户未提供完整赛题/结果，或仅要求撰写论文结构中的某部分（摘要、参考文献等），或仅要求撰写某个问题的理论/结果。 |
| **单纯格式转换** | 用户明确只要求把 markdown 文件转换为 docx，不需要额外撰写内容。 |

若用户意图模糊，先向用户确认：
- 是否需要撰写完整论文？
- 当前提供的材料是否完整？
- 最终是否需要生成 docx？

---

## 强制参考阅读门禁（所有模式）

**这是开始任何撰写或转换任务之前的硬性前置条件，优先级高于快速开始、用户的立即产出要求及下文各模式的步骤。**

1. 先根据已锁定的语言、模式和用户所需内容，从下方「模式参考清单」建立本次任务的逐项清单。仅选择对应语言的参考；未列入本次任务的另一语言参考无需阅读。
2. 对清单中的每一项，按「本地文件访问规则」从 Skill 目录完整读取对应的本地文件。阅读标题、本 Skill 的摘要、文件名、历史对话或模型记忆均不算完成阅读。
3. 只有清单的每一项都已完整读取后，才可开始撰写。开始撰写包括输出大纲、章节标题、正文、公式、表格、图表说明、伪代码、Markdown 文件和 docx 中的任何论文内容。不得边阅读边起草，也不得以占位内容绕过门禁。
4. 若一个规定参考的本地文件不存在或无法读取，停止撰写或转换，向用户报告具体参考项和失败情况；不得依据猜测、缓存摘要或其他非指定材料继续产出。
5. 在最终交付前，复核本次清单逐项完成。后续若用户把请求切换到另一种内容类型、语言或模式，重新建立并完成新清单后再继续。

### 模式参考清单

| 模式与请求类型 | 必须完整阅读的参考内容 | 允许开始的条件 |
|---|---|---|
| 模式 A：完整论文（中文） | 国赛完整论文模板；获奖论文风格参考；两篇国赛获奖论文示例；国赛问题撰写规范；国赛理论图表参考；国赛结果图表参考 | 以上 7 项全部完成 |
| 模式 A：完整论文（英文） | 美赛完整论文模板；获奖论文风格参考；两篇美赛获奖论文示例；美赛问题撰写规范；美赛理论图表参考；美赛结果图表参考 | 以上 7 项全部完成 |
| 模式 B：论文结构中的某部分（中文/英文） | 对应语言的部分内容撰写参考；获奖论文风格参考；对应语言的两篇获奖论文示例 | 以上 4 项全部完成 |
| 模式 B：某个问题的完整内容（中文/英文） | 对应语言的问题撰写规范；对应语言的理论图表参考；对应语言的结果图表参考 | 以上 3 项全部完成 |
| 模式 B：某个问题的理论部分（中文/英文） | 对应语言的问题撰写规范；对应语言的理论图表参考 | 以上 2 项全部完成 |
| 模式 B：某个问题的结果部分（中文/英文） | 对应语言的问题撰写规范；对应语言的结果图表参考 | 以上 2 项全部完成 |
| 模式 C：单纯格式转换 | md 转 docx 详细说明 | 完整读取后才可运行脚本、预检或转换 |

「两篇获奖论文示例」是获奖论文风格参考中按语言指定的两篇完整 `.md` 示例，必须与该索引一并读取，不得只读索引而跳过示例。若完整论文或部分内容任务需要生成 docx，还必须在执行转换前完成模式 C 的阅读门禁。

---

## 模式 A：撰写完整论文

### A.1 阅读完整论文模板

先完成「强制参考阅读门禁」中模式 A 对应语言的全部 7 项，再根据语言选择并阅读对应模板参考：
- 中文：`references/cn_full_paper_template_guide.md`
- 英文：`references/en_full_paper_template_guide.md`
- 获奖论文风格参考：`references/prized_paper_style_guide.md`

模板参考明确：
- 论文章节安排（摘要、问题重述、问题分析、模型假设、符号说明、模型的建立与求解、模型评价、参考文献、附录等）。
- 字体、字号、行距、页边距、标题样式、图表标题位置、页码等排版要求。
- 问题章节以「问题一 / 问题二 / Problem 1 / Problem 2」形式组织。

获奖论文示例只用于参考各模块写作形式与风格，不复制内容；PDF 转 md 造成的格式错误可以忽略。模板参考**不涉及**每个章节的具体撰写风格，因此继续阅读以下参考文档。未完成 A.2、A.3 及获奖论文示例的全部阅读前，不得开始 A.4。

### A.2 阅读问题撰写规范

阅读对应语言的问题撰写规范：
- 中文：`references/cn_problem_writing_guide.md`
- 英文：`references/en_problem_writing_guide.md`

该规范明确：
- 每个问题拆分为 **理论内容** 与 **结果内容** 两部分。
  - 理论内容要求：简洁、精炼、可视化；建模/方法/算法运行路径用流程图，算法伪代码或算法步骤用算法文字图（三线表）。
- 结果内容要求：简洁、精炼，与理论配套，使用折线图、柱状图、表格等可视化结果，并在每张图表后紧跟分析文字。

### A.3 阅读图表绘制参考

- 理论部分图表：
  - 中文：`references/cn_theory_chart_guide.md`
  - 英文：`references/en_theory_chart_guide.md`
- 结果部分图表：
  - 中文：`references/cn_result_chart_guide.md`
  - 英文：`references/en_result_chart_guide.md`

绘制任何图表前，必须先阅读对应参考。流程图、算法流程图、结构图和示意图应先保存 Mermaid 源文件，再渲染为图片并放入 markdown 同级 `images/` 目录，在 md 中用标准 markdown 图片语法引用。算法文字图优先写成 markdown 表格，由转换脚本输出三线表。

同一张 Mermaid 图只能维护一份源代码：如果写作说明中展示 Mermaid 代码，就把同一段代码保存为 `.mmd` 文件并渲染；如果最终 markdown 只引用图片，也必须保留对应 `.mmd` 源文件。禁止 markdown 预览图和 `.mmd` 渲染图使用不同结构。

### A.4 分问题循环撰写

仅在模式 A 阅读门禁完成后执行本节。

对每个问题依次执行：

1. **撰写理论内容**
   - 说明本题建模思路、使用的方法/模型。
   - 必须配有图表：建模流程、方法流程或算法运行路径用流程图；算法伪代码或算法步骤用算法文字图（三线表）。
   - 图表绘制前阅读理论部分图表绘制参考：`references/cn_theory_chart_guide.md`（国赛）或 `references/en_theory_chart_guide.md`（美赛）。
   - 文字简洁，不堆砌公式；核心公式保留。

2. **撰写结果内容**
   - 呈现本题求解/分析结果。
   - 必须与理论内容配套：结果是理论的实践。
   - 使用折线图、柱状图、散点图、表格等可视化结果，**不得**使用流程图、算法图。
   - 每张图表后必须紧跟分析文字。
   - 图表绘制前阅读结果部分图表绘制参考：`references/cn_result_chart_guide.md`（国赛）或 `references/en_result_chart_guide.md`（美赛）。

3. 重复上述两步直至所有问题撰写完毕。

### A.5 组装完整论文

按对应模板的章节顺序组织全文：中文使用 `references/cn_full_paper_template_guide.md`；英文使用 `references/en_full_paper_template_guide.md`。将摘要、问题重述、问题分析、模型假设、符号说明、各问题的理论+结果、模型评价、参考文献、附录等组合成完整 markdown 文件。

硬性边界：
- 摘要 / Summary 不写公式、图表或引用，只能用文字、关键数据和结论表达；允许用 `**...**` 对摘要正文中的关键方法、关键数值或关键结论加粗。
- 问题分析只分析题目文本、条件、约束和每个具体问题的解题思路。若题目有几个问题，就只设置几个对应小标题，例如「问题一的分析」「问题二的分析」。灵敏度分析、误差分析、模型检验、稳定性检验、模型评价不得放入问题分析。
- 美赛 markdown 源文件不手写 Summary Sheet / Problem Chosen / Team Control Number 信息栏；该信息栏只在 `scripts/convert_md_to_docx.js` 转换为 docx 时通过 `--mcm-problem`、`--mcm-year`、`--mcm-team` 参数生成。

### A.6 输出与质量门禁

- 生成 `{论文标题}.md`。
- 若使用 Mermaid 流程图，先调用本地脚本 `scripts/render_mermaid.js`，将 `.mmd` 渲染为 `images/*.png`。
- 调用本地脚本 `scripts/preflight_md.js` 检查 markdown：不允许残留 Mermaid 代码块、缺失图片、占位图、公式分隔符不平衡等问题。
- 预检通过后调用本地脚本 `scripts/convert_md_to_docx.js` 生成 `{论文标题}.docx`。
- 保留 md 与 docx 两种格式。
- 中文论文全部使用中文；英文论文全部使用英文。

---

## 模式 B：部分内容撰写

### B.1 判断用户想要的内容类型

先按「强制参考阅读门禁」确定并完成该行规定的全部参考阅读，再执行下表对应的撰写工作。

| 用户请求 | 处理方式 |
|----------|----------|
| 撰写论文结构中的某部分（摘要、问题重述、模型假设、符号说明、参考文献、附录等） | 阅读对应语言的部分内容撰写参考：`references/cn_partial_sections_guide.md`（国赛）或 `references/en_partial_sections_guide.md`（美赛）。并阅读 `references/prized_paper_style_guide.md` 中对应语言示例路径，按该参考撰写。 |
| 撰写某个问题的完整内容 | 按「模式 A」中单个问题的流程（理论 + 结果）撰写。先阅读问题撰写规范：`references/cn_problem_writing_guide.md`（国赛）或 `references/en_problem_writing_guide.md`（美赛）。再阅读理论图表参考：`references/cn_theory_chart_guide.md`（国赛）或 `references/en_theory_chart_guide.md`（美赛）。最后阅读结果图表参考：`references/cn_result_chart_guide.md`（国赛）或 `references/en_result_chart_guide.md`（美赛）。 |
| 只撰写某个问题的理论部分 | 阅读问题撰写规范：`references/cn_problem_writing_guide.md`（国赛）或 `references/en_problem_writing_guide.md`（美赛）。再阅读理论图表参考：`references/cn_theory_chart_guide.md`（国赛）或 `references/en_theory_chart_guide.md`（美赛），只输出理论内容。 |
| 只撰写某个问题的结果部分 | 阅读问题撰写规范：`references/cn_problem_writing_guide.md`（国赛）或 `references/en_problem_writing_guide.md`（美赛）。再阅读结果图表参考：`references/cn_result_chart_guide.md`（国赛）或 `references/en_result_chart_guide.md`（美赛），只输出结果内容（仍需要知道理论配套关系，因此也要快速浏览问题撰写规范）。 |

### B.2 输出

- 生成 `{内容名称}.md`。
- 若用户需要 docx，调用本地脚本 `scripts/convert_md_to_docx.js` 生成对应 docx。
- 若内容包含 Mermaid 或图片，先运行本地脚本 `scripts/render_mermaid.js`，以及 `scripts/preflight_md.js`。
- 中文内容使用中文；英文内容使用英文。

---

## 模式 C：单纯格式转换

1. 完成模式 C 的阅读门禁：完整阅读 `references/md_to_docx_guide.md`。在此之前不得运行脚本、预检或转换。
2. 确认源文件路径与目标路径。
3. 若 markdown 中有 Mermaid 代码块，先要求生成或补齐对应 `.mmd` 源文件并用本地脚本 `scripts/render_mermaid.js` 渲染为图片；不要让 Mermaid 原始代码块进入转换。
4. 运行本地脚本 `scripts/preflight_md.js`。
5. 根据文件语言选择国赛或美赛 docx 模板参数，调用本地脚本 `scripts/convert_md_to_docx.js` 转换。
6. 输出 docx 文件。

---

## 图表通用规则

### 国赛图表原则
- 简洁、简约、黑白线条。
- 流程图：使用 Mermaid `flowchart`，白色填充、黑色描边、黑色文字，避免阴影与渐变。
- 算法文字图：使用 markdown 表格写成三线表，优先列为「步骤 / 操作」。
- 表格：一律使用 **三线表**（顶线 1.5pt、底线 1.5pt、栏目线 0.75pt，无竖线）。
- 结果图：matplotlib 默认配色尽量克制，优先黑白或单色，线型/标记区分数据系列。

### 美赛图表原则
- 美观、多配色。
- 流程图：可使用彩色节点、圆角矩形、清晰配色方案。
- 算法文字图：使用 markdown 表格写成 three-line table，优先列为 `Step / Operation`。
- 表格：同样使用 **三线表**。
- 结果图：使用清晰、协调的配色，标注完整（标题、轴标签、图例），符合 MCM/ICM 视觉风格。

### 图表引用
- md 中使用 `![图 1-1 描述](images/xxx.png)` 或 `![Figure 1-1 Description](images/xxx.png)`。
- 图标题置于图片下方；表标题置于表格上方。
- 图表编号按章节分别计数（图 1-1、图 1-2、表 2-1 等）。

---

## Markdown 写作规则

1. **章节标题**：使用 markdown `#`、`##`、`###`，对应论文一级、二级、三级标题。
2. **公式**：行内用 `$...$`，行间用 `$$...$$`。
3. **表格**：使用 markdown 表格语法；转换脚本会自动处理为三线表。
4. **引用**：中文论文使用 `[1]` 上标引用；英文论文同样使用 `[1]` 上标引用。
5. **图片**：统一放在 `images/` 目录，使用相对路径引用。
6. **Mermaid**：不在最终 markdown 中保留 Mermaid 代码块；必须先渲染成图片再引用。
7. **算法文字图**：使用 markdown 表格，不使用代码块或 Mermaid 代替。
8. **分页**：在 md 中不需要显式分页，转换脚本会在摘要后、参考文献前等关键位置自动分页。
9. **标题编号**：markdown 的 `##` 章节标题可保留国赛「一、」或美赛「1」等章节编号；`###`、`####` 子标题会由转换脚本自动编号。若子标题中已有 `2.1`、`5.1` 等编号，转换脚本会先剥离再自动编号，避免重复编号。
10. **美赛参赛信息栏**：不要在 markdown 中写 `Problem Chosen / MCM/ICM Summary Sheet / Team Control Number` 表格；转换 docx 时由脚本统一生成。

---

## 参考文档索引

以下路径均为本 Skill 根目录下的本地相对路径，直接读取本地文件即可。

| 文档 | 本地路径 | 用途 |
|------|----------|------|
| 国赛完整论文模板 | `references/cn_full_paper_template_guide.md` | 国赛完整论文模板（章节、排版、格式） |
| 美赛完整论文模板 | `references/en_full_paper_template_guide.md` | 美赛完整论文模板（章节、排版、格式） |
| 国赛问题撰写规范 | `references/cn_problem_writing_guide.md` | 国赛单个问题撰写规范（理论+结果） |
| 美赛问题撰写规范 | `references/en_problem_writing_guide.md` | 美赛单个问题撰写规范（理论+结果） |
| 国赛理论图表参考 | `references/cn_theory_chart_guide.md` | 国赛理论部分图表绘制参考 |
| 美赛理论图表参考 | `references/en_theory_chart_guide.md` | 美赛理论部分图表绘制参考 |
| 国赛结果图表参考 | `references/cn_result_chart_guide.md` | 国赛结果部分图表绘制参考 |
| 美赛结果图表参考 | `references/en_result_chart_guide.md` | 美赛结果部分图表绘制参考 |
| 国赛部分内容撰写参考 | `references/cn_partial_sections_guide.md` | 国赛论文结构各部分撰写参考（摘要、参考文献等） |
| 美赛部分内容撰写参考 | `references/en_partial_sections_guide.md` | 美赛论文结构各部分撰写参考（Summary、References 等） |
| 获奖论文风格参考 | `references/prized_paper_style_guide.md` | 国赛/美赛获奖论文示例路径与模块风格参考规则 |
| 国赛获奖论文示例 1 | `references/CUMCM_Prized_Papers/CUMCM_2023_B_B226/CUMCM_2023_B_B226.md` | 国赛获奖论文完整示例 |
| 国赛获奖论文示例 2 | `references/CUMCM_Prized_Papers/CUMCM_2023_C_C228/CUMCM_2023_C_C228.md` | 国赛获奖论文完整示例 |
| 美赛获奖论文示例 1 | `references/MCM_ICM_Prized_Papers/MCM_ICM_2025_A_2500836/MCM_ICM_2025_A_2500836.md` | 美赛获奖论文完整示例 |
| 美赛获奖论文示例 2 | `references/MCM_ICM_Prized_Papers/MCM_ICM_2025_C_2507817/MCM_ICM_2025_C_2507817.md` | 美赛获奖论文完整示例 |
| md 转 docx 详细说明 | `references/md_to_docx_guide.md` | md 转 docx 的详细说明与脚本使用 |

---

## 脚本清单

以下脚本均位于本 Skill 根目录的 `scripts/` 目录下，直接以 `node scripts/<脚本名>` 方式调用。

| 脚本 | 本地路径 | 用途 |
|------|----------|------|
| `convert_md_to_docx.js` | `scripts/convert_md_to_docx.js` | 将 markdown 论文转换为 docx，支持国赛/美赛两种排版 |
| `mathml-to-docx.js` | `scripts/mathml-to-docx.js` | MathML → Word 原生公式（被 convert 脚本依赖） |
| `render_mermaid.js` | `scripts/render_mermaid.js` | 将 `.mmd` Mermaid 源文件渲染为 PNG/SVG/PDF 图片 |
| `preflight_md.js` | `scripts/preflight_md.js` | 转换前检查 markdown 中的图片、Mermaid、公式分隔符和引用 |

---

## 反模式

| # | 反模式 | 正确做法 |
|---|--------|----------|
| 1 | 中英文规范混用 | 根据识别出的语言全程只使用一种规范 |
| 2 | 理论部分缺少图表 | 流程用流程图，算法步骤用算法文字图（三线表） |
| 3 | 结果部分使用流程图 | 结果部分只用数据可视化图表 |
| 4 | 图表后没有分析文字 | 每张结果图表后必须紧跟结果分析 |
| 5 | 表格使用普通边框 | 所有表格必须使用三线表 |
| 6 | 未阅读图表参考就绘制 | 绘制前必须先阅读对应语言的理论/结果图表参考 |
| 7 | 只输出 docx 不保留 md | 必须同时保留 md 与 docx |
| 8 | 虚构引用或结果 | 所有引用和结果必须基于用户提供的材料 |
| 9 | 使用占位流程图 | 必须渲染真实 Mermaid 源文件，并通过 preflight |
| 10 | 最终 markdown 残留 Mermaid 代码块 | 先渲染为图片，再用图片语法引用 |
| 11 | 未完成本模式全部参考阅读就开始输出 | 先完成「强制参考阅读门禁」的逐项清单，再撰写、预检或转换 |
| 12 | 联网访问 Gitee/GitHub 获取参考文档或脚本 | 所有参考文档和脚本均从本 Skill 目录的本地相对路径读取 |

---

## 输出语言

- 中文输入 → 中文论文（国赛规范）。
- 英文输入 → 英文论文（美赛规范）。
- 术语保留英文（如 MCM、ARIMA、CNN），但解释用对应语言。

---

## 示例文件

当前仓库未提供独立示例文件；请直接阅读「参考文档索引」中的完整论文模板、部分内容撰写参考和获奖论文风格参考。
