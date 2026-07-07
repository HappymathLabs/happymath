# Markdown 转 DOCX 指南

本 skill 使用 `scripts/convert_md_to_docx.js` 将数学建模论文的 markdown 文件转换为符合国赛或美赛格式规范的 `.docx` 文件。

## 1. 安装依赖

在 skill 目录下执行：

```bash
cd math-modeling-paper-writer
npm install docx temml fast-xml-parser
```

若需要把 Mermaid 流程图渲染为图片，还需要安装 Mermaid CLI 相关依赖：

```bash
npm install @mermaid-js/mermaid-cli
```

## 2. 基本用法

### 2.1 中文论文（国赛格式）

```bash
node scripts/convert_md_to_docx.js input.md --lang cn --output output.docx
```

### 2.2 英文论文（美赛格式）

```bash
node scripts/convert_md_to_docx.js input.md --lang en --output output.docx
```

### 2.3 自动检测语言

```bash
node scripts/convert_md_to_docx.js input.md --output output.docx
```

脚本会根据正文中中文字符的比例自动判断使用 `cn` 还是 `en`。

### 2.4 美赛摘要页眉

若需要生成 MCM/ICM 的摘要页眉（包含 Problem Chosen、Year、Team Control Number），可添加参数：

```bash
node scripts/convert_md_to_docx.js input.md --lang en \
  --mcm-problem C --mcm-year 2026 --mcm-team 1111111 \
  --output output.docx
```

美赛 markdown 源文件不要手写这张信息栏表格；如果源文件顶部已经包含 `Problem Chosen / MCM/ICM Summary Sheet / Team Control Number` 表格，`preflight_md.js` 会报错，避免 docx 中出现两份参赛信息栏。

转换脚本会按 `mcmthesis` 的 Summary Sheet 视觉生成：三栏顶端对齐，黑色加粗标签，红色 `\Large` 题号/队号，下方 1.5pt 全宽横线。

## 3. Markdown 写法规范

### 3.1 标题与章节层级

脚本按以下规则解析 markdown 标题：

| Markdown | 输出 | 说明 |
|----------|------|------|
| `# 论文标题` | 居中论文标题 | 只应出现一次 |
| `## 摘要` / `## Abstract` | 居中摘要标题 | 特殊处理，不计入章节编号 |
| `## 一、引言` / `## 1 Introduction` | 一级标题（章节） | 会触发章节计数器 |
| `### ...` | 二级标题（1.1、1.2） | 自动编号 |
| `#### ...` | 三级标题（1.1.1） | 自动编号 |

**推荐结构示例（中文）：**

```markdown
# 基于遗传算法的某优化问题研究

## 摘要
...
关键词：...

## 一、问题重述
...

## 二、问题分析
...

## 三、模型假设
...

## 四、符号说明

表 1-1  符号说明

| 符号 | 说明 |
|------|------|
| ...  | ...  |

## 五、模型的建立与求解

### 5.1 问题一

#### 理论内容
...

#### 结果内容
...
```

**推荐结构示例（英文）：**

```markdown
# An Optimization Study Based on Genetic Algorithm

## Abstract
...
Keywords: ...

## 1 Introduction

### 1.1 Problem Background
...

## 2 Assumptions and Justifications
...

## 3 Notations

Table 1-1  Notations

| Symbol | Description |
|--------|-------------|
| ...    | ...         |

## 4 Model for Problem 1

### Theoretical Content
...

### Result Content
...
```

> 注意：脚本将所有 `##` 标题视为论文章节（摘要除外），并按出现顺序建立图、表、公式的章节计数。`##` 标题中的「一、」或「1」等章节编号会保留；`###`、`####` 标题由脚本自动编号。若源 markdown 子标题中已有 `2.1`、`5.1` 等编号，脚本会先剥离编号再生成 Word 编号，避免出现 `2.12.1 方法说明` 这类重复编号。图、表编号会按章节分别计数。

### 3.2 摘要

```markdown
## 摘要

开头段……

问题一中……

关键词：关键词1 关键词2 关键词3
```

或英文：

```markdown
## Abstract

... ...

Keywords: keyword1; keyword2; keyword3
```

脚本会在关键词后自动插入分页符。

摘要 / Summary 中禁止出现公式、图、表和参考文献引用；只能使用文字、关键数据和结论表达。可以用 `**...**` 对关键方法、关键数据或核心结论加粗强调。

### 3.3 公式

- 行内公式：`$E=mc^2$`
- 行间公式：
  ```markdown
  $$
  E = mc^2
  $$
  ```
- 不要在 markdown 公式后手写 `(1)`、`(2)`。转换脚本会把每个 `$$...$$` 自动转换为三列表格布局：中间为 Word 原生公式，右侧为公式编号。
- 推荐使用 Temml 能稳定处理的基础 LaTeX：上下标、分式、根号、求和、积分、希腊字母、常见关系符。
- 谨慎使用复杂环境：`align`、`cases`、`array`、`matrix`、`tikzpicture`、自定义宏。若必须使用，转换后必须打开 docx 检查。
- 一个 `$$...$$` 块只放一个核心公式或一组非常短的等价变形。长推导放入附录。

### 3.4 表格

```markdown
表 1-1  符号说明

| 符号 | 说明 |
|------|------|
| S    | 状态空间 |
| A    | 动作空间 |
```

脚本会自动将表格转换为三线表。表标题必须放在表格上方，格式为「表 章-序号  表名」或「Table X-Y Description」。

### 3.5 图片

```markdown
![问题一流程图](images/problem1_flow.png)
```

图片路径相对于 markdown 文件。脚本会自动在图片下方添加图标题「图 X-Y  描述」。若 alt 文本中已包含「图 X-Y」前缀，会自动去除避免重复。

脚本会读取 PNG/JPEG/GIF/BMP/SVG 的自然尺寸，并按页面可用空间等比例缩放；不要预先把图片压成固定比例，也不要用 Word 手工拉伸图片。

Mermaid 代码块不会被 `convert_md_to_docx.js` 直接转换。正确流程是：

1. 将 Mermaid 源码保存为 `.mmd` 文件，建议放在 `diagrams/` 目录。
2. 运行 `node scripts/render_mermaid.js diagrams/problem1_flow.mmd --output images/problem1_flow.png`。
3. 在 markdown 中只引用渲染后的图片。
4. 转换前运行 `node scripts/preflight_md.js paper.md`，确认图片存在且没有未渲染的 Mermaid 代码块。

### 3.6 引用

正文中使用 `[1]`、`[2]` 上标引用。参考文献区每条文献单独成段，保留手写编号：

```markdown
## 参考文献

[1] 张三. 遗传算法在优化问题中的应用[J]. 数学建模学报, 2020, 1(1): 10-20.
[2] 李四. 优化模型[M]. 北京: 科学出版社, 2021.
```

英文论文同样使用编号制：

```markdown
## References

[1] Smith J, Doe A. Genetic algorithms in optimization[J]. Journal of Mathematical Modeling, 2020, 1(1): 10-20.
```

不要混用脚注、尾注、BibTeX、作者年份制或链接裸列。正文中出现的编号必须能在参考文献区找到对应条目。

### 3.7 算法文字图

论文中的“算法图”指三线伪代码表，不是流程图。推荐写成 markdown 表格，转换后会得到三线表：

```markdown
表 2-1  遗传算法求解步骤

| 步骤 | 操作 |
|------|------|
| 1 | 初始化种群规模、交叉概率和变异概率 |
| 2 | 计算每个个体的适应度函数值 |
| 3 | 按适应度执行选择、交叉和变异操作 |
| 4 | 若达到最大迭代次数，则输出最优个体；否则返回步骤 2 |
```

英文论文使用：

```markdown
Table 2-1  Genetic algorithm procedure

| Step | Operation |
|------|-----------|
| 1 | Initialize population size, crossover probability, and mutation probability |
| 2 | Evaluate the fitness value of each individual |
| 3 | Perform selection, crossover, and mutation |
| 4 | If the stopping criterion is met, output the best individual; otherwise return to Step 2 |
```

## 4. 页面与字体规范

### 4.1 国赛（cn）

- 纸张：A4
- 页边距：2.5 cm
- 正文字体：宋体 12pt
- 标题字体：黑体
- 行距：单倍
- 首行缩进：2 字符

### 4.2 美赛（en）

- 纸张：US Letter
- 页边距：1 inch
- 正文字体：Times New Roman 12pt
- 标题字体：Times New Roman Bold
- 行距：1.15 倍
- 首行缩进：按模板正文缩进设置

## 5. 注意事项

1. 所有图片需提前保存到 `images/` 目录。
2. Mermaid 代码块会被跳过，需先生成对应的图片再引用。
3. 转换前运行 `node scripts/preflight_md.js input.md`。
4. 若转换失败，请检查 Node.js 版本（建议 ≥ 16）以及依赖是否安装完整。
5. 公式转换依赖 `temml`，复杂公式可能存在兼容性问题，生成 docx 后请打开检查。
6. 摘要含公式、图表、引用，中文问题分析含非具体问题小标题，美赛 markdown 手写 Summary Sheet，都会被 preflight 拦截。

## 6. 脚本参数速查

| 参数 | 说明 |
|------|------|
| `--lang cn` / `--lang en` | 指定语言/规范 |
| `--output <path>` | 输出 docx 路径 |
| `--mcm-problem <letter>` | 美赛题目编号 |
| `--mcm-year <year>` | 美赛年份 |
| `--mcm-team <number>` | 美赛队伍控制号 |

## 7. 推荐转换流程

```bash
node scripts/render_mermaid.js diagrams/problem1_flow.mmd --output images/problem1_flow.png
node scripts/preflight_md.js paper.md
node scripts/convert_md_to_docx.js paper.md --lang cn --output paper.docx
```

若 `preflight_md.js` 报错，应先修复 markdown、图片或公式，再生成 docx。
