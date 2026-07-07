# MCM/ICM Full Paper Template Reference

This document is derived from `美赛论文模版.docx` and defines the overall structure and formatting of an MCM/ICM competition paper. It specifies *which sections exist* and *how they are formatted*; the detailed writing style for each section is covered in other reference documents.

## 1. Overall Structure

An MCM/ICM paper is recommended to follow this order:

1. Summary Sheet header (generated during docx conversion; do not write it in markdown)
2. Title
3. Summary
4. Keywords
5. Table of Contents
6. Introduction
   - Problem Background
   - Restatement of the Problem
   - Literature Review (optional)
   - Our Work
7. Assumptions and Justifications
8. Notations
9. Model sections (one per problem/model)
   - The name of model 1
   - The name of model 2
   - ……
10. Sensitivity Analysis
11. Model Evaluation and Further Discussion
    - Strengths
    - Weaknesses
    - Further Discussion
12. Conclusion
13. References
14. Appendices

## 2. Page and Font Specifications

### 2.1 Page Setup

| Item | Specification |
|------|---------------|
| Paper | US Letter (8.5in × 11in) or A4 if required |
| Margins | Typically 2.5cm / 1in on all sides |
| Header | Usually none, or simple header |
| Footer | Centered page number |

### 2.2 Fonts and Sizes

| Element | Font | Size |
|---------|------|------|
| Body text | Times New Roman / Cambria Math | 12pt |
| Heading 1 | Times New Roman / Cambria Math, Bold | 16pt |
| Heading 2 | Times New Roman / Cambria Math, Bold | 14pt |
| Heading 3 | Times New Roman / Cambria Math, Bold | 12pt |
| Figure/Table captions | Times New Roman / Cambria Math, Bold | 11pt |
| References | Times New Roman / Cambria Math | 12pt |

### 2.3 Paragraph and Spacing

- Body text and Summary body: first-line indent, 1.15 line spacing.
- Heading 1: centered or left-aligned, with appropriate space before/after.
- Heading 2/3: left-aligned.
- Figures, captions, tables, and surrounding body text should keep compact but clear spacing. Captions are centered.

## 3. Section Format Requirements

### 3.1 Summary Sheet Header

Do not write the Summary Sheet header in the markdown source. The converter creates it only in the docx output when called with `--mcm-problem`, `--mcm-year`, and `--mcm-team`.

The generated header follows the current `mcmthesis` sheet layout: three top-aligned columns, bold black labels, red `\Large` problem/team values, and a 1.5pt full-width horizontal rule below the header.

| Problem Chosen | Year MCM/ICM Summary Sheet | Team Control Number |
|----------------|----------------------------|---------------------|
| C              | 2026                       | 1111111             |

### 3.2 Title

- Centered, bold, larger than Heading 1.
- Should reflect the problem or the main model.

### 3.3 Summary

- The summary is extremely important in MCM/ICM.
- Include background, problem overview, methods used, key results, and conclusions.
- Keep it concise but informative; judges weigh it heavily.
- Followed by **Keywords:** keyword1; keyword2; keyword3; keyword4.
- Do not include formulas, figures, tables, or citations in the Summary. Bold emphasis with `**...**` is allowed for key methods, numbers, and conclusions.

### 3.4 Table of Contents

- Auto-generated from headings.
- Hyperlinks to sections are recommended.

### 3.5 Introduction

Recommended subsections:

- **Problem Background**: summarize the problem context; orient it toward your approach.
- **Restatement of the Problem**: describe the problems in your own words.
- **Literature Review** (optional): summarize prior work; include only if you can do it well.
- **Our Work**: brief overview of modeling framework; often accompanied by a workflow diagram.
- Do not place Sensitivity Analysis, Error Analysis, Model Testing, Stability Testing, Model Evaluation, or result discussion inside Introduction / Problem Analysis. These belong to their own later sections or model sections.

### 3.6 Assumptions and Justifications

- List all assumptions explicitly.
- Provide justification for each assumption.
- This is weighted more heavily in MCM/ICM than in CUMCM.

### 3.7 Notations

- Use a three-line table listing key symbols, descriptions, and units.
- Optional: combine with Definitions if specialized terms need clarification.

### 3.8 Model Sections

- Each problem or model gets its own major section.
- Section titles should describe the model or question, not just "Model 1".
- Typical subsections:
  - Data Description
  - The Establishment of Model
  - The Solution of Model
- **Note**: In this skill, each problem is further split into "Theoretical Content" and "Result Content"; see `en_problem_writing_guide.md`.

### 3.9 Sensitivity Analysis

- Discuss how model outputs change with parameter variations.
- Include figures/tables showing sensitivity results.
- This section should not be a purely verbal claim that the model is robust.
- Common components include:
  - Sensitivity analysis: vary one important parameter while keeping other parameters fixed, then observe changes in model outputs.
  - Error analysis: identify data errors, parameter errors, numerical errors, or errors caused by model simplification.
  - Model testing: include model-specific tests when needed, such as consistency checks or distributional checks before model use.
  - Stability testing: test robustness under input perturbations, initial condition changes, or random seeds after model use.
- Prefer a figure or three-line table that directly supports the analysis.

### 3.10 Model Evaluation and Further Discussion

- **Strengths**: 2-4 items.
- **Weaknesses**: 1-2 items (fewer than strengths).
- **Further Discussion**: model improvements and extensions.
- Recommended subsection logic:
  - Strengths: discuss model rationality, computational efficiency, result stability, interpretability, or transferability.
  - Weaknesses: state objective limits such as data dependence, parameter sensitivity, simplifying assumptions, restricted scope, or algorithmic limitations.
  - Improvements: propose concrete remedies for the weaknesses, such as richer data, a multi-objective model, stochastic factors, or stronger optimization algorithms.
  - Extensions: explain where the model can be transferred and which data, parameters, or constraints must be adjusted.
- Avoid generic claims such as "the model is good" or "it has broad application value" unless they are tied to this paper's model or results.

### 3.11 Conclusion

- Restate main findings.
- Discuss implications and potential applications.

### 3.12 References

- Numbered in order of appearance: [1], [2], ...
- Translate any Chinese references into English.
- In-text citations use `[1]`, `[2]`, etc.; the converter renders them as superscripts.
- Each reference should occupy one paragraph and keep its manual number. Do not use footnotes, endnotes, BibTeX, or author-year citation style in the markdown source.
- Write each reference directly as `[1] Author...` under the References section. Do not use markdown ordered-list syntax for references.
- Standard academic citation formats are acceptable, but this skill recommends a simple numbered format for stable docx conversion:
  - Journal article: Author. Title[J]. Journal, Year, Vol(Issue): Pages.
  - Book: Author. Title[M]. City: Publisher, Year.
  - Thesis: Author. Title[D]. Institution, Year.
  - Conference paper: Author. Title[C]//Conference. City: Publisher, Year: Pages.
  - Online source: Author. Title[EB/OL]. URL, Access date.

### 3.13 Appendices

- Start on a new page.
- Include code, detailed derivations, data sources, large flowcharts, and additional results.
- MCM/ICM has page limits; keep appendices concise.

## 4. Figure and Table Specifications

### 4.1 Figures

- Caption below the figure: "Figure X-Y  Description" (e.g., "Figure 1-1  Workflow").
- Use clear labels, legends, and axis titles.
- MCM/ICM favors aesthetically pleasing, color-rich figures.

### 4.2 Tables

- Caption above the table: "Table X-Y  Description" (e.g., "Table 1-1  Notations").
- **Must use three-line table style**: thick top/bottom (1.5pt), thin header-bottom (0.75pt), no vertical lines.

### 4.3 Formulas

- Inline formulas: `$...$`.
- Displayed formulas use standalone `$$...$$` blocks. Do not manually type equation numbers in markdown; the converter generates right-aligned equation numbers.
- Keep each displayed formula focused on one core relation. Move long derivations to appendices.
- Avoid unverified complex LaTeX environments such as `align`, `cases`, `array`, and `tikzpicture`. If a piecewise function or matrix is necessary, inspect the generated docx manually.

## 5. Output Requirements

- Produce a `.md` file first.
- Then call `scripts/convert_md_to_docx.js` to generate `.docx`.
- Keep both formats.
- The entire paper must be in English.
