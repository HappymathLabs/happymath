# Skills

This folder contains AI skills designed specifically for working with the `happymath` library.

Each skill lives in its own subdirectory and follows the standard skill format (`SKILL.md` with YAML frontmatter plus optional bundled resources under `scripts/`, `references/`, or `assets/`).

## Available skills

### `happymath_skill`

**File:** `happymath_skill/SKILL.md`

A comprehensive skill for installing, configuring, updating, learning, and using the `happymath` Python library. It covers:

- Environment setup with conda (and pip fallback)
- Version checking against PyPI
- Module selection across AutoML, Decision, DiffEq, and Opt
- Dynamic documentation and source-code lookup
- Code execution inside the correct conda environment
- Fallback strategies when `happymath` does not directly support a task

Use this skill whenever an AI assistant needs to help with `happymath`-related tasks, including mathematical modeling, machine learning, multi-criteria decision making, differential equations, optimization, and library environment management.

### `math-modeling-paper-writer`

**File:** `math-modeling-paper-writer/SKILL.md`

A skill for writing mathematical modeling competition papers. It covers:

- Automatic Chinese/English standard selection (CUMCM for Chinese, MCM/ICM for English)
- Full paper writing, partial section writing, and markdown-to-docx conversion
- Problem-specific "theory + results" structure
- Flowchart and Mermaid-based figure generation
- Three-line tables, formula conventions, and citation formatting
- Pre-flight checks and Word document generation

Use this skill whenever the user needs to write a math modeling paper, solution report, or convert a modeling paper from markdown to docx.
