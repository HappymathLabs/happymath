# MCM/ICM Partial Section Writing Reference

Use this document when the user only needs to write a specific section of the paper (e.g., Summary, Problem Restatement, Assumptions, Notations, References, Appendices).

## 1. Summary

### 1.1 Structure

1. Opening paragraph (2-3 sentences summarizing background, methods, and main conclusions).
2. Problem 1 paragraph: what was solved, what method was used, what result was obtained.
3. Problem 2 paragraph: same as above.
4. Problem 3 paragraph: same as above (continue for additional problems).
5. Closing paragraph: summarize the paper, highlight strengths, or mention possible extensions.

### 1.2 Writing Tips

- Write the summary last, after the full paper content is finalized.
- Be concise; avoid lengthy background descriptions.
- Highlight methods, results, and conclusions.
- Do not include formulas, figures, tables, or citations. Bold emphasis with `**...**` is allowed for key methods, numbers, and conclusions.

### 1.3 Keywords

- Format: **Keywords:** keyword1; keyword2; keyword3; keyword4
- 4-6 keywords, including main models or core concepts.

## 2. Restatement of the Problem

- Restate the problem in your own words; do not copy the original problem statement.
- Include background and specific sub-problems.
- Be concise and focus on the core issues to be solved.

## 3. Problem Analysis

- May be organized by problem (e.g., "Analysis of Problem 1").
- Include: problem information, condition analysis, overall approach, and intended methods.
- A flowchart is recommended to illustrate the thought process.
- Analyze only; do not present conclusions yet.
- Do not place Sensitivity Analysis, Error Analysis, Model Testing, Stability Testing, Model Evaluation, or result discussion inside Problem Analysis / Introduction. These belong to their own later sections or model sections.

## 4. Assumptions and Justifications

- List assumptions in numbered items.
- Common types:
  1. Conditions explicitly given in the problem.
  2. Exclusion of low-probability events.
  3. Focus on core factors while ignoring minor ones.
  4. Assumptions required by the model itself.
  5. Assumptions about parameter forms or distributions.
  6. Simplifications closely tied to the problem.
- Each assumption should be briefly justified.

## 5. Notations

- Use a three-line table.
- Columns: Symbol, Description, Unit (optional).
- Include only important variables; temporary variables may be omitted.
- Define each symbol again when it first appears in the text.

## 6. Sensitivity Analysis and Model Testing

- May be a separate section or embedded within each model section.
- **Sensitivity analysis**: vary one important parameter while holding others fixed; observe changes in results.
- **Error analysis**: identify sources of error or estimate error magnitude.
- **Stability testing**: assess robustness of results to input perturbations.

## 7. Model Evaluation and Further Discussion

### 7.1 Strengths

- 2-4 specific points.
- Consider model rationality, algorithm efficiency, and result reliability.

### 7.2 Weaknesses

- 1-2 points, fewer than strengths.
- Objectively note simplifications, limitations, or unconsidered factors.

### 7.3 Further Discussion / Improvements

- Suggest improvements for the identified weaknesses.
- Mention what could be done with more time or data.

### 7.4 Model Extension

- Extend the model to broader scenarios.
- Discuss practical application value.

## 8. References

- Number in order of appearance: [1], [2], [3].
- Use superscript citations in the text.
- Common formats:
  - Journal article: Author. Title[J]. Journal, Year, Vol(Issue): Pages.
  - Book: Author. Title[M]. City: Publisher, Year.
  - Thesis: Author. Title[D]. Institution, Year.
  - Conference paper: Author. Title[C]//Conference. City: Publisher, Year: Pages.
  - Online resource: Author. Title[EB/OL]. URL, Access date.
- Translate any Chinese references into English.

## 9. Appendices

- Start on a new page.
- Common contents:
  - List of supporting material files.
  - Source code (include language and purpose).
  - Detailed proofs or derivations.
  - Large flowcharts.
  - Complex tables or computational results.

## 10. Output Requirements

- Produce the corresponding `.md` file.
- If docx is needed, call `scripts/convert_md_to_docx.js`.
- Keep both md and docx formats.
- Write in English.
