# MCM/ICM Result Chart Reference

Result-section charts display the data, conclusions, and insights obtained after solving the model. The core principle for MCM/ICM result charts is: **clear, visually appealing, data-centered, and color-friendly**.

## 1. Allowed Chart Types

Only charts that visualize the data/results themselves are allowed:

- **Line charts**: show trends, processes, time-series results.
- **Bar charts**: compare different categories.
- **Scatter plots**: show relationships or distributions between variables.
- **Pie charts**: show proportional composition (use sparingly; avoid too many categories).
- **Heatmaps**: show matrix-like data or spatial distributions.
- **Box plots**: show data distribution and outliers.
- **Tables**: present precise values, evaluation results, parameter settings, etc.

## 2. Prohibited Chart Types

The following chart types are **not allowed** in the result section:

- Flowcharts
- Algorithm diagrams
- Conceptual diagrams
- Any chart meant to show ideas rather than data

## 3. MCM/ICM Style Configuration

- **Color palette**: use clean, harmonious, and distinguishable colors. Recommended palettes include ColorBrewer Set2, Tableau 10, or similar professional palettes.
- **Background**: white background; light gray grid lines optional.
- **Font**: Times New Roman / Cambria Math; sizes consistent with body text.
- **Lines/markers**: clear and sufficiently thick.
- **Decorations**: avoid excessive shadows, gradients, or 3D effects.

## 4. matplotlib Example Configuration

```python
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, y1, color='#1f77b4', marker='o', label='Alternative A')
ax.plot(x, y2, color='#ff7f0e', marker='s', linestyle='--', label='Alternative B')
ax.set_xlabel('Parameter x', fontsize=11)
ax.set_ylabel('Metric y', fontsize=11)
ax.set_title('Comparison of Alternatives', fontsize=12)
ax.legend(frameon=True)
ax.grid(True, linestyle=':', linewidth=0.5)
plt.tight_layout()
plt.savefig('images/problem1_result.png', dpi=150)
```

## 5. Table Standards

- All tables must use the **three-line table** style.
- Caption above the table: "Table X-Y  Description".
- Numeric columns right-aligned; text columns left-aligned.
- No vertical lines; no extra horizontal lines.

Example markdown table:

```markdown
Table 1-1  Evaluation results for each alternative

| Alternative | Score | Rank |
|-------------|-------|------|
| A           | 0.85  | 1    |
| B           | 0.72  | 2    |
```

## 6. Chart Analysis Text

Every figure or table must be followed by an analysis paragraph covering:

1. What data the chart displays.
2. Key values or trends.
3. What these results mean for the current problem.
4. Whether the results validate the theoretical expectations.

Example:

```markdown
![Figure 1-2 Comparison of alternatives](images/problem1_result.png)

As shown in Figure 1-2, the metric y for Alternative A increases monotonically as parameter x increases, whereas Alternative B stabilizes after x > 0.8. At x = 1.0, Alternative A reaches 0.85, higher than Alternative B's 0.72, indicating that Alternative A performs better in this scenario.
```

## 7. Common Pitfalls

| Pitfall | Correct Approach |
|---------|------------------|
| Chart uses too many bright colors | Use a professional, limited palette |
| Missing axis labels | Add x-axis and y-axis labels with units |
| Multiple series are indistinguishable | Use distinct colors, line styles, and markers |
| Tables have vertical lines | Remove vertical lines; use three-line table style |
| No analysis after chart/table | Add an analysis paragraph |

## 8. Output Workflow

1. Generate images using Python matplotlib, R ggplot2, Excel, or similar tools.
2. Save as PNG at 150 dpi or higher.
3. Place in the `images/` directory.
4. Reference in markdown and immediately follow with analysis text.
