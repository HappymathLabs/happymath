# MCM/ICM Single-Problem Writing Guide

This document defines how each specific problem should be written in an MCM/ICM paper. Every problem must be divided into **Theoretical Content** and **Result Content**.

## 1. Overall Structure

Under each problem section, use two subsections:

```markdown
## Model for Problem 1

### Theoretical Content

(Modeling idea, method introduction, flowchart/algorithm flowchart/algorithm text diagram)

### Result Content

(Solution results, data visualization, result analysis)
```

Section titles may be adjusted to reflect the actual content (e.g., "A Optimization Model for Problem 1"), but the internal "theory + result" structure must remain.

## 2. Theoretical Content

### 2.1 Objective

Introduce the main methods used to solve the problem in a concise, precise, and visual way. Describe the solution logic clearly and intuitively so judges can quickly grasp the modeling approach.

### 2.2 Required Elements

- **Method overview**: 1-2 paragraphs explaining which model/method is used and why.
- **Core formulas**: Keep 1-3 key formulas that capture the essence of the model.
- **Visual diagrams**:
  - Modeling or method process → flowchart
  - Algorithm execution path → algorithm flowchart
  - Algorithm pseudocode or procedure → algorithm text diagram as a three-line table
- Before drawing any chart, read `en_theory_chart_guide.md`.

### 2.3 Writing Principles

- Do not pile up excessive formulas and definitions.
- Do not copy textbook-style long derivations.
- Focus on "how I use this model to solve this specific problem."
- Every figure must be accompanied by explanatory text describing its role in the overall approach.

### 2.4 Example Structure

```markdown
### Theoretical Content

For Problem 1, we abstract the phenomenon in the problem as an optimization problem. First, decision variables are defined; second, the objective function is formulated according to the constraints; finally, a specific algorithm is employed for solution.

The decision variables are defined as:

$$x_i = \cdots$$

The objective function is:

$$\min Z = \cdots$$

The solution process is shown in Figure 1-1.

![Figure 1-1 Solution flowchart for Problem 1](images/problem1_flow.png)

As shown in Figure 1-1, the solution process consists of three steps: ...

The algorithm procedure is summarized in Table 1-1.

Table 1-1  Genetic algorithm procedure

| Step | Operation |
|------|-----------|
| 1 | Initialize the population and algorithm parameters |
| 2 | Evaluate fitness values and perform genetic operations |
| 3 | Check the stopping criterion and output the best solution |
```

## 3. Result Content

### 3.1 Objective

Present the solution or analysis results for this problem in a concise and precise manner. The result content must be the practical implementation of the theoretical content, showing the data, conclusions, and insights obtained after running the model.

### 3.2 Required Elements

- **Data/result presentation**: Use tables, line charts, bar charts, scatter plots, heatmaps, etc.
- **Result analysis**: Every figure or table must be followed by an analysis paragraph explaining what the chart shows, whether the results are as expected, and how it answers the problem.
- **Visualizations**:
  - Flowcharts and algorithm diagrams are **not allowed**.
  - Use only data-visualization charts that display the results themselves.
- Before drawing any chart, read `en_result_chart_guide.md`.

### 3.3 Writing Principles

- Results and analysis must correspond one-to-one.
- Do not present data tables without explanation.
- Do not present figures without conclusions.
- Highlight the direct answer to the problem.

### 3.4 Example Structure

```markdown
### Result Content

Using the above model, we obtain the evaluation results for each alternative, as shown in Table 1-1.

Table 1-1  Evaluation results for each alternative

| Alternative | Score | Rank |
|-------------|-------|------|
| A           | 0.85  | 1    |
| B           | 0.72  | 2    |

Table 1-1 shows that Alternative A achieves the highest score. Furthermore, Figure 1-2 illustrates the stability of Alternative A under different parameters.

![Figure 1-2 Sensitivity analysis results](images/problem1_sensitivity.png)

From Figure 1-2, the ranking of Alternative A remains stable when the parameter varies within [0.5, 1.5], indicating strong robustness of the model result.
```

## 4. Alignment Between Theory and Results

- Theory answers "what method is used, why it is used, and how it works."
- Results answer "what was obtained by using the method and what it means."
- The two parts must correspond directly; avoid disconnect between theory and results.
- If results do not match theoretical expectations, analyze the reasons in the result section rather than altering the theoretical description to fit the results.

## 5. Common Pitfalls

| Pitfall | Correct Approach |
|---------|------------------|
| Theory section has only text, no figures/tables | Add a flowchart, algorithm flowchart, or algorithm text diagram |
| Result section has only data, no analysis | Add an analysis paragraph after every figure/table |
| Result section uses a flowchart | Replace with line charts, bar charts, scatter plots, heatmaps, or tables |
| Theory section piles up formulas | Keep only core formulas; move details to appendix |
| Theoretical derivation is too long | Summarize the idea with figures; put details in appendix |
| Algorithm pseudocode is drawn as a flowchart | Use a flowchart only for execution paths; use a three-line table for pseudocode-style algorithm steps |
