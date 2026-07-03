# 安装指南

## 推荐安装方式：Conda

**Conda 是 HappyMath 的推荐安装方式**，可以获得最佳的依赖兼容性和完整的求解器支持。

```bash
conda install -c conda-forge happymath
```

## 备选安装方式：Pip

也可以直接使用 pip 安装：

```bash
pip install happymath
```

### Pip 安装的注意事项

使用 pip 安装时，以下组件不会默认包含：

- `ipopt` 求解器
- `lightgbm` 模型

这可能导致部分优化问题无法求解，或 AutoML 功能受限。建议通过 conda 补充安装：

```bash
conda install -c conda-forge ipopt
conda install -c conda-forge lightgbm
```

## 开发安装

如果你想从源码安装并进行开发：

```bash
git clone https://github.com/HappyMathLabs/happymath.git
cd happymath
pip install -e ".[dev]"
```

## 环境要求

- Python >= 3.11
- 核心依赖会自动安装，主要包括：
  - numpy, scipy, sympy, pandas
  - scikit-learn, pycaret, xgboost, catboost
  - pyomo, pymoo
  - matplotlib, seaborn
  - py-pde

## 验证安装

安装完成后，可以通过以下命令验证：

```python
import happymath
print(happymath.__version__)
```

正常情况下会输出类似 `0.2.0` 的版本号。
