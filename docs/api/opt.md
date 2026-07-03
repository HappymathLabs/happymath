# API 参考 - Opt

本页列出 `happymath.Opt` 模块中优化问题建模、求解与结果获取的主要类。

## 优化主入口

::: happymath.Opt.OptModule.OptModule
    options:
      members:
        - __init__
        - solve
      show_root_heading: true
      show_root_toc_entry: false

## 优化结果容器

::: happymath.Opt.results.opt_result.OptResult
    options:
      members:
        - success
        - variables
        - objective_value
        - solution
        - all_solutions
        - all_solvers
        - raw_solution
        - raw_all_solutions
        - solver
        - message
        - solver_name
        - pareto_front
      show_root_heading: true
      show_root_toc_entry: false

## 优化基类

::: happymath.Opt.opt_core.opt_base.OptBase
    options:
      members:
        - __init__
        - parse_result
        - pyomo_problem_type
        - pymoo_problem_type
      show_root_heading: true
      show_root_toc_entry: false

## Pyomo 求解器

::: happymath.Opt.solvers.pyomo_solver.PyomoSolver
    options:
      members:
        - __init__
        - solve
        - get_available_solvers
      show_root_heading: true
      show_root_toc_entry: false

## Pymoo 求解器

::: happymath.Opt.solvers.pymoo_solver.PymooSolver
    options:
      members:
        - __init__
        - solve
        - get_available_solvers
      show_root_heading: true
      show_root_toc_entry: false

## 表达式处理器

::: happymath.Opt.opt_expr.processor.ExpressionProcessor
    options:
      members:
        - process
      show_root_heading: true
      show_root_toc_entry: false

## 解析结果

::: happymath.Opt.opt_expr.core.results.parse_result.ParseResult
    options:
      members:
        - objective_funcs
        - objective_exprs
        - senses
        - constraints
        - variables
        - sorted_symbols
        - bound_manager
        - ir_problem
        - get_pyomo_problem_type
        - get_pymoo_problem_type
      show_root_heading: true
      show_root_toc_entry: false

## 功能配置

::: happymath.Opt.functional.config.ODEIVPConfig
    options:
      members: true
      show_root_heading: true
      show_root_toc_entry: false

::: happymath.Opt.functional.config.ODEBVPConfig
    options:
      members: true
      show_root_heading: true
      show_root_toc_entry: false

::: happymath.Opt.functional.config.DomainConfig
    options:
      members: true
      show_root_heading: true
      show_root_toc_entry: false

::: happymath.Opt.functional.config.ControlParamConfig
    options:
      members: true
      show_root_heading: true
      show_root_toc_entry: false

::: happymath.Opt.functional.config.MetricSpec
    options:
      members: true
      show_root_heading: true
      show_root_toc_entry: false
