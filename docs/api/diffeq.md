# API 参考 - DiffEq

本页列出 `happymath.DiffEq` 模块中 ODE 与 PDE 求解的主要入口类和结果类。

## ODE 求解入口

::: happymath.DiffEq.ODE.ODEModule.ODEModule
    options:
      members:
        - __init__
        - ode2scipy
        - ana_solve
        - num_solve
        - stand_ode
        - show_stand_ode
      show_root_heading: true
      show_root_toc_entry: false

## ODE 结果容器

::: happymath.DiffEq.ODE.core.result.ODESolutionResult
    options:
      members:
        - domain
        - solution
        - error
        - solution_func
        - substitution_dict
        - success
        - message
      show_root_heading: true
      show_root_toc_entry: false

## PDE 求解入口

::: happymath.DiffEq.PDE.PDEModule.PDEModule
    options:
      members:
        - __init__
        - stand_pde
        - to_solvable_pde
        - num_solve
      show_root_heading: true
      show_root_toc_entry: false

## PDE 结果容器

::: happymath.DiffEq.PDE.core.result.PDESolutionResult
    options:
      members: true
      show_root_heading: true
      show_root_toc_entry: false

## PDE 求解适配器

::: happymath.DiffEq.PDE.adapters.pde_adapter.solve_pde
    options:
      show_root_heading: true
      show_root_toc_entry: false

## 微分方程基础类

::: happymath.DiffEq.diffeq_core.de_base.DEBase
    options:
      members:
        - __init__
        - is_ode
        - is_pde
        - core_symbol
      show_root_heading: true
      show_root_toc_entry: false
