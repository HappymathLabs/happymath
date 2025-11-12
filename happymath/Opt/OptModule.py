"""
OptModule - 重构版本

使用清理后的OptBase和新的求解器接口，删除了所有向后兼容依赖。
添加了全面的输入验证和错误处理。
"""

import time
import numpy as np
from .opt_core.opt_base import OptBase
from .solvers.pyomo_solver import PyomoSolver
from .solvers.pymoo_solver import PymooSolver
from .results.opt_result import OptResult
from .results.preference_selector import select_preferred_from_pymoo
from .validation import validate_inputs, validate_solver_params


class OptModule(OptBase):
    @validate_inputs
    def __init__(self, obj_func, constraints=None, mode="auto", default_search_range=100, show_bound_warnings=True, tighten_bounds=None, **kwargs):
        super().__init__(
            obj_func,
            constraints,
            default_search_range=default_search_range,
            show_bound_warnings=show_bound_warnings,
            tighten_bounds=tighten_bounds,
            **kwargs
        )

        self.mode = mode

        # 使用新的接口获取目标函数数量
        self.is_single_obj = len(self.parse_result.objective_funcs) <= 1

        # 严格边界策略：Pymoo 仅在所有变量有上下界时允许
        import warnings
        all_bounded = self._check_all_variables_bounded()

        if self.is_single_obj:
            if mode == "auto":
                if all_bounded:
                    self.libraries = ["pyomo", "pymoo"]
                else:
                    warnings.warn(
                        "检测到存在无上下界的变量：已禁用 Pymoo，自动切换到 Pyomo 后端。"
                    )
                    self.libraries = ["pyomo"]
            elif mode == "pyomo":
                self.libraries = ["pyomo"]
            elif mode == "pymoo":
                if not all_bounded:
                    # 收集无界变量名
                    try:
                        ub_vars = self._collect_unbounded_variables()
                        details = ", ".join(ub_vars) if ub_vars else "(未知变量)"
                    except Exception:
                        details = "(变量列表收集失败)"
                    raise ValueError(
                        "Pymoo 严格模式：检测到无上下界的变量，无法使用启发式后端。"
                        f" 请为变量显式提供上下界或改用 Pyomo。未界定变量：{details}"
                    )
                self.libraries = ["pymoo"]
            else:
                raise ValueError(f"Invalid mode for single-objective problem: {mode}")
        else:  # Multi-objective
            if mode == "pyomo":
                raise ValueError("Pyomo does not support multi-objective optimization.")
            # 多目标目前仅支持 Pymoo；严格策略下，若变量无界则直接报错
            if not all_bounded:
                try:
                    ub_vars = self._collect_unbounded_variables()
                    details = ", ".join(ub_vars) if ub_vars else "(未知变量)"
                except Exception:
                    details = "(变量列表收集失败)"
                raise ValueError(
                    "多目标优化需要 Pymoo，但检测到存在无上下界的变量。"
                    f" 请为变量显式提供上下界后再求解。未界定变量：{details}"
                )
            self.libraries = ["pymoo"]

        # 初始化求解器，使用新的接口
        self.pyomo_solver = PyomoSolver(self.parse_result, epsilon=self.epsilon)
        self.pymoo_solver = PymooSolver(self.parse_result, epsilon=self.epsilon)

    def _check_all_variables_bounded(self):
        """检查是否所有变量都有边界"""
        return self.parse_result.bound_manager.check_all_variables_bounded()

    def _prepare_opt_module_info(self):
        """
        准备传递给OptResult的OptModule信息

        Returns:
            dict: OptModule的基本信息字典
        """
        problem_type = "Unknown"
        if "pyomo" in self.libraries:
            problem_type = self.pyomo_problem_type
        elif "pymoo" in self.libraries:
            problem_type = self.pymoo_problem_type

        return {
            'libraries': self.libraries,
            'mode': self.mode,
            'obj_func': self.parse_result.objective_exprs,
            'senses': self.parse_result.senses,
            'problem_type': problem_type,
            'ir_problem': self.parse_result.ir_problem,
            # 保持符号对象形式，便于后续变量解码与映射
            'sorted_symbols': self.parse_result.sorted_symbols,
        }

    def _collect_unbounded_variables(self):
        """返回未设置上下界的变量名列表。"""
        bm = self.parse_result.bound_manager
        xl = bm.lower_bounds
        xu = bm.upper_bounds
        names = []
        
        for i, sym in enumerate(self.parse_result.sorted_symbols):
            if not (np.isfinite(xl[i]) and np.isfinite(xu[i])):
                names.append(str(sym))
        return names

    def solve(self, solver: str = None, use_auto_solvers: bool = True, max_solvers: int = 3, ref: dict | None = None):
        """
        求解优化问题

        Args:
            solver: 指定求解器/算法名称，如果为None则根据问题类型自动选择
            use_auto_solvers: 是否使用多个求解器/算法求解，True时使用多个求解器，False时只使用单个求解器
            max_solvers: 最大使用的求解器/算法数量，默认3个
            ref: Pymoo结果后处理使用的参考点字典；None表示采用无先验理想点ASF

        Returns:
            OptResult: 封装求解结果的OptResult对象

        Raises:
            TypeError: 如果参数类型不正确
            ValueError: 如果参数值不正确
        """
        # 验证求解器参数
        try:
            solver, use_auto_solvers, max_solvers = validate_solver_params(
                solver, use_auto_solvers, max_solvers
            )
        except (TypeError, ValueError) as e:
            raise type(e)(
                f"求解参数验证失败: {str(e)}\n\n"
                f"使用示例:\n"
                f"  result = opt.solve()  # 自动选择\n"
                f"  result = opt.solve('cbc')  # 指定求解器\n"
                f"  result = opt.solve(['cbc', 'glpk'])  # 指定多个求解器"
            ) from e

        # 记录开始时间
        start_time = time.time()

        pymoo_available = {name.upper() for name in self.pymoo_solver.get_available_solvers()}
        pyomo_solver_arg = solver
        pymoo_solver_arg = solver

        if isinstance(solver, str):
            if solver.upper() in pymoo_available:
                pyomo_solver_arg = None
                pymoo_solver_arg = solver
            else:
                pyomo_solver_arg = solver
                pymoo_solver_arg = None
        elif isinstance(solver, list):
            pyomo_list = []
            pymoo_list = []
            for item in solver:
                if isinstance(item, str) and item.upper() in pymoo_available:
                    pymoo_list.append(item)
                else:
                    pyomo_list.append(item)
            pyomo_solver_arg = pyomo_list if pyomo_list else None
            pymoo_solver_arg = pymoo_list if pymoo_list else None
        else:
            pyomo_solver_arg = solver
            pymoo_solver_arg = solver

        try:
            results = []
            run_pyomo = "pyomo" in self.libraries
            run_pymoo = "pymoo" in self.libraries

            if self.mode == "pyomo":
                run_pymoo = False
            elif self.mode == "pymoo":
                run_pyomo = False

            if isinstance(solver, str):
                if solver.upper() in pymoo_available:
                    run_pyomo = False
                else:
                    run_pymoo = False
            elif isinstance(solver, list):
                if not pyomo_solver_arg:
                    run_pyomo = False
                if not pymoo_solver_arg:
                    run_pymoo = False

            if run_pyomo:
                pyomo_results = self.pyomo_solver.solve(pyomo_solver_arg, use_auto_solvers and run_pymoo, max_solvers)
                results.extend(pyomo_results)

            if run_pymoo:
                pymoo_results = self.pymoo_solver.solve(pymoo_solver_arg, use_auto_solvers, max_solvers)
                results.extend(pymoo_results)

            if not results:
                raise ValueError(f"No valid library to solve the problem.")

            if "pymoo" in self.libraries:
                results = select_preferred_from_pymoo(
                    results=results,
                    senses=self.parse_result.senses,
                    objective_exprs=self.parse_result.objective_exprs,
                    ir_problem=self.parse_result.ir_problem,
                    sorted_symbols=self.parse_result.sorted_symbols,
                    ref=ref,
                )

            # 求解成功，创建并返回OptResult对象
            opt_module_info = self._prepare_opt_module_info()
            return OptResult(results, opt_module_info)

        except Exception as e:
            # 求解失败时创建失败的结果
            solve_time = time.time() - start_time
            failed_result = {
                'algorithm': solver if solver else "auto",
                'result': None,
                'success': False,
                'message': f"求解失败: {str(e)}",
                'exec_time': solve_time,
                'solver_type': self.libraries,
            }

            # 创建包含失败信息的OptResult对象
            opt_module_info = self._prepare_opt_module_info()
            return OptResult([failed_result], opt_module_info)
