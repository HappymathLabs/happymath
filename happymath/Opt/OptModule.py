"""
OptModule - Refactored version.

Uses a cleaned OptBase and new solver interfaces; removes legacy compatibility layers.
Includes comprehensive input validation and error handling.
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

        # Strict bound policy: Pymoo only allowed when all variables are bounded
        import warnings
        all_bounded = self._check_all_variables_bounded()

        if self.is_single_obj:
            if mode == "auto":
                if all_bounded:
                    self.libraries = ["pyomo", "pymoo"]
                else:
                    warnings.warn(
                        "Detected variables without bounds: disabling Pymoo and falling back to Pyomo backend."
                    )
                    self.libraries = ["pyomo"]
            elif mode == "pyomo":
                self.libraries = ["pyomo"]
            elif mode == "pymoo":
                if not all_bounded:
                    # Collect names of unbounded variables
                    try:
                        ub_vars = self._collect_unbounded_variables()
                        details = ", ".join(ub_vars) if ub_vars else "(unknown variable)"
                    except Exception:
                        details = "(failed to collect variable list)"
                    raise ValueError(
                        "Pymoo strict mode: variables without bounds detected; heuristic backend cannot be used."
                        f" Please provide explicit lower/upper bounds for variables or switch to Pyomo. Unbounded: {details}"
                    )
                self.libraries = ["pymoo"]
            else:
                raise ValueError(f"Invalid mode for single-objective problem: {mode}")
        else:  # Multi-objective
            if mode == "pyomo":
                raise ValueError("Pyomo does not support multi-objective optimization.")
            # Multi-objective currently supports only Pymoo; under strict policy, error if any variable is unbounded
            if not all_bounded:
                try:
                    ub_vars = self._collect_unbounded_variables()
                    details = ", ".join(ub_vars) if ub_vars else "(unknown variable)"
                except Exception:
                    details = "(failed to collect variable list)"
                raise ValueError(
                    "Multi-objective optimization requires Pymoo, but unbounded variables were detected."
                    f" Please provide explicit bounds before solving. Unbounded: {details}"
                )
            self.libraries = ["pymoo"]

        # 初始化求解器，使用新的接口
        self.pyomo_solver = PyomoSolver(self.parse_result, epsilon=self.epsilon)
        self.pymoo_solver = PymooSolver(self.parse_result, epsilon=self.epsilon)

    def _check_all_variables_bounded(self):
        """Return True if all variables have both lower and upper bounds."""
        return self.parse_result.bound_manager.check_all_variables_bounded()

    def _prepare_opt_module_info(self):
        """
        Prepare OptModule metadata for OptResult.

        Returns:
            dict: Basic information about the problem and solvers.
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
            # Keep symbol objects to facilitate decoding/mapping later
            'sorted_symbols': self.parse_result.sorted_symbols,
        }

    def _collect_unbounded_variables(self):
        """Return a list of variable names without proper bounds."""
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
        Solve the optimization problem.

        Args:
            solver: Name of solver/algorithm. None to auto-select by problem type.
            use_auto_solvers: If True, try multiple solvers/algorithms; otherwise use a single solver.
            max_solvers: Maximum number of solvers/algorithms to try.
            ref: Reference-point dict for Pymoo post-processing; None to use ASF without prior ideal point.

        Returns:
            OptResult with solver outputs.

        Raises:
            TypeError: Invalid parameter types.
            ValueError: Invalid parameter values.
        """
        # 验证求解器参数
        try:
            solver, use_auto_solvers, max_solvers = validate_solver_params(
                solver, use_auto_solvers, max_solvers
            )
        except (TypeError, ValueError) as e:
            raise type(e)(
                f"Solver parameter validation failed: {str(e)}\n\n"
                f"Examples:\n"
                f"  result = opt.solve()  # auto select\n"
                f"  result = opt.solve('cbc')  # choose a solver\n"
                f"  result = opt.solve(['cbc', 'glpk'])  # choose multiple solvers"
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
                'message': f"Solve failed: {str(e)}",
                'exec_time': solve_time,
                'solver_type': self.libraries,
            }

            # 创建包含失败信息的OptResult对象
            opt_module_info = self._prepare_opt_module_info()
            return OptResult([failed_result], opt_module_info)
