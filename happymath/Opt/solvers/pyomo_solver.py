"""
PyomoSolver - 重构版本

继承BaseSolver基类，消除代码重复，专注于Pyomo特定的求解逻辑。
"""

import warnings
import pyomo.environ as pyo
import numpy as np
import time
from typing import List, Dict, Any, Union

from .base.solver_base import BaseSolver
from ..interfaces.problem_definition import IProblemDefinition
from ..adapters.pyomo_adapter import PyomoAdapter
from ..adapters.pyomo_dae_adapter import PyomoDAEAdapter


class PyomoSolver(BaseSolver):
    """
    Pyomo求解器类

    继承BaseSolver，处理所有与Pyomo相关的求解逻辑。
    专注于Pyomo特定的功能，公共逻辑由基类处理。
    """

    def __init__(self, problem: IProblemDefinition, epsilon: float = 1e-6):
        """
        初始化PyomoSolver

        Args:
            problem: 问题定义接口
            epsilon: epsilon值，用于约束处理
        """
        super().__init__(problem)
        self.epsilon = epsilon

    def get_solver_type(self) -> str:
        """获取求解器类型"""
        return 'pyomo'

    def _get_default_solvers(self, max_solvers: Union[int, str]) -> List[str]:
        """
        根据问题类型获取默认求解器列表

        Args:
            max_solvers: 最大求解器数量

        Returns:
            求解器名称列表
        """
        problem_type = self.problem.get_pyomo_problem_type()

        # 根据问题类型选择求解器
        solver_mapping = {
            'LP': ['cbc', 'glpk'],       # Linear Programming
            'QP': ['scip','ipopt'],     # Quadratic Programming
            'NP': ['scip','ipopt'],     # Nonlinear Programming (NLP)
            'MILP': ['scip', 'cbc'],     # Mixed-Integer Linear Programming
            'MIQP': ['scip', 'mindtpy'], # Mixed-Integer Quadratic Programming
            'MINP': ['scip', 'mindtpy']  # Mixed-Integer Nonlinear Programming (MINLP)
        }

        if problem_type not in solver_mapping:
            raise ValueError(f"Unsupported problem type: {problem_type}")

        # 基础候选
        solvers = list(solver_mapping[problem_type])

        # 针对凸QP尝试加入 OSQP（仅连续变量）
        if problem_type == 'QP':
            convex_qp = bool(getattr(self.problem, 'is_convex_qp', False))
            if convex_qp:
                try:
                    s = pyo.SolverFactory('osqp')
                    if s is not None and s.available():
                        solvers.insert(0, 'osqp')
                except Exception:
                    pass
            else:
                warnings.warn("Detected non-convex or degenerate QP: skipping OSQP and using generic solvers.")

        # 过滤不可用求解器，提升成功率与日志整洁
        available = []
        for name in solvers:
            try:
                s = pyo.SolverFactory(name)
                if s is not None and s.available():
                    available.append(name)
            except Exception:
                continue

        if not available:
            # 若全不可用，返回原始列表（由上层捕获错误并提示安装）
            candidates = solvers
        else:
            candidates = available

        if max_solvers == "all":
            return candidates
        else:
            return candidates[:max_solvers]

    def _get_or_create_model(self) -> pyo.ConcreteModel:
        """
        获取或创建Pyomo模型

        Returns:
            Pyomo ConcreteModel实例
        """
        if self._model_cache is None:
            # 检查是否存在功能型（FUNCTIONAL）目标/约束，且提供 ODE/IVP 配置
            pr = getattr(self, 'problem', None)
            functional_cfg = getattr(pr, '_functional_config', None)
            has_functional = False
            try:
                for obj in pr.ir_problem.objectives:
                    if getattr(obj, 'is_functional', False):
                        has_functional = True
                        break
                if not has_functional:
                    for con in pr.ir_problem.constraints:
                        if str(getattr(con, 'category', '')) == 'functional':
                            has_functional = True
                            break
            except Exception:
                has_functional = False

            if has_functional and functional_cfg is not None:
                # 若检测到“变量指数”形式（Pow 的指数含变量），提示改用 Pymoo
                from sympy import Pow
                def _has_variable_exponent(expr) -> bool:
                    try:
                        for pow_node in expr.atoms(Pow):
                            expn = pow_node.args[1]
                            if hasattr(expn, 'free_symbols') and expn.free_symbols:
                                return True
                    except Exception:
                        return False
                    return False
                var_exp = False
                try:
                    metas = []
                    metas.append((functional_cfg.objective_meta or {}).get(0, {}))
                    for _, v in (functional_cfg.constraint_meta or {}).items():
                        metas.append(v)
                    for meta in metas:
                        expr = (meta or {}).get('expr')
                        if expr is not None and _has_variable_exponent(expr):
                            var_exp = True
                            break
                except Exception:
                    var_exp = False
                if var_exp:
                    raise ValueError("Detected variable-in-exponent form (e.g., x**alpha). Pyomo.DAE does not support this nonlinearity; please use Pymoo mode.")
                # 走 DAE 适配器（仅 ODE/IVP）
                dae = PyomoDAEAdapter(pr, functional_cfg, epsilon=self.epsilon)
                self._model_cache = dae.convert()
            else:
                adapter = PyomoAdapter(self.problem, self.epsilon)
                self._model_cache = adapter.convert()
        return self._model_cache

    def _solve_single(self, model: pyo.ConcreteModel, solver_name: str) -> Dict[str, Any]:
        """
        使用单个求解器求解

        Args:
            model: Pyomo模型
            solver_name: 求解器名称

        Returns:
            求解结果字典
        """
        start_time = time.time()

        try:
            # 验证模型
            self._validate_model(model)

            # 创建求解器
            solver_pyomo = self._create_solver(solver_name)

            # 求解
            results = solver_pyomo.solve(model, tee=False)

            # 记录执行时间
            exec_time = time.time() - start_time

            # 处理结果
            return self._process_solve_results(
                model, results, solver_name, exec_time
            )

        except Exception as e:
            exec_time = time.time() - start_time
            return self._create_failed_result(solver_name, str(e), exec_time)

    def _validate_model(self, model: pyo.ConcreteModel) -> None:
        """
        验证模型的有效性

        Args:
            model: Pyomo模型

        Raises:
            ValueError: 如果模型无效
        """
        if model is None:
            raise ValueError(
                "Pyomo模型为None，无法求解。\n"
                "Possible reasons: expression parsing failed or model conversion failed"
            )

        # 验证变量数量
        n_vars = sum(1 for _ in model.component_objects(pyo.Var, active=True))
        if n_vars == 0:
            raise ValueError(
                "Pyomo模型中没有变量，无法求解。\n"
                "Please ensure the objective and constraints involve variables"
            )

    def _create_solver(self, solver_name: str) -> pyo.SolverFactory:
        """
        创建Pyomo求解器

        Args:
            solver_name: 求解器名称

        Returns:
            Pyomo求解器实例

        Raises:
            RuntimeError: 如果求解器创建失败或不可用
        """
        try:
            solver_pyomo = pyo.SolverFactory(solver_name)
        except Exception as e:
            raise RuntimeError(
                f"Unable to create solver '{solver_name}'。\n"
                f"Error details: {str(e)}\n"
                f"Please ensure the solver is installed"
            ) from e

        if not solver_pyomo.available():
            raise RuntimeError(
                f"Solver '{solver_name}' 不可用。\n"
                f"可能原因：\n"
                f"1. 求解器未安装\n"
                f"2. 求解器可执行文件不在PATH中\n"
                f"3. 许可证问题"
            )

        if solver_name.lower() == 'mindtpy':
            self._configure_mindtpy_defaults(solver_pyomo)

        # 保守：不强制设置预处理选项，避免不同求解器的参数差异导致失败

        return solver_pyomo

    def _process_solve_results(
        self,
        model: pyo.ConcreteModel,
        results: Any,
        solver_name: str,
        exec_time: float
    ) -> Dict[str, Any]:
        """
        处理求解结果

        Args:
            model: Pyomo模型
            results: Pyomo求解结果
            solver_name: 求解器名称
            exec_time: 执行时间

        Returns:
            标准化的结果字典
        """
        # 检查求解状态
        solve_status = results.solver.status
        termination_condition = results.solver.termination_condition

        success = (solve_status == pyo.SolverStatus.ok and
                  termination_condition == pyo.TerminationCondition.optimal)

        # 提取变量值
        variables = {}
        if success:
            for var in model.component_objects(pyo.Var, active=True):
                for index in var:
                    var_name = str(var[index])
                    variables[var_name] = pyo.value(var[index])

        # 提取目标函数值
        objective_value = None
        if success and model.component_objects(pyo.Objective, active=True):
            obj = next(model.component_objects(pyo.Objective, active=True))
            objective_value = pyo.value(obj)

        # 构造结果字典
        result_info = {
            'algorithm': solver_name,
            'result': results,
            'success': success,
            'message': f"Solver {solver_name} finished" if success else f"Solver {solver_name} failed",
            'exec_time': exec_time,
            'solver_type': 'pyomo',
            'variables': variables,
            'objective_value': objective_value,
            'solve_status': str(solve_status),
            'termination_condition': str(termination_condition)
        }

        return result_info

    def _create_failed_result(
        self,
        solver_name: str,
        error_message: str,
        exec_time: float
    ) -> Dict[str, Any]:
        """
        创建failed result dict

        Args:
            solver_name: 求解器名称
            error_message: 错误消息
            exec_time: 执行时间

        Returns:
            failed结果字典
        """
        return {
            'algorithm': solver_name,
            'result': None,
            'success': False,
            'message': f"Solver {solver_name} failed: {error_message}",
            'exec_time': exec_time,
            'solver_type': 'pyomo',
            'variables': {},
            'objective_value': None
        }

    def _configure_mindtpy_defaults(self, solver_pyomo) -> None:
        """为 MindtPy 注入稳健的默认配置。"""
        config = getattr(solver_pyomo, 'config', None)
        if config is None:
            config = getattr(solver_pyomo, 'CONFIG', None)
        if config is None:
            return
        nlp_solver = self._pick_available_solver(['ipopt', 'cyipopt'])
        mip_solver = self._pick_available_solver(['cbc', 'glpk', 'scip'])
        self._set_config_if_absent(config, 'nlp_solver', nlp_solver)
        self._set_config_if_absent(config, 'mip_solver', mip_solver)
        self._set_config_if_absent(config, 'strategy', 'OA')
        self._set_config_if_absent(config, 'time_limit', 300)
        self._set_config_if_absent(config, 'integer_tolerance', 1e-6)
        self._set_config_if_absent(config, 'feasibility_norm', 'Linf')

    def _pick_available_solver(self, candidates):
        """按顺序挑选可用求解器名称。"""
        for name in candidates:
            try:
                solver = pyo.SolverFactory(name)
                if solver is not None and solver.available():
                    return name
            except Exception:
                continue
        return candidates[0]

    @staticmethod
    def _set_config_if_absent(config, key, value):
        """仅在用户未手动设置时配置 MindtPy 选项。"""
        current = getattr(config, key, None)
        if current in (None, '', 0):
            setattr(config, key, value)

    def get_available_solvers(self) -> List[str]:
        """
        获取可用求解器列表

        Returns:
            可用求解器名称列表
        """
        # 常见的Pyomo求解器
        common_solvers = [
            'cbc', 'glpk', 'ipopt', 'scip', 'gurobi', 'cplex',
            'mindtpy', 'baron', 'couenne', 'bonmin'
        ]

        available = []
        for solver_name in common_solvers:
            try:
                solver = pyo.SolverFactory(solver_name)
                if solver.available():
                    available.append(solver_name)
            except:
                pass  # 忽略不可用的求解器

        return available
