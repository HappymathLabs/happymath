"""
Pymoo solver - refactored version.

Inherits BaseSolver and uses dedicated modules to simplify structure.
"""

import time
import warnings
from typing import List, Dict, Any, Union, Optional

import numpy as np
from pymoo.optimize import minimize

from .base.solver_base import BaseSolver
from ..interfaces.problem_definition import IProblemDefinition
from ..adapters.pymoo_adapter import PymooAdapter
from .pymoo.algorithm_factory import PymooAlgorithmFactory


class PymooSolver(BaseSolver):
    """
    Pymoo solver class (refactored).

    Uses separate modules to handle responsibilities, e.g. algorithm selection/creation.
    """

    def __init__(self, problem: IProblemDefinition, epsilon: float = 1e-6):
        """
        Initialize the PymooSolver.

        Args:
            problem: Problem definition instance.
            epsilon: Epsilon for constraint handling.
        """
        super().__init__(problem)
        self.epsilon = epsilon

        # Initialize specialized modules
        self.algorithm_factory = PymooAlgorithmFactory()
        # 不再从 IR context 读取 Pymoo 配置；使用内置默认
        self._verbosity = True
        self._budget_override: Optional[int] = None

    def get_solver_type(self) -> str:
        """Return solver type."""
        return 'pymoo'

    def _get_default_solvers(self, max_solvers: Union[int, str]) -> List[str]:
        """
        Get default algorithm names based on problem type.

        Args:
            max_solvers: Maximum number of algorithms.

        Returns:
            List of algorithm names.
        """
        problem_type_dict = self.problem.get_pymoo_problem_type()
        return self.algorithm_factory.get_recommended_algorithms(
            problem_type_dict, max_solvers
        )

    def _get_or_create_model(self) -> Any:
        """
        Get or create the Pymoo problem instance.

        Returns:
            Pymoo problem instance.
        """
        if self._model_cache is None:
            adapter = PymooAdapter(self.problem, self.epsilon)
            self._model_cache = adapter.convert()
            # Set evaluation budget hint if provided
            try:
                if self._budget_override is not None:
                    setattr(self._model_cache, '_budget_hint', int(self._budget_override))
            except Exception:
                pass

        return self._model_cache

    def _solve_single(self, problem: Any, algorithm_name: str) -> Dict[str, Any]:
        """
        Solve with a single algorithm.

        Args:
            problem: Pymoo problem instance.
            algorithm_name: Algorithm name.

        Returns:
            Result dictionary.
        """
        start_time = time.time()

        try:
            # 根据问题特性过滤算法
            problem_type_dict = self.problem.get_pymoo_problem_type()
            ok, reason = self.algorithm_factory.is_algorithm_compatible(algorithm_name, problem_type_dict)
            if not ok:
                raise ValueError(f"Algorithm {algorithm_name} is incompatible with problem settings: {reason}")

            algorithm = self.algorithm_factory.create_algorithm(algorithm_name, problem)
            if algorithm is None:
                raise ValueError(f"Unable to create algorithm: {algorithm_name}")

            # Use the algorithm directly for solving (no auto-tuning)
            n_evals = self._calculate_evaluation_budget(problem)
            final_result = minimize(
                problem,
                algorithm,
                termination=("n_evals", n_evals),
                seed=1,
                verbose=False
            )

            # Result dict consistent with previous path, for extraction
            result_info = {
                'algorithm': algorithm_name,
                'result': final_result,
                'success': True,
                'message': f"Algorithm {algorithm_name} solved successfully",
                'exec_time': time.time() - start_time,
                'solver_type': 'pymoo',
                'X': getattr(final_result, 'X', None),
                'F': getattr(final_result, 'F', None),
                'G': getattr(final_result, 'G', None),
                'CV': getattr(final_result, 'CV', None),
                'n_evals': getattr(final_result, 'evaluator', {}).get('n_eval', n_evals)
            }

            # Extract solution info
            return self._extract_solution_info(result_info, algorithm_name)

        except Exception as e:
            exec_time = time.time() - start_time
            return self._create_failed_result(algorithm_name, str(e), exec_time)

    def _calculate_evaluation_budget(self, problem: Any) -> int:
        """Compute evaluation budget used directly for solving (no auto-tuning)."""
        # Prefer budget hint on the problem object
        try:
            hint = getattr(problem, '_budget_hint', None)
            if hint is not None:
                return int(hint)
        except Exception:
            pass

        # Else compute based on problem size similar to adapter logic
        n_constr = 0
        try:
            n_constr = int(getattr(problem, 'n_ieq_constr', 0) + getattr(problem, 'n_eq_constr', 0))
        except Exception:
            n_constr = 0

        if problem.n_obj == 1:
            return min(1000, max(200, 200 + 50 * int(problem.n_var) + 20 * n_constr))
        else:
            return min(3000, max(400, 400 + 80 * int(problem.n_var) + 40 * n_constr))

    def _extract_solution_info(
        self,
        result_info: Dict[str, Any],
        algorithm_name: str
    ) -> Dict[str, Any]:
        """
        Extract solution information from the solver result.

        Args:
            result_info: Result dictionary.
            algorithm_name: Algorithm name.

        Returns:
            Standardized result dictionary.
        """
        # 如果直接提供了 X/F 字段（不含 result 对象），仍需执行最小成功性校验
        if 'X' in result_info and 'F' in result_info and 'result' not in result_info:
            x_val = result_info.get('X')
            success_flag = result_info.get('success', True)
            if x_val is None or (hasattr(x_val, 'size') and x_val.size == 0):
                success_flag = False
                result_info['message'] = result_info.get('message') or f"Algorithm {algorithm_name} did not find a feasible solution"
            result_info['success'] = success_flag
            return result_info

        # 否则从 result 对象中提取信息
        final_result = result_info.get('result')
        if final_result is None:
            return result_info

        # 提取变量和目标值
        variables = self._extract_variables(final_result)
        objective_value = self._extract_objective_value(final_result)

        # 更新结果信息
        result_info.update({
            'variables': variables,
            'objective_value': objective_value,
            'X': getattr(final_result, 'X', None),
            'F': getattr(final_result, 'F', None),
            'G': getattr(final_result, 'G', None),
            'CV': getattr(final_result, 'CV', None)
        })

        # 成功性判定：若缺少可行解，则标记失败
        x_val = result_info.get('X')
        success_flag = result_info.get('success', True)
        if x_val is None:
            success_flag = False
        else:
            x_arr = np.asarray(x_val)
            if x_arr.size == 0 or np.any(np.isnan(x_arr)):
                success_flag = False

        # 方案3：将成功判据与软等式阈值τ联动
        if success_flag and result_info.get('CV') is not None:
            cv_arr = np.asarray(result_info['CV'])
            if cv_arr.size > 0:
                # 默认阈值
                cv_threshold = 1e-6
                try:
                    # 若问题存在等式，则允许用当前τ放宽（取 max(τ, 1e-6)）
                    has_eq = getattr(self._model_cache, 'n_eq_constr', 0) > 0
                    if has_eq:
                        soft_state = getattr(self._model_cache, '_soft_eq_state', None)
                        if isinstance(soft_state, dict):
                            tau = float(soft_state.get('current', cv_threshold))
                            cv_threshold = max(cv_threshold, tau)
                except Exception:
                    pass
                if np.any(cv_arr > cv_threshold):
                    success_flag = False

        if not success_flag:
            result_info['success'] = False
            result_info['message'] = result_info.get('message') or f"算法 {algorithm_name} 未找到可行解"
        else:
            result_info['success'] = True
        self._attach_search_box_diagnostics(result_info)

        return result_info

    def _extract_variables(self, result: Any) -> Dict[str, float]:
        """
        从Pymoo结果中提取变量值

        Args:
            result: Pymoo求解结果

        Returns:
            变量值字典
        """
        variables = {}

        if hasattr(result, 'X') and result.X is not None:
            x_values = np.asarray(result.X)
            if x_values.ndim > 1:
                x_values = x_values[0]  # 取第一个解

            # 如果问题模型提供了枚举/整数类型映射，则进行解码与四舍五入
            model = getattr(self, '_model_cache', None)
            var_types = getattr(model, '_var_types', None) if model is not None else None
            enum_maps = getattr(model, '_enum_maps', None) if model is not None else None

            decoded = np.array(x_values, copy=True, dtype=float)
            if var_types is not None:
                for idx, vtype in enumerate(var_types):
                    try:
                        if vtype.name.lower() == 'enum' and enum_maps and idx in enum_maps:
                            mapping = enum_maps[idx]
                            choice = int(np.clip(np.round(decoded[idx]), 0, len(mapping)-1))
                            decoded[idx] = mapping[choice]
                        elif vtype.name.lower() == 'binary':
                            decoded[idx] = float(np.clip(np.round(decoded[idx]), 0, 1))
                        elif vtype.name.lower() == 'integer':
                            decoded[idx] = float(np.round(decoded[idx]))
                    except Exception:
                        # 容错：保持原值
                        pass

            sorted_symbols = self.problem.sorted_symbols
            for i, symbol in enumerate(sorted_symbols):
                if i < len(decoded):
                    value = decoded[i]
                    if hasattr(value, 'item'):
                        value = value.item()
                    variables[str(symbol)] = float(value)

        return variables

    def _extract_objective_value(self, result: Any) -> Union[float, List[float], None]:
        """
        从Pymoo结果中提取目标函数值

        Args:
            result: Pymoo求解结果

        Returns:
            单目标时返回float，多目标时返回List[float]，失败时返回None
        """
        if hasattr(result, 'F') and result.F is not None:
            f_values = np.asarray(result.F)

            # 检查问题的目标数
            n_obj = self._model_cache.n_obj if self._model_cache else 1

            # 处理多解情况：取第一个解
            if f_values.ndim > 1:
                f_values = f_values[0]  # shape: (n_obj,)

            # 根据目标数返回适当的格式
            if n_obj == 1:
                # 单目标：返回float
                if f_values.ndim == 0:
                    return float(f_values)
                else:
                    return float(f_values[0]) if len(f_values) > 0 else float(f_values)
            else:
                # 多目标：返回List[float]
                if f_values.ndim == 0:
                    return [float(f_values)]
                else:
                    return [float(f) for f in f_values]
        return None

    def _create_failed_result(
        self,
        algorithm_name: str,
        error_message: str,
        exec_time: float
    ) -> Dict[str, Any]:
        """
        创建失败的结果字典

        Args:
            algorithm_name: 算法名称
            error_message: 错误消息
            exec_time: 执行时间

        Returns:
            失败结果字典
        """
        return {
            'algorithm': algorithm_name,
            'result': None,
            'success': False,
            'message': f"算法 {algorithm_name} 失败: {error_message}",
            'exec_time': exec_time,
            'solver_type': 'pymoo',
            'variables': {},
            'objective_value': None,
            'n_evals': 0
        }

    def _attach_search_box_diagnostics(self, result_info: Dict[str, Any]) -> None:
        """将搜索边界注释写入结果，帮助用户理解默认信赖域。"""
        model = getattr(self, '_model_cache', None)
        if model is None:
            return
        annotations = getattr(model, '_search_box_annotations', None)
        if not annotations:
            return
        diagnostics = result_info.setdefault('diagnostics', {})
        diagnostics['search_box_overrides'] = list(annotations)
        search_range = getattr(model, '_search_bound_range', None)
        if search_range is not None:
            diagnostics['search_box_range'] = float(search_range)

    def get_available_solvers(self) -> List[str]:
        """
        获取可用算法列表

        Returns:
            可用算法名称列表
        """
        return self.algorithm_factory.get_available_algorithms()
