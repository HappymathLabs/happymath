"""Optimization result container"""

from datetime import datetime
from typing import Any, Dict, List, Union, Optional
import numpy as np


class OptResult:
    """Unified wrapper for results from PyomoSolver and PymooSolver"""

    def __init__(self, results: Union[Dict, List[Dict]], opt_module_info: Dict):
        """
        初始化OptResult对象

        Args:
            results: 求解器返回的结果，可能是单个字典或字典列表
            opt_module_info: OptModule的基本信息字典
        """
        # 存储OptModule基本信息
        self._opt_module_info = opt_module_info
        self._libraries = opt_module_info.get('libraries', ['unknown'])
        self._mode = opt_module_info.get('mode', 'unknown')
        self._creation_time = datetime.now()

        # 处理求解结果
        if results is None:
            self._all_results = []
        elif isinstance(results, dict):
            self._all_results = [results]
        elif isinstance(results, list):
            self._all_results = results
        else:
            self._all_results = []

        # 找到最佳结果
        self._best_result = self._find_best_result()

        # 缓存机制
        self._all_solutions_cache = None
        self._all_solvers_cache = None
        self._pareto_front_cache = None

    # =================== 缓存的核心属性 ===================

    @property
    def all_solutions(self) -> List[Dict]:
        """
        返回所有求解结果 - 缓存版本

        Returns:
            List[Dict]: 所有求解结果的列表，每个解都包含variables和objective_value
        """
        # 使用缓存避免重复计算
        if self._all_solutions_cache is not None:
            return self._all_solutions_cache

        solutions = []

        for result in self._all_results:
            # 直接提取，而非修改状态
            variables = self._extract_variables(result)
            obj_value = self._extract_objective_value(result)

            # 只有当解成功且有有效数据时才添加
            if result.get('success', False) and (variables or obj_value is not None):
                solutions.append({
                    'variables': variables,
                    'objective_value': obj_value,
                    'algorithm': result.get('algorithm', 'unknown'),
                    'solver_type': result.get('solver_type', 'unknown'),
                    'exec_time': result.get('exec_time', 0.0)
                })
            elif variables or obj_value is not None:
                # 即使没有成功标志，但有有效数据也添加
                solutions.append({
                    'variables': variables,
                    'objective_value': obj_value,
                    'algorithm': result.get('algorithm', 'unknown'),
                    'solver_type': result.get('solver_type', 'unknown'),
                    'exec_time': result.get('exec_time', 0.0)
                })

        # 缓存结果
        self._all_solutions_cache = solutions
        return solutions

    @property
    def all_solvers(self) -> List[Dict]:
        """
        返回所有求解器信息 - 缓存版本

        Returns:
            List[Dict]: 所有求解器信息的列表
        """
        # 使用缓存避免重复计算
        if self._all_solvers_cache is not None:
            return self._all_solvers_cache

        solvers = []
        for result in self._all_results:
            solver_info = {
                'algorithm': result.get('algorithm', 'unknown'),
                'solver_type': result.get('solver_type', self._libraries),
                'exec_time': result.get('exec_time', 0.0),
                'success': result.get('success', False),
                'message': result.get('message', ''),
                'n_evals': result.get('n_evals', 0)
            }
            solvers.append(solver_info)

        # 缓存结果
        self._all_solvers_cache = solvers
        return solvers

    # =================== 变量和目标值提取方法 ===================

    def _extract_variables(self, result: Dict) -> Dict[str, float]:
        """
        从求解结果中提取变量值

        Args:
            result: 求解结果字典

        Returns:
            变量值字典
        """
        variables = {}

        # 直接从结果中获取variables字段
        if 'variables' in result and result['variables']:
            return result['variables']

        # 对于Pymoo结果，从X字段提取
        if result.get('solver_type') == 'pymoo' and 'X' in result:
            x_values = result['X']
            if x_values is not None:
                x_values = np.asarray(x_values)
                if x_values.ndim > 1:
                    x_values = x_values[0]  # 取第一个解

                # 使用符号映射
                sorted_symbols = self._opt_module_info.get('sorted_symbols')
                if not sorted_symbols:
                    # 回退到IR的变量顺序
                    ir_problem = self._opt_module_info.get('ir_problem')
                    if ir_problem is not None:
                        try:
                            sorted_symbols = [str(v.symbol) for v in ir_problem.variables]
                        except Exception:
                            sorted_symbols = []
                if sorted_symbols:
                    for i, sym_name in enumerate(sorted_symbols):
                        if i < len(x_values):
                            value = x_values[i]
                            if hasattr(value, 'item'):
                                value = value.item()
                            variables[str(sym_name)] = float(value)

        return variables

    def _extract_objective_value(self, result: Dict) -> Union[float, List[float], None]:
        """
        从求解结果中提取目标函数值

        Args:
            result: 求解结果字典

        Returns:
            单目标时返回float，多目标时返回List[float]，失败时返回None
        """
        # 直接从结果中获取objective_value字段
        if 'objective_value' in result:
            return result['objective_value']

        # 对于Pymoo结果，从F字段提取
        if result.get('solver_type') == 'pymoo' and 'F' in result:
            f_values = result['F']
            if f_values is not None:
                f_values = np.asarray(f_values)

                # 检测目标数（通过F的维度）
                if f_values.ndim > 1:
                    # 多解情况，取第一个解
                    f_values = f_values[0]  # shape: (n_obj,)

                # 判断是单目标还是多目标
                if f_values.ndim == 0:
                    # 标量，单目标
                    return float(f_values)
                elif len(f_values) == 1:
                    # 长度为1的数组，单目标
                    return float(f_values[0])
                else:
                    # 长度>1的数组，多目标
                    return [float(f) for f in f_values]

        return None

    # =================== 原有属性保持不变 ===================

    @property
    def solution(self) -> Optional[Dict]:
        """
        返回最佳的求解结果
        """
        if not self.success:
            return None

        variables = self.variables
        obj_value = self.objective_value

        if not variables and obj_value is None:
            return None

        return {
            'variables': variables,
            'objective_value': obj_value
        }

    @property
    def variables(self) -> Dict[str, float]:
        """
        返回最佳解的变量值
        """
        if self._best_result is None:
            return {}
        return self._extract_variables(self._best_result)

    @property
    def objective_value(self) -> Union[float, List[float], None]:
        """
        返回最佳解的目标函数值

        对于单目标问题返回float，多目标问题返回List[float]
        """
        if self._best_result is None:
            return None
        return self._extract_objective_value(self._best_result)

    @property
    def success(self) -> bool:
        """
        返回是否有成功的求解结果
        """
        return (self._best_result is not None and
                self._best_result.get('success', False))

    @property
    def raw_solution(self) -> Optional[Dict]:
        """
        返回原始的最佳求解结果
        """
        return self._best_result

    @property
    def raw_all_solutions(self) -> List[Dict]:
        """
        返回所有原始求解结果
        """
        return self._all_results.copy()

    @property
    def solver(self) -> Optional[Dict]:
        """
        返回最佳求解结果对应的solver信息字典
        """
        if self._best_result is None:
            return None

        return {
            'algorithm': self._best_result.get('algorithm', 'unknown'),
            'solver_type': self._best_result.get('solver_type', self._libraries),
            'exec_time': self._best_result.get('exec_time', 0.0),
            'success': self._best_result.get('success', False),
            'message': self._best_result.get('message', ''),
            'result': self._best_result.get('result', None)
        }

    @property
    def message(self) -> str:
        """
        返回求解消息（兼容旧OptResult接口）
        """
        solver_info = self.solver
        if solver_info:
            return solver_info.get('message', '') or ''
        if self._best_result:
            return self._best_result.get('message', '') or ''
        return ''

    @property
    def solver_name(self) -> str:
        """
        返回求解器名称（兼容旧接口）
        """
        solver_info = self.solver
        if solver_info:
            solver_type = solver_info.get('solver_type')
            if isinstance(solver_type, (list, tuple)) and solver_type:
                return solver_type[0]
            return solver_type or 'unknown'
        return 'unknown'

    # =================== 其他方法 ===================

    def _find_best_result(self) -> Optional[Dict]:
        """从所有结果中选出“最佳”结果：
        - 单目标：根据 min/max 方向选最优 objective_value（若有），否则退化为最短时间
        - 多目标：目前退化为最短时间（待扩展：帕累托占优比较）
        """
        if not self._all_results:
            return None

        successful = [r for r in self._all_results if r.get('success', False)]
        if not successful:
            return self._all_results[0]

        # 读取目标方向
        senses = self._opt_module_info.get('senses', [])
        n_obj = len(senses) if senses else 0

        # 单目标择优
        if n_obj == 1:
            direction = senses[0]

            # 仅在 objective_value 存在时按数值比较
            candidates = [r for r in successful if r.get('objective_value') is not None]
            if candidates:
                if direction == 'min':
                    return min(candidates, key=lambda r: float(r.get('objective_value')))
                else:
                    return max(candidates, key=lambda r: float(r.get('objective_value')))

            # 否则退化为最短时间
            return min(successful, key=lambda r: r.get('exec_time', float('inf')))

        # 多目标：基于F前沿的质量择优（若有），否则退化为最短时间
        import numpy as _np
        best = None
        best_metric = None
        for r in successful:
            F = r.get('F', None)
            if F is None:
                continue
            try:
                F_arr = _np.asarray(F)
                if F_arr.ndim == 1:
                    score = float(_np.sum(F_arr))
                else:
                    # 取前沿中 sum(F) 的最小值作为质量指标（越小越好）
                    score = float(_np.min(_np.sum(F_arr, axis=1)))
            except Exception:
                continue
            if (best_metric is None) or (score < best_metric):
                best_metric = score
                best = r

        if best is not None:
            return best
        return min(successful, key=lambda r: r.get('exec_time', float('inf')))

    def clear_cache(self):
        """
        清除所有缓存

        当结果数据可能发生变化时调用此方法
        """
        self._all_solutions_cache = None
        self._all_solvers_cache = None
        self._pareto_front_cache = None

    # =================== 多目标优化支持 ===================

    @property
    def pareto_front(self) -> List[Dict]:
        """
        返回帕累托前沿解（多目标优化）

        Returns:
            List[Dict]: 帕累托前沿解列表
        """
        if self._pareto_front_cache is not None:
            return self._pareto_front_cache

        # 提取多目标结果
        multi_objective_solutions = []
        for result in self._all_results:
            if result.get('solver_type') != 'pymoo':
                continue
            post_meta = result.get('postprocess') or {}
            front_source = post_meta.get('raw_front', result.get('F'))
            if front_source is None:
                continue
            f_values = np.asarray(front_source)
            if f_values.ndim <= 1:
                continue
            x_source = post_meta.get('raw_decisions', result.get('X'))
            x_values = None
            if x_source is not None:
                try:
                    x_values = np.asarray(x_source)
                except Exception:
                    x_values = None
            for i, f_val in enumerate(f_values):
                x_val = None
                if x_values is not None and i < x_values.shape[0]:
                    x_val = x_values[i]
                multi_objective_solutions.append({
                    'objectives': f_val.tolist(),
                    'variables_array': x_val.tolist() if x_val is not None else None,
                    'algorithm': result.get('algorithm', 'unknown')
                })

        # 缓存结果
        self._pareto_front_cache = multi_objective_solutions
        return multi_objective_solutions

    def __str__(self) -> str:
        """
        返回结果的字符串表示
        """
        if self.success:
            return f"OptResult(success=True, algorithm={self.solver.get('algorithm', 'unknown')}, " \
                   f"objective_value={self.objective_value})"
        else:
            return f"OptResult(success=False, {len(self._all_results)} results)"

    def __repr__(self) -> str:
        return self.__str__()
