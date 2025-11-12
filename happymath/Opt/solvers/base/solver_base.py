"""
BaseSolver - 基础求解器类

提取PyomoSolver和PymooSolver的公共逻辑，消除代码重复。
包含统一的求解器参数处理逻辑和求解流程控制。
"""

import time
import warnings
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional
from ...interfaces.solver import ISolver
from ...interfaces.problem_definition import IProblemDefinition


class BaseSolver(ISolver):
    """
    基础求解器类

    提供统一的求解器选择逻辑和多求解器管理，
    子类只需实现特定的求解器获取和单求解器求解逻辑。
    """

    def __init__(self, problem: IProblemDefinition):
        """
        初始化基础求解器

        Args:
            problem: 问题定义接口
        """
        self.problem = problem
        self._model_cache = None

    def solve(
        self,
        solver: Optional[Union[str, List[str]]] = None,
        use_auto_solvers: bool = True,
        max_solvers: Union[int, str] = 3
    ) -> List[Dict[str, Any]]:
        """统一的求解方法"""
        start_time = time.time()

        # 参数验证
        self._validate_solve_parameters(max_solvers)

        # 获取模型（由子类实现具体转换）
        model = self._get_or_create_model()

        # 解析求解器列表
        solvers = self._resolve_solver_list(solver, use_auto_solvers, max_solvers)

        # 求解
        results = []
        if len(solvers) == 1:
            # 单求解器求解
            result = self._solve_single(model, solvers[0])
            results.append(result)
        else:
            # 多求解器求解
            results = self._solve_multiple(model, solvers)

        # 记录总执行时间
        total_time = time.time() - start_time
        for result in results:
            if 'total_exec_time' not in result:
                result['total_exec_time'] = total_time

        return results

    def _validate_solve_parameters(self, max_solvers: Union[int, str]) -> None:
        """验证求解参数"""
        if max_solvers != "all":
            if not isinstance(max_solvers, int):
                raise ValueError("max_solvers参数必须是整数或'all'")
            if max_solvers < 1:
                raise ValueError("max_solvers参数不能小于1")

    def _resolve_solver_list(
        self,
        solver: Optional[Union[str, List[str]]],
        use_auto_solvers: bool,
        max_solvers: Union[int, str]
    ) -> List[str]:
        """统一的求解器列表解析逻辑"""
        if solver is None:
            # 用户未指定求解器，根据问题类型自动选择
            default_solvers = self._get_default_solvers(max_solvers)

            if use_auto_solvers:
                return default_solvers
            else:
                return [default_solvers[0]] if default_solvers else []

        elif isinstance(solver, str):
            if use_auto_solvers:
                # 确保指定的求解器在第一位，然后添加其他默认求解器
                all_solvers = self._get_default_solvers("all")
                remaining = [s for s in all_solvers if s != solver]

                solvers = [solver]
                if max_solvers == "all":
                    solvers.extend(remaining)
                else:
                    solvers.extend(remaining[:max_solvers-1])
                return solvers
            else:
                return [solver]

        elif isinstance(solver, list):
            if use_auto_solvers:
                # 用户指定了多个求解器
                if max_solvers == "all":
                    return solver
                else:
                    return solver[:max_solvers]
            else:
                return solver

        else:
            raise ValueError("solver参数必须是None、字符串或字符串列表")

    def _solve_multiple(self, model: Any, solvers: List[str]) -> List[Dict[str, Any]]:
        """
        多求解器求解逻辑
        """
        results = []

        for solver_name in solvers:
            try:
                result = self._solve_single(model, solver_name)
                results.append(result)

                # 如果找到成功的解，可以选择是否继续尝试其他求解器
                if result.get('success', False):
                    pass  # 继续尝试其他求解器以比较结果

            except Exception as e:
                # 单个求解器失败时记录错误但继续尝试其他求解器
                failed_result = {
                    'algorithm': solver_name,
                    'success': False,
                    'message': f"求解器 {solver_name} 失败: {str(e)}",
                    'solver_type': self.get_solver_type(),
                    'exec_time': 0.0
                }
                results.append(failed_result)

                warnings.warn(f"求解器 {solver_name} 执行失败: {str(e)}")

        return results

    # === 抽象方法 - 由子类实现 ===

    @abstractmethod
    def _get_default_solvers(self, max_solvers: Union[int, str]) -> List[str]:
        """
        获取默认求解器列表

        Args:
            max_solvers: 最大求解器数量

        Returns:
            默认求解器名称列表
        """
        pass

    @abstractmethod
    def _get_or_create_model(self) -> Any:
        """
        获取或创建模型

        Returns:
            特定框架的模型对象
        """
        pass

    @abstractmethod
    def _solve_single(self, model: Any, solver_name: str) -> Dict[str, Any]:
        """
        单求解器求解

        Args:
            model: 模型对象
            solver_name: 求解器名称

        Returns:
            求解结果字典
        """
        pass

    @abstractmethod
    def get_available_solvers(self) -> List[str]:
        """获取可用求解器列表"""
        pass

    @abstractmethod
    def get_solver_type(self) -> str:
        """获取求解器类型"""
        pass
