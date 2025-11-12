"""
Pymoo算法工厂

负责创建和管理Pymoo算法实例，从PymooSolver中提取出来的专门模块。
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple

# 导入常用算法
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.algorithms.soo.nonconvex.sres import SRES
from pymoo.algorithms.soo.nonconvex.isres import ISRES
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.soo.nonconvex.g3pcx import G3PCX

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.dnsga2 import DNSGA2


class PymooAlgorithmFactory:
    """
    Pymoo算法工厂类

    负责根据问题类型推荐算法，并创建算法实例。
    统一管理所有Pymoo算法的创建逻辑。
    """

    _ALGO_CAPABILITIES: Dict[str, Dict[str, Any]] = {
        # 单目标
        "GA": {"objective": "single", "supports_constraints": True},
        "DE": {"objective": "single", "supports_constraints": True},
        "BRKGA": {"objective": "single", "supports_constraints": True},
        "NELDERMEAD": {"objective": "single", "supports_constraints": False},
        "PATTERNSEARCH": {"objective": "single", "supports_constraints": False},
        "CMAES": {"objective": "single", "supports_constraints": False},
        "ES": {"objective": "single", "supports_constraints": False},
        "SRES": {"objective": "single", "supports_constraints": True},
        "ISRES": {"objective": "single", "supports_constraints": True},
        "PSO": {"objective": "single", "supports_constraints": False},
        "G3PCX": {"objective": "single", "supports_constraints": False},
        # 多/多目标
        "NSGA2": {"objective": "multi", "supports_constraints": True},
        "NSGA3": {"objective": "many", "supports_constraints": True},
        "UNSGA3": {"objective": "many", "supports_constraints": True},
        "MOEAD": {"objective": "multi", "supports_constraints": False},
        "RVEA": {"objective": "many", "supports_constraints": False},
        "AGEMOEA": {"objective": "multi", "supports_constraints": False},
        "AGEMOEA2": {"objective": "many", "supports_constraints": False},
        "CTAEA": {"objective": "many", "supports_constraints": True},
        "SMSEMOA": {"objective": "multi", "supports_constraints": True},
        "DNSGA2": {"objective": "multi", "supports_constraints": True},
    }

    def __init__(self):
        """初始化算法工厂（移除动态参考点状态）"""
        self._last_rejected: List[Tuple[str, str]] = []

    def get_recommended_algorithms(
        self,
        problem_type_dict: Dict[str, Any],
        max_algorithms: Union[int, str] = 3
    ) -> List[str]:
        """
        根据问题类型获取推荐算法列表

        Args:
            problem_type_dict: 问题类型字典，包含：
                - 'objective_type': 'single', 'multi', 或 'many'
                - 'has_constraints': True/False
                - 'n_objectives': 目标函数数量
                - 'n_constraints': 约束数量
                - 'has_discrete_vars': 是否有离散变量
            max_algorithms: 控制返回的算法数量

        Returns:
            推荐的算法名称列表，按优先级排序
        """
        objective_type = problem_type_dict['objective_type']
        has_constraints = problem_type_dict['has_constraints']
        has_discrete_vars = problem_type_dict['has_discrete_vars']
        n_objectives = problem_type_dict['n_objectives']

        if objective_type == 'single':
            # 单目标优化算法
            if has_constraints:
                if has_discrete_vars:
                    # 离散变量 + 约束：进化算法最适合
                    algorithms = ['GA', 'BRKGA', 'DE', 'ISRES', 'SRES', 'NelderMead', 'PatternSearch']
                else:
                    # 连续变量 + 约束：DE约束处理能力强，ISRES专门为约束优化设计
                    algorithms = ['DE', 'ISRES', 'SRES', 'GA', 'NelderMead', 'PatternSearch', 'BRKGA']
            else:
                if has_discrete_vars:
                    # 离散变量 + 无约束：GA首选，BRKGA组合优化特别有效
                    algorithms = ['GA', 'BRKGA', 'DE', 'PSO', 'ES', 'G3PCX', 'CMAES']
                else:
                    # 连续变量 + 无约束：CMAES是连续优化的金标准，DE全局优化能力强
                    algorithms = ['CMAES', 'DE', 'PSO', 'ES', 'GA', 'G3PCX', 'BRKGA']

        elif objective_type == 'multi':
            # 多目标优化算法（2-3个目标）
            if has_constraints:
                if n_objectives == 2:
                    # 双目标 + 约束：NSGA2经典且稳定，RNSGA2适合有参考点的情况
                    algorithms = ['NSGA2', 'NSGA3', 'UNSGA3', 'CTAEA', 'SMSEMOA']
                else:  # 3个目标
                    # 三目标 + 约束：NSGA3专门为多目标设计，表现优秀
                    algorithms = ['NSGA3', 'UNSGA3', 'CTAEA', 'NSGA2', 'SMSEMOA']
            else:
                if n_objectives == 2:
                    # 双目标 + 无约束：NSGA2被广泛验证，MOEAD分解方法有时更好
                    algorithms = ['NSGA2', 'MOEAD', 'AGEMOEA', 'RVEA', 'AGEMOEA2']
                else:  # 3个目标
                    # 三目标 + 无约束：NSGA3和MOEAD都很好，NSGA3稍优
                    algorithms = ['NSGA3', 'MOEAD', 'RVEA', 'AGEMOEA', 'AGEMOEA2', 'NSGA2']

        else:  # objective_type == 'many'
            # 高维多目标优化算法（4+个目标）
            if has_constraints:
                if n_objectives <= 6:
                    # 中等维度（4-6个目标）+ 约束：NSGA3和CTAEA都很好
                    algorithms = ['NSGA3', 'CTAEA', 'RVEA', 'UNSGA3', 'MOEAD']
                else:
                    # 高维度（7+个目标）+ 约束：RVEA专门为高维设计
                    algorithms = ['RVEA', 'MOEAD', 'NSGA3', 'CTAEA']
            else:
                if n_objectives <= 6:
                    # 中等维度（4-6个目标）+ 无约束：NSGA3和RVEA都很好
                    algorithms = ['NSGA3', 'RVEA', 'MOEAD', 'AGEMOEA2', 'CTAEA']
                else:
                    # 高维度（7+个目标）+ 无约束：RVEA首选，MOEAD分解方法也不错
                    algorithms = ['RVEA', 'MOEAD', 'NSGA3', 'AGEMOEA2']

        # 根据max_algorithms参数返回算法列表
        compatible, rejected = self.filter_algorithms(algorithms, problem_type_dict)
        self._last_rejected = rejected
        if max_algorithms == "all":
            return compatible
        else:
            return compatible[:max_algorithms]

    def create_algorithm(self, algo_name: str, problem) -> Optional[Any]:
        """
        根据算法名称创建对应的pymoo算法实例

        Args:
            algo_name: 算法名称
            problem: pymoo问题实例

        Returns:
            算法实例或None
        """
        name = algo_name.upper() if isinstance(algo_name, str) else algo_name
        try:
            # 单目标算法
            if name == "GA":
                # 调整默认种群规模以匹配较小评估预算（提高代数）
                return GA(pop_size=20)
            elif name == "DE":
                return DE()
            elif name == "BRKGA":
                return BRKGA()
            elif name == "NELDERMEAD":
                return NelderMead()
            elif name == "PATTERNSEARCH":
                return PatternSearch()
            elif name == "CMAES":
                return CMAES()
            elif name == "ES":
                return ES()
            elif name == "SRES":
                return SRES()
            elif name == "ISRES":
                return ISRES()
            elif name == "PSO":
                return PSO()
            elif name == "G3PCX":
                return G3PCX()

            # 多目标算法
            elif name == "NSGA2":
                return NSGA2(pop_size=100)  # 增加种群大小以提高求解质量
            elif name == "NSGA3":
                # 需要参考方向
                from pymoo.util.ref_dirs import get_reference_directions
                ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=12)
                return NSGA3(ref_dirs=ref_dirs, pop_size=100)
            elif name == "UNSGA3":
                from pymoo.util.ref_dirs import get_reference_directions
                ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=12)
                return UNSGA3(ref_dirs=ref_dirs)
            elif name == "MOEAD":
                from pymoo.util.ref_dirs import get_reference_directions
                ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=12)
                return MOEAD(ref_dirs, n_neighbors=20)
            elif name == "RVEA":
                from pymoo.util.ref_dirs import get_reference_directions
                ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=12)
                return RVEA(ref_dirs)
            elif name == "AGEMOEA":
                return AGEMOEA()
            elif name == "AGEMOEA2":
                return AGEMOEA2()
            elif name == "CTAEA":
                from pymoo.util.ref_dirs import get_reference_directions
                ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=12)
                return CTAEA(ref_dirs)
            elif name == "SMSEMOA":
                return SMSEMOA()
            elif name == "DNSGA2":
                return DNSGA2()

            else:
                print(f"不支持的算法: {algo_name}")
                return None

        except Exception as e:
            print(f"创建算法 {algo_name} 时出错: {str(e)}")
            return None

    def get_available_algorithms(self) -> List[str]:
        """
        获取所有可用算法列表

        Returns:
            算法名称列表
        """
        single_objective = [
            'GA', 'DE', 'BRKGA', 'NelderMead', 'PatternSearch',
            'CMAES', 'ES', 'SRES', 'ISRES', 'PSO', 'G3PCX'
        ]

        multi_objective = [
            'NSGA2', 'NSGA3', 'UNSGA3',
            'MOEAD', 'RVEA', 'AGEMOEA', 'AGEMOEA2', 'CTAEA',
            'SMSEMOA', 'DNSGA2'
        ]

        return single_objective + multi_objective

    def is_algorithm_compatible(
        self, algo_name: str, problem_type_dict: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """判断算法是否与问题特性兼容"""
        name = algo_name.upper()
        meta = self._ALGO_CAPABILITIES.get(name, {"objective": "any", "supports_constraints": True})

        objective_type = problem_type_dict.get('objective_type', 'single')
        problem_has_constraints = bool(problem_type_dict.get('has_constraints'))

        algo_obj_type = meta.get('objective', 'any')
        if objective_type == 'single' and algo_obj_type in {'multi', 'many'}:
            return False, "仅支持多目标问题"
        if objective_type != 'single' and algo_obj_type == 'single':
            return False, "仅支持单目标问题"
        if problem_has_constraints and not meta.get('supports_constraints', True):
            return False, "该算法不支持约束处理"
        return True, None

    def filter_algorithms(
        self,
        algorithms: List[str],
        problem_type_dict: Dict[str, Any]
    ) -> Tuple[List[str], List[Tuple[str, str]]]:
        """根据问题特性过滤兼容的算法"""
        compatible: List[str] = []
        rejected: List[Tuple[str, str]] = []
        for algo in algorithms:
            ok, reason = self.is_algorithm_compatible(algo, problem_type_dict)
            if ok:
                compatible.append(algo)
            else:
                rejected.append((algo, reason or "不兼容"))
        return compatible, rejected
