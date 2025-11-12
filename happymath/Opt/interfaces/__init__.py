"""Module exports."""
Opt库的接口抽象层

这个module定义了核心组件的接口，支持依赖注入和面向接口的编程。
通过这些接口，我们可以：
- 提高代码的可测试性
- 支持插件式扩展
- 减少组件间的耦合
- 符合SOLID原则
"""

from .problem_definition import IProblemDefinition
from .solver import ISolver, ISolverFactory
from .model_adapter import IModelAdapter

__all__ = [
    'IProblemDefinition',
    'ISolver',
    'ISolverFactory',
    'IModelAdapter'
]