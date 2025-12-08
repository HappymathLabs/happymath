"""
HappyMath: A comprehensive mathematical computing and machine learning library.

HappyMath provides a unified interface for:
- Automated Machine Learning (AutoML)
- Multi-Criteria Decision Making (MCDM)
- Differential Equations (ODE/PDE)
- Mathematical Optimization

Author: HappyMathLabs
Email: tonghui_zou@happymath.com.cn
Homepage: https://github.com/HappyMathLabs/happymath
"""

import platform
from matplotlib.font_manager import FontManager
import subprocess
import warnings
import matplotlib.pyplot as plt

# Import version from dedicated version module
from ._version import __version__

# ============================================================================
# 重要：字体检测必须在子模块导入之前完成
# 否则子模块导入时 zh_font_paths 还未被设置（循环导入时序问题）
# ============================================================================

def available_ch_font():
    """
    判断系统中可用的中文字体。
    返回: (字体名称列表, 字体文件路径字典)
    """
    fm = FontManager()
    mat_fonts = {f.name: f.fname for f in fm.ttflist}  # 同时保存名称和路径
    available = {}

    # 尝试获取系统中文字体，区分不同操作系统
    system = platform.system()

    if system == "Linux":
        try:
            output = subprocess.check_output(
                'fc-list :lang=zh -f "%{family}\n"', shell=True, text=True)
            zh_fonts = set(f.split(',', 1)[0] for f in output.split('\n') if f.strip())
            for name in zh_fonts:
                if name in mat_fonts:
                    available[name] = mat_fonts[name]
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
    elif system == "Darwin":  # macOS
        # macOS系统中文字体检测
        zh_font_candidates = ['PingFang SC', 'Heiti SC', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        for name in zh_font_candidates:
            if name in mat_fonts:
                available[name] = mat_fonts[name]
    elif system == "Windows":
        # Windows系统中文字体检测
        zh_font_candidates = ['SimHei', 'Microsoft YaHei', 'Microsoft YaHei UI', 'KaiTi', 'FangSong']
        for name in zh_font_candidates:
            if name in mat_fonts:
                available[name] = mat_fonts[name]

    if not available:
        warnings.warn("There are no Chinese fonts available in the system, please download the relevant fonts.", UserWarning)
        return ["Arial"], {}
    else:
        names = sorted(list(available.keys()))
        return names, available

# 设置全局中文字体变量（必须在导入子模块之前完成）
_font_result = available_ch_font()
zh_font_available = _font_result[0] if isinstance(_font_result, tuple) else _font_result
zh_font_paths = _font_result[1] if isinstance(_font_result, tuple) else {}

# 在模块加载时立即设置 matplotlib 全局字体，确保后续所有绑定都使用中文字体
if zh_font_available and len(zh_font_available) > 0:
    plt.rcParams['font.sans-serif'] = [zh_font_available[0]]
    plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 导入子模块（此时 zh_font_available 和 zh_font_paths 已经设置完成）
# ============================================================================
from . import AutoML
from . import Decision
from . import DiffEq
from . import Opt

__all__ = [
    "AutoML",
    "Decision",
    "DiffEq",
    "Opt",
    "__version__",
    "available_ch_font",
    "zh_font_available",
    "zh_font_paths"
]