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

# Import main modules
from . import AutoML
from . import Decision
from . import DiffEq
from . import Opt

def available_ch_font():
    """判断系统中可用的中文字体"""
    fm = FontManager()
    mat_fonts = set(f.name for f in fm.ttflist)
    available = set()
    
    # 尝试获取系统中文字体，区分不同操作系统
    system = platform.system()
    
    if system == "Linux":
        try:
            output = subprocess.check_output(
                'fc-list :lang=zh -f "%{family}\n"', shell=True, text=True)
            zh_fonts = set(f.split(',', 1)[0] for f in output.split('\n') if f.strip())
            available = mat_fonts & zh_fonts
        except (subprocess.SubprocessError, FileNotFoundError):
            available = set()
    elif system == "Darwin":  # macOS
        # macOS系统中文字体检测
        zh_font_candidates = ['PingFang SC', 'Heiti SC', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        available = mat_fonts & set(zh_font_candidates)
    elif system == "Windows":
        # Windows系统中文字体检测
        zh_font_candidates = ['SimHei', 'Microsoft YaHei', 'Microsoft YaHei UI', 'KaiTi', 'FangSong']
        available = mat_fonts & set(zh_font_candidates)
    
    if not available:
        warnings.warn("There are no Chinese fonts available in the system, please download the relevant fonts.", UserWarning)
        available = ["Arial"]
    else:
        available = sorted(list(available))
        
    return available

# 设置全局中文字体变量
zh_font_available = available_ch_font()

__all__ = [
    "AutoML",
    "Decision", 
    "DiffEq",
    "Opt",
    "__version__",
    "available_ch_font",
    "zh_font_available"
]