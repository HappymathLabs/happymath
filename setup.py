"""Setup script for HappyMath package."""

from setuptools import setup, find_packages
import os

# Read the README file
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "Happymath is a high-level mathematical modeling Python library. Its core philosophy lies in reducing users' learning costs through high-level encapsulation, enabling efficient mathematical modeling. It is particularly suitable for mathematical modeling competitions and applied mathematics fields."

# Package metadata
NAME = "happymath"
VERSION = "0.1.5"
DESCRIPTION = "Happymath is a high-level mathematical modeling Python library. Its core philosophy lies in reducing users' learning costs through high-level encapsulation, enabling efficient mathematical modeling. It is particularly suitable for mathematical modeling competitions and applied mathematics fields."
AUTHOR = "HappymathLabs"
AUTHOR_EMAIL = "tonghui_zou@happymath.com.cn"
URL = "https://github.com/HappymathLabs/happymath"

# Dependencies
INSTALL_REQUIRES = [
    # Core scientific computing
    "numpy",
    "scipy==1.10.0",
    "sympy",
    "pandas<2.2.0",
    
    # Machine learning core
    "scikit-learn",
    "catboost", 
    "xgboost",
    "pycaret==3.3.2",
    
    # Optimization solvers (Python interfaces)
    "pyomo",
    "pymoo",
    "cyipopt",        # IPOPT interface  
    
    # Visualization and UI
    "matplotlib<3.8.0",
    "seaborn",
    "dash",
    "dash-bootstrap-components",
    "panel",
    "jupyter_bokeh",
    
    # Statistics and analysis
    "statsmodels",
    "patsy",
    "shap",
    "umap-learn",
    
    # Additional tools
    "timeout-decorator",
    "py-pde",
    "ipython",
    "ipykernel",
    "watchfiles"
]

# Optional dependencies
EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=6.0",
        "pytest-cov",
        "black",
        "flake8",
        "mypy"
    ],
    "docs": [
        "sphinx",
        "sphinx-rtd-theme",
        "sphinx-autodoc-typehints"
    ],
    "all": [
        # Additional machine learning libraries
        "lightgbm",
        
        # Additional optimization solvers
        "cylp",           # COIN-OR CBC interface  
        "swiglpk",        # GLPK interface
        "PySCIPOpt",      # SCIP interface
    ]
}

# Classifiers
CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules"
]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    project_urls={
        "Homepage": URL,
        "Repository": "https://github.com/HappyMathLabs/happymath",
        "Documentation": "https://github.com/HappyMathLabs/happymath",
        "Bug Tracker": "https://github.com/HappyMathLabs/happymath/issues",
    },
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.11",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    classifiers=CLASSIFIERS,
    keywords="mathematics machine-learning optimization differential-equations decision-analysis",
    license="MIT",
    zip_safe=False,
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.rst", "*.md"],
    },
)
