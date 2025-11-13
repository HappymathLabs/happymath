"""Version management for happymath package."""

import importlib.metadata as metadata
import importlib.util as util
from pathlib import Path


def get_version():
    """Get version number using multiple fallback strategies."""
    
    # Method 1: Try to get from installed package metadata (most reliable)
    try:
        return metadata.version("happymath")
    except metadata.PackageNotFoundError:
        pass
    except Exception:
        pass
    
    # Method 2: Development mode - read directly from pyproject.toml
    try:
        # Find pyproject.toml relative to this file
        current_dir = Path(__file__).parent
        pyproject_path = current_dir.parent / "pyproject.toml"
        
        if pyproject_path.exists():
            import re
            with open(pyproject_path, 'r', encoding='utf-8') as f:
                content = f.read()
                version_match = re.search(r'version\s*=\s*"([^"]*)"', content)
                if version_match:
                    return version_match.group(1)
    except Exception:
        pass
    
    # Method 3: Try to import from setup.py as fallback
    try:
        setup_path = Path(__file__).parent.parent / "setup.py"
        if setup_path.exists():
            import re
            with open(setup_path, 'r', encoding='utf-8') as f:
                content = f.read()
                version_match = re.search(r'VERSION\s*=\s*"([^"]*)"', content)
                if version_match:
                    return version_match.group(1)
    except Exception:
        pass
    
    # Final fallback
    return "unknown"


# Version string - this is what users will import
__version__ = get_version()