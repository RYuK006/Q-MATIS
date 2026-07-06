import sys
import os

# Add root project dir to pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Apply platform patches immediately to prevent pytest worker DLL crashes
from superconductor.compat import apply_platform_patches
apply_platform_patches()
