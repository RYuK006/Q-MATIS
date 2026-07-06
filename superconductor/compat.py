import os
import logging
import platform

logger = logging.getLogger(__name__)

def apply_platform_patches():
    """
    Applies necessary platform-specific environment patches before heavy C++ libraries 
    (PyTorch, PyArrow, OpenMP) are imported.
    """
    if platform.system() == "Windows":
        # Prevents silent C++ access violations when PyTorch/OpenMP and NumPy/PyArrow
        # are loaded concurrently in the same process on Windows.
        if os.environ.get('KMP_DUPLICATE_LIB_OK') != 'True':
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
            logger.debug("Applied Windows compatibility patch: KMP_DUPLICATE_LIB_OK=True")
