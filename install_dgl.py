import subprocess
import sys
import torch
import logging

logger = logging.getLogger(__name__)

def install_dgl():
    try:
        import dgl
        logger.info(f"DGL already installed: version {dgl.__version__}")
        return True
    except ImportError:
        logger.info("DGL not found. Attempting automatic installation...")
        
    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda
    
    logger.info(f"CUDA Available: {cuda_available}")
    if cuda_available:
        logger.info(f"CUDA Version: {cuda_version}")
        
    # We will use uv pip to install
    # DGL installation commands vary. For PyTorch 2.x on Windows/Linux:
    # CPU: uv pip install dgl -f https://data.dgl.ai/wheels/repo.html
    # CUDA 11.8: uv pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
    # CUDA 12.1: uv pip install dgl -f https://data.dgl.ai/wheels/cu121/repo.html
    
    base_url = "https://data.dgl.ai/wheels/repo.html"
    if cuda_available and cuda_version:
        # e.g., '11.8' -> 'cu118', '12.1' -> 'cu121'
        clean_cuda = cuda_version.replace('.', '')
        base_url = f"https://data.dgl.ai/wheels/cu{clean_cuda}/repo.html"
        
    cmd = [sys.executable, "-m", "uv", "pip", "install", "dgl", "-f", base_url]
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        # Fallback to standard pip if uv is not available
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning("uv pip failed, falling back to standard pip...")
            cmd = [sys.executable, "-m", "pip", "install", "dgl", "-f", base_url]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
        if result.returncode == 0:
            logger.info("DGL installed successfully!")
            return True
        else:
            logger.error(f"Failed to install DGL: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Exception during DGL installation: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    install_dgl()
