import torch
import torch_geometric
import platform
import os

cuda_available = torch.cuda.is_available()
if not cuda_available:
    raise RuntimeError("CUDA is NOT available. Halting as per instructions.")

device_name = torch.cuda.get_device_name(0)
torch_ver = torch.__version__
pyg_ver = torch_geometric.__version__

env_md = f"""# Environment Log
- **Platform**: {platform.platform()}
- **Python**: {platform.python_version()}
- **PyTorch**: {torch_ver}
- **PyG**: {pyg_ver}
- **CUDA Available**: {cuda_available}
- **GPU Model**: {device_name}
- **Pipeline Stage GPU/CPU Breakdown**:
  - **Data Parsing (JSON -> PyMatgen)**: CPU (Multiprocessed)
  - **Graph Building (Distance / Radius Graph)**: CPU (PyMatgen neighbor lists are CPU bound, parallelized via multiprocessing, while tensor prep is vectorized)
  - **DataLoader Batching**: CPU (with `pin_memory=True`)
  - **Model Training (Forward/Backward/Loss)**: GPU (`.to('cuda')`)
"""

os.makedirs("results", exist_ok=True)
with open("results/environment.md", "w") as f:
    f.write(env_md)
print("Environment verified and logged.")
