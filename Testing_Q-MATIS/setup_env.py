import os
import subprocess
import sys

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    # 1. Create venv
    venv_dir = ".venv_exp2"
    if not os.path.exists(venv_dir):
        run_cmd(f"uv venv --python 3.12 {venv_dir}")
        
    # Python executable
    python_exe = os.path.join(venv_dir, "Scripts", "python.exe")
    
    # 2. Install torch
    run_cmd(f"uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --python {python_exe}")
    
    # 3. Get torch version to install matching PyG
    output = subprocess.check_output([python_exe, "-c", "import torch; print(torch.__version__)"], text=True).strip()
    torch_version = output.split("+")[0] # e.g., '2.5.1'
    pyg_url = f"https://data.pyg.org/whl/torch-{torch_version}+cu124.html"
    
    # 4. Install PyG and extensions
    run_cmd(f"uv pip install torch-geometric --python {python_exe}")
    run_cmd(f"uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f {pyg_url} --python {python_exe}")
    
    # 5. Install other packages
    run_cmd(f"uv pip install pymatgen scipy pandas matbench --python {python_exe}")
    
    # 6. Verify CUDA and write environment.md
    verify_script = """
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

env_md = f\"\"\"# Environment Log
- **Platform**: {platform.platform()}
- **Python**: {platform.python_version()}
- **PyTorch**: {torch_ver}
- **PyG**: {pyg_ver}
- **CUDA Available**: {cuda_available}
- **GPU Model**: {device_name}
- **Pipeline Stage GPU/CPU Breakdown**:
  - **Data Parsing (JSON -> PyMatgen)**: CPU (Multiprocessed)
  - **Graph Building (Distance / Radius Graph)**: CPU (PyMatgen neighbor lists are CPU bound, parallelized via multiprocessing)
  - **DataLoader Batching**: CPU (with `pin_memory=True`)
  - **Model Training (Forward/Backward/Loss)**: GPU (`.to('cuda')`)
\"\"\"

os.makedirs("results", exist_ok=True)
with open("results/environment.md", "w") as f:
    f.write(env_md)
print("Environment verified and logged.")
"""
    
    with open("verify_cuda.py", "w") as f:
        f.write(verify_script)
        
    run_cmd([python_exe, "verify_cuda.py"])

if __name__ == "__main__":
    main()
