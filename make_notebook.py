import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cell1 = nbf.v4.new_markdown_cell("# BEE-NET Superconductor Screening\nThis notebook is configured to run BEE-NET on Google Colab (T4 GPU).")
cell2 = nbf.v4.new_code_cell("!pip install -q huggingface_hub pymatgen pandas torch")

cell3 = nbf.v4.new_code_cell("""import torch
import sys

# 1. Detect GPU and abort if missing
if not torch.cuda.is_available():
    print("ERROR: No GPU detected! Go to Runtime > Change runtime type and select T4 GPU.")
    sys.exit(1)
    
gpu_name = torch.cuda.get_device_name(0)
print(f"GPU Detected: {gpu_name}")
""")

cell4 = nbf.v4.new_code_cell("""import os
from google.colab import drive
from huggingface_hub import snapshot_download

# 2. Mount Drive
drive.mount('/content/drive')
drive_dir = '/content/drive/MyDrive/BEE_NET_Project'
weights_dir = os.path.join(drive_dir, 'bee_net_weights')
os.makedirs(drive_dir, exist_ok=True)

# Replace with the exact Hugging Face repo ID for BEE-NET
HF_REPO_ID = "YOUR_HF_REPO_HERE" 

print("Checking Drive for model weights...")
if not os.path.exists(weights_dir) or len(os.listdir(weights_dir)) == 0:
    print("Downloading weights from Hugging Face (~12GB) to Google Drive...")
    snapshot_download(repo_id=HF_REPO_ID, local_dir=weights_dir, local_dir_use_symlinks=False)
    print("Download complete.")
else:
    print("Weights found on Drive. Skipping download.")
""")

cell5 = nbf.v4.new_code_cell("""import pandas as pd
import time
import os
import torch

results_file = os.path.join(drive_dir, 'bee_net_results.csv')

# Load the candidates (Replace with your actual 43 candidates or the test 2)
# Here we define the 2-structure test:
candidates = [
    {"material_id": "mp-aaabfqpo", "formula": "Sr2Co2O5"},
    {"material_id": "mp-aaacpwsj", "formula": "Nb8PtSe20"}
]

# Check existing completed candidates
if os.path.exists(results_file):
    completed_df = pd.read_csv(results_file)
    completed_ids = set(completed_df['material_id'].tolist())
    print(f"Found {len(completed_ids)} completed candidates on Drive.")
else:
    completed_ids = set()

# Initialize BEE-NET model here...
# model = load_bee_net_model(weights_dir)
# model.to('cuda')

total_candidates = len(candidates)
start_time = time.time()
processed = 0

print(f"Starting inference for {total_candidates} candidates...")

for i, cand in enumerate(candidates, 1):
    mat_id = cand['material_id']
    formula = cand['formula']
    
    if mat_id in completed_ids:
        print(f"[{i}/{total_candidates}] {formula} already done. Skipping.")
        continue
        
    t0 = time.time()
    
    # ----------------------------------------------------
    # RUN BEE-NET PREDICTION HERE
    # Example:
    # struct = get_structure_for_id(mat_id)
    # result = model.predict(struct)
    # predicted_tc = result['Tc']
    # ----------------------------------------------------
    
    # Simulating inference...
    time.sleep(5) 
    predicted_tc = 0.0 # Placeholder
    
    elapsed = time.time() - t0
    processed += 1
    
    # Write immediately to Drive CSV
    new_row = pd.DataFrame([{"material_id": mat_id, "formula": formula, "predicted_tc": predicted_tc}])
    new_row.to_csv(results_file, mode='a', header=not os.path.exists(results_file), index=False)
    
    # Progress Reporting
    total_elapsed = time.time() - start_time
    avg_time = total_elapsed / processed
    rem_time = avg_time * (total_candidates - i)
    
    mem_alloc = torch.cuda.memory_allocated(0) / 1024**3
    mem_res = torch.cuda.memory_reserved(0) / 1024**3
    
    print(f"[{i}/{total_candidates}] {formula} done | elapsed: {int(elapsed//60)}m {int(elapsed%60)}s "
          f"| avg: {avg_time:.1f}s/candidate | est. remaining: {int(rem_time//60)}m {int(rem_time%60)}s "
          f"| GPU Mem: {mem_alloc:.1f}GB alloc, {mem_res:.1f}GB res")

print("Run complete!")
""")

nb.cells = [cell1, cell2, cell3, cell4, cell5]
with open('bee_net_colab.ipynb', 'w') as f:
    nbf.write(nb, f)
