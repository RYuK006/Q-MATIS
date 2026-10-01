import time
import os
import torch
import pandas as pd
from pymatgen.core import Structure
from chgnet.model.model import CHGNet
from chgnet.model.dynamics import StructOptimizer

# Use Hugging Face datasets to get WBM
from datasets import load_dataset
print("Loading WBM test data...")
try:
    dataset = load_dataset("janosh/wbm", split="test")
    df = dataset.to_pandas()
    print("Loaded from janosh/wbm")
except Exception as e:
    print(f"HF failed: {e}. Trying direct download...")
    # fallback to matbench_discovery or direct url if HF fails
    url = "https://figshare.com/ndownloader/files/40594376" # typically WBM summary
    import urllib.request
    urllib.request.urlretrieve(url, "wbm.csv")
    df = pd.read_csv("wbm.csv")

print(f"Total WBM size: {len(df)}")
# sample 1000
df_sample = df.sample(n=1000, random_state=42)

# check columns
print("Columns:", df_sample.columns.tolist())
