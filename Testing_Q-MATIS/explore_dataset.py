from matminer.datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd

print("Loading matbench_phonons via matminer...")
df = load_dataset("matbench_phonons")

# Perform the exact deterministic split we will use
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print(f"Total rows: {len(df)}")
print(f"Train rows: {len(train_df)}")
print(f"Test rows: {len(test_df)}")

print("\n--- Target (ph_freq) Distribution in TRAIN ---")
print(train_df['last phdos peak'].describe())

print("\n--- Target (ph_freq) Distribution in TEST ---")
print(test_df['last phdos peak'].describe())
