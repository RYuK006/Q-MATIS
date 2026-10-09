import pandas as pd
from huggingface_hub import HfApi

df = pd.read_parquet("novel_candidates_results.parquet")
print("Local parquet columns:")
print(df.columns.tolist())

api = HfApi()
repo_id = "Aaron006/qmatis-stability-pilot"
print("\nFiles in HF repo:")
files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
print(files)

print("\nRecent commits:")
commits = api.list_repo_commits(repo_id=repo_id, repo_type="dataset")
for c in commits[:5]:
    print(f"Commit: {c.commit_id} - {c.title} (at {c.created_at})")
