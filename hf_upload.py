import os
from huggingface_hub import HfApi

api = HfApi()

repo_id = "Aaron006/qmatis-stability-pilot"
print(f"Creating repo {repo_id}...")
api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

print("Uploading HF_README.md...")
api.upload_file(
    path_or_fileobj="HF_README.md",
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="dataset"
)

print("Uploading novel_candidates_results.parquet...")
api.upload_file(
    path_or_fileobj="novel_candidates_results.parquet",
    path_in_repo="novel_candidates_results.parquet",
    repo_id=repo_id,
    repo_type="dataset"
)



print(f"Success! Dataset uploaded to https://huggingface.co/datasets/{repo_id}")
