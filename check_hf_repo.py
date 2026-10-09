import requests
import subprocess
from huggingface_hub import snapshot_download

print("1. Checking snapshot_download with repo_type='dataset'")
try:
    snapshot_download(repo_id="paprakash/BEE-NET", repo_type="dataset", local_dir="BEE-NET-models-test")
    print("snapshot_download (dataset) succeeded!")
except Exception as e:
    print(f"snapshot_download (dataset) failed: {e}")

print("\n----------------------------------")
print("2. Checking huggingface-cli download")
result = subprocess.run(["huggingface-cli", "download", "paprakash/BEE-NET", "--local-dir", "BEE-NET-models"], capture_output=True, text=True)
if result.returncode == 0:
    print("huggingface-cli succeeded!")
    print(result.stdout)
else:
    print(f"huggingface-cli failed with return code {result.returncode}")
    print(result.stderr)

print("\n----------------------------------")
print("3. Checking Hugging Face API directly (models)")
url_model = "https://huggingface.co/api/models/paprakash/BEE-NET"
res_model = requests.get(url_model)
print(f"Status: {res_model.status_code}")
print(f"Body: {res_model.text}")

print("\n----------------------------------")
print("4. Checking Hugging Face API directly (datasets)")
url_dataset = "https://huggingface.co/api/datasets/paprakash/BEE-NET"
res_dataset = requests.get(url_dataset)
print(f"Status: {res_dataset.status_code}")
print(f"Body: {res_dataset.text}")
