import os
import urllib.request
import hashlib
import sys

MODEL_URL = "https://huggingface.co/Aaron006/Discovery/resolve/main/rf_tc_model.joblib"
MODEL_PATH = "models/rf_tc_model.joblib"
EXPECTED_HASH = "96F7A0CFA391C729B3F0BF22CF2F127F864E9D9DC35566A78C566E39DC3FF5AB"

def download_if_missing():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Downloading...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        size = os.path.getsize(MODEL_PATH)
        print(f"Downloaded model. Size: {size} bytes")
    
    # Hash check
    m = hashlib.sha256()
    with open(MODEL_PATH, 'rb') as f_in:
        for chunk in iter(lambda: f_in.read(4096), b''):
            m.update(chunk)
    file_hash = m.hexdigest().upper()
    if file_hash != EXPECTED_HASH:
        print(f"ERROR: SHA-256 hash mismatch! Expected {EXPECTED_HASH}, got {file_hash}", file=sys.stderr)
        sys.exit(1)
        
    return MODEL_PATH

if __name__ == "__main__":
    download_if_missing()
