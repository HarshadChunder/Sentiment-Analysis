import os
import gdown

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "datasets")

files = {
    os.path.join(DATA_DIR, "Sentiment140v1.csv"): "https://drive.google.com/uc?id=1vUy-JlcjDz5m0hF8dvLA5BbvWUm-cK8g",
    os.path.join(DATA_DIR, "GoogleNews-vectors-negative300.bin"): "https://drive.google.com/uc?id=16oHiwyiZdeLZpP9VMYShL_YN_GtMV_w5"
}

os.makedirs(DATA_DIR, exist_ok=True)

def download_file(file_path, file_link):
    if not os.path.exists(file_path):
        print(f"Downloading {file_path}...")
        gdown.download(f"{file_link}", file_path, quiet=False)
    else:
        print(f"{file_path} already exists, skipping download.")

for path, link in files.items():
    download_file(path, link)

print("All required files are ready.")