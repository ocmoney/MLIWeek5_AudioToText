import os
import requests
from tqdm import tqdm
import zipfile

def download_fma_small(destination_dir="fma_small"):
    url = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
    local_zip = os.path.join(destination_dir, "fma_small.zip")
    os.makedirs(destination_dir, exist_ok=True)

    # Download with progress bar
    if not os.path.exists(local_zip):
        print("Downloading FMA-small dataset...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(local_zip, 'wb') as f, tqdm(
            desc=local_zip,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
    else:
        print("FMA-small zip already downloaded.")

    # Extract
    extract_dir = os.path.join(destination_dir, "fma_small")
    if not os.path.exists(extract_dir):
        print("Extracting...")
        with zipfile.ZipFile(local_zip, 'r') as zip_ref:
            zip_ref.extractall(destination_dir)
        print("Extraction complete.")
    else:
        print("FMA-small already extracted.")

if __name__ == "__main__":
    download_fma_small()
