import os
import requests
import logging
from pathlib import Path
import yaml
import zipfile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        logger.info(f"File already exists: {dest_path}")
        return
    logger.info(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    logger.info(f"Saved to {dest_path}")

def main():
    config = load_config()
    base_dir = Path(config['paths']['datasets_raw'])
    
    # ---------------------------------------------------------
    # Dataset 1: Default of Credit Card Clients (ID: 350)
    # ---------------------------------------------------------
    d1_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
    d1_dir = base_dir / "default_of_credit_card_clients"
    d1_dir.mkdir(parents=True, exist_ok=True)
    download_file(d1_url, d1_dir / "default of credit card clients.xls")
    
    # ---------------------------------------------------------
    # Dataset 2: South German Credit (ID: 522 ZIP file)
    # ---------------------------------------------------------
    d2_url = "https://archive.ics.uci.edu/static/public/522/south+german+credit.zip"
    d2_dir = base_dir / "south_german_credit"
    d2_dir.mkdir(parents=True, exist_ok=True)
    zip_path = d2_dir / "south+german+credit.zip"
    
    # Download the zip
    download_file(d2_url, zip_path)
    
    # Extract the zip if not already extracted
    asc_file_path = d2_dir / "SouthGermanCredit.asc"
    if not asc_file_path.exists():
        logger.info(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(d2_dir)
        logger.info(f"Extracted to {d2_dir}")
    else:
        logger.info("SouthGermanCredit.asc already exists, skipping extraction.")

if __name__ == "__main__":
    main()