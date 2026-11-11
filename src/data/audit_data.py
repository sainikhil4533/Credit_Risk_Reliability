import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def audit_credit_card_clients():
    logger.info("Auditing Default of Credit Card Clients...")
    path = Path("datasets/raw/default_of_credit_card_clients/default of credit card clients.xls")
    
    # The first row is an ID header, the second row has the true column names.
    df = pd.read_excel(path, header=1)
    df = df.drop(columns=['ID'], errors='ignore')
    
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Missing values: {df.isnull().sum().sum()}")
    logger.info(f"Duplicates: {df.duplicated().sum()}")
    
    # Basic stats for target
    target = 'default payment next month'
    if target in df.columns:
        logger.info(f"Target distribution:\n{df[target].value_counts(normalize=True)}")
    else:
        logger.warning("Target column not found!")
    
    return df

def audit_south_german_credit():
    logger.info("Auditing South German Credit...")
    path = Path("datasets/raw/south_german_credit/SouthGermanCredit.asc")
    
    # Read as whitespace-separated
    df = pd.read_csv(path, sep=r'\s+')
    
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Missing values: {df.isnull().sum().sum()}")
    logger.info(f"Duplicates: {df.duplicated().sum()}")
    
    # The target in this dataset is the last column, typically 'kredit' or 'credit_risk'
    target = df.columns[-1]
    logger.info(f"Target column found: '{target}'")
    logger.info(f"Target distribution:\n{df[target].value_counts(normalize=True)}")
    
    return df

if __name__ == "__main__":
    config = load_config()
    df1 = audit_credit_card_clients()
    df2 = audit_south_german_credit()
    
    # Save a simple summary CSV
    summary = pd.DataFrame([
        {"dataset": "Default of Credit Card Clients", "rows": df1.shape[0], "cols": df1.shape[1], "missing": df1.isnull().sum().sum(), "duplicates": df1.duplicated().sum()},
        {"dataset": "South German Credit", "rows": df2.shape[0], "cols": df2.shape[1], "missing": df2.isnull().sum().sum(), "duplicates": df2.duplicated().sum()}
    ])
    summary.to_csv("results/csv/dataset_summary.csv", index=False)
    logger.info("Saved dataset_summary.csv")