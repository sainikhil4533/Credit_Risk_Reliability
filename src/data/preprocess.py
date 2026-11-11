import pandas as pd
import numpy as np
import logging
from pathlib import Path
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def preprocess_credit_card_clients():
    logger.info("Preprocessing Default of Credit Card Clients...")
    raw_path = Path("datasets/raw/default_of_credit_card_clients/default of credit card clients.xls")
    processed_path = Path("datasets/processed/credit_card_clients_processed.csv")
    
    # Load with header=1 (row 1 is the actual header)
    df = pd.read_excel(raw_path, header=1)
    
    # Drop the ID column if it exists
    df = df.drop(columns=['ID'], errors='ignore')
    
    # Rename target to a standard format
    df = df.rename(columns={'default payment next month': 'target'})
    
    # Drop duplicates to prevent data leakage
    initial_rows = len(df)
    df = df.drop_duplicates()
    logger.info(f"Dropped {initial_rows - len(df)} duplicates. Remaining rows: {len(df)}")
    
    # Ensure target is 0/1 integer
    df['target'] = df['target'].astype(int)
    
    df.to_csv(processed_path, index=False)
    logger.info(f"Saved to {processed_path}")
    return df

def preprocess_south_german_credit():
    logger.info("Preprocessing South German Credit...")
    raw_path = Path("datasets/raw/south_german_credit/SouthGermanCredit.asc")
    processed_path = Path("datasets/processed/south_german_credit_processed.csv")
    
    # Read whitespace-separated file
    df = pd.read_csv(raw_path, sep=r'\s+')
    
    # Flip target: original 1=Good, 0=Bad. We want 1=Bad, 0=Good.
    # So we replace 1 with 0, and 0 with 1.
    df['kredit'] = df['kredit'].map({1: 0, 0: 1})
    df = df.rename(columns={'kredit': 'target'})
    
    # Mapping dictionary for South German Credit features (based on codetable.txt)
    # We will map codes like 'A11' to 'A11_less_than_0_DM', etc.
    # Note: The first 4 columns are numerical (laufkont, laufzeit, moral, verw), 
    # the rest are categorical.
    
    feature_mapping = {
        'laufkont': {1: 'no_checking', 2: 'less_than_0', 3: '0_to_200', 4: 'greater_200'},
        'laufzeit': 'numeric', # duration in months
        'moral': {1: 'none', 2: 'co_applicant', 3: 'guarantor', 4: 'owner'},
        'verw': {1: 'unskilled', 2: 'skilled', 3: 'management', 4: 'unemployed'},
        'sparkont': {1: 'less_than_100', 2: '100_to_500', 3: '500_to_1000', 4: 'greater_1000', 5: 'unknown'},
        'beszeit': {1: 'unemployed', 2: 'less_than_1', 3: '1_to_4', 4: '4_to_7', 5: 'greater_7'},
        'rate': {1: 'less_than_20', 2: '20_to_25', 3: '25_to_35', 4: 'greater_35'},
        'famges': {1: 'male_single', 2: 'male_married', 3: 'male_divorced', 4: 'female_single', 5: 'female_married'},
        'buerge': {1: 'none', 2: 'co_applicant', 3: 'guarantor'},
        'wohnzeit': {1: 'less_than_1', 2: '1_to_3', 3: '3_to_6', 4: 'greater_6'},
        'verm': {1: 'less_than_100', 2: '100_to_500', 3: '500_to_1000', 4: 'greater_1000'},
        'alter': 'numeric', # age
        'weitkred': {1: 'none', 2: 'bank', 3: 'stores', 4: 'other'},
        'wohn': {1: 'rent', 2: 'own', 3: 'free'},
        'bishkred': {1: 'none', 2: 'paid_back', 3: 'current', 4: 'past_due'},
        'beruf': {1: 'unemployed', 2: 'unskilled', 3: 'skilled', 4: 'management'},
        'pers': {1: 'male', 2: 'female'},
        'gueltig': {1: 'less_than_1', 2: '1_to_4', 3: 'greater_4'},
        'nrkred': {1: 'none', 2: '1_to_2', 3: 'greater_2'},
        'telefon': {1: 'yes', 2: 'no'},
        'ausland': {1: 'yes', 2: 'no'}
    }
    
    # Apply mappings
    for col, mapping in feature_mapping.items():
        if col in df.columns and mapping != 'numeric':
            # Add prefix to column values to avoid collisions
            df[col] = df[col].map(mapping)
            df[col] = df[col].astype(str)
            
    # One-Hot Encode categorical columns
    df = pd.get_dummies(df, drop_first=True)
    
    # Convert boolean columns to int (0/1)
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
            
    # Save processed dataset
    df.to_csv(processed_path, index=False)
    logger.info(f"Saved to {processed_path}. Shape: {df.shape}")
    return df

if __name__ == "__main__":
    config = load_config()
    
    # Ensure processed directory exists
    Path("datasets/processed").mkdir(parents=True, exist_ok=True)
    
    df1 = preprocess_credit_card_clients()
    df2 = preprocess_south_german_credit()
    
    logger.info("Preprocessing complete!")