import pandas as pd
import numpy as np
import logging
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure directories exist
Path("models").mkdir(exist_ok=True)
Path("results/csv").mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

def train_and_evaluate(dataset_name, data_path):
    logger.info(f"--- Training models for {dataset_name} ---")
    
    # 1. Load Data
    df = pd.read_csv(data_path)
    X = df.drop(columns=['target'])
    y = df['target']
    
    # 2. Split Data (80% Train, 20% Test)
    # We stratify to maintain the class imbalance ratio in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    # Save split indices for reproducibility
    splits = {
        "train_indices": list(X_train.index),
        "test_indices": list(X_test.index)
    }
    with open(f"datasets/processed/splits_{dataset_name}.json", "w") as f:
        json.dump(splits, f)
    
    results = []
    
    # 3. Logistic Regression
    logger.info("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    lr.fit(X_train, y_train)
    y_pred_proba_lr = lr.predict_proba(X_test)[:, 1]
    
    results.append({
        "dataset": dataset_name,
        "model": "Logistic Regression",
        "auc": roc_auc_score(y_test, y_pred_proba_lr),
        "accuracy": accuracy_score(y_test, (y_pred_proba_lr > 0.5).astype(int)),
        "brier_score": brier_score_loss(y_test, y_pred_proba_lr)
    })
    joblib.dump(lr, f"models/lr_{dataset_name}.joblib")
    
    # 4. XGBoost
    logger.info("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.1, 
        eval_metric='logloss', 
        random_state=RANDOM_SEED
    )
    xgb.fit(X_train, y_train)
    y_pred_proba_xgb = xgb.predict_proba(X_test)[:, 1]
    
    results.append({
        "dataset": dataset_name,
        "model": "XGBoost",
        "auc": roc_auc_score(y_test, y_pred_proba_xgb),
        "accuracy": accuracy_score(y_test, (y_pred_proba_xgb > 0.5).astype(int)),
        "brier_score": brier_score_loss(y_test, y_pred_proba_xgb)
    })
    joblib.dump(xgb, f"models/xgb_{dataset_name}.joblib")
    
    return results

if __name__ == "__main__":
    all_results = []
    
    # Run for both datasets
    all_results.extend(train_and_evaluate("credit_card", "datasets/processed/credit_card_clients_processed.csv"))
    all_results.extend(train_and_evaluate("south_german", "datasets/processed/south_german_credit_processed.csv"))
    
    # Save results to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("results/csv/base_model_results.csv", index=False)
    
    logger.info("\n--- BASE MODEL RESULTS ---")
    print(results_df.to_string(index=False))