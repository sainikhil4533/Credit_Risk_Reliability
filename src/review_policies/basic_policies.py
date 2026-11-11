import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from sklearn.metrics import recall_score, precision_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
BUDGET = 0.10  # 10% review budget

def evaluate_policy(y_true, selected_indices):
    """Calculate Recall and Precision for a given selection."""
    # Create a boolean mask of size len(y_true)
    y_pred = np.zeros(len(y_true), dtype=int)
    y_pred[selected_indices] = 1
    
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    return recall, precision

def run_policy_experiment(dataset_name, model_name):
    logger.info(f"--- Running Basic Policies for {dataset_name} ({model_name}) ---")
    
    # 1. Load Data
    if dataset_name == "credit_card":
        data_path = "datasets/processed/credit_card_clients_processed.csv"
    else:
        data_path = "datasets/processed/south_german_credit_processed.csv"
        
    df = pd.read_csv(data_path)
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Load splits
    with open(f"datasets/processed/splits_{dataset_name}.json", "r") as f:
        splits = json.load(f)
        
    X_test = X.loc[splits['test_indices']].reset_index(drop=True)
    y_test = y.loc[splits['test_indices']].reset_index(drop=True)
    
    # We need probabilities from BOTH models for the Disagreement policy.
    # For simplicity, we will train quick models on the fly for this test.
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    
    # Train LR
    lr_pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_SEED))
    lr_pipeline.fit(X.loc[splits['train_indices']], y.loc[splits['train_indices']])
    prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]
    
    # Train XGB
    xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, eval_metric='logloss', random_state=RANDOM_SEED)
    xgb.fit(X.loc[splits['train_indices']], y.loc[splits['train_indices']])
    prob_xgb = xgb.predict_proba(X_test)[:, 1]
    
    # Use the requested model's probabilities for basic policies
    if model_name == "logistic_regression":
        model_probs = prob_lr
    else:
        model_probs = prob_xgb
        
    n_budget = int(len(y_test) * BUDGET)
    results = []
    
    # --- Policy 1: Random Selection ---
    np.random.seed(RANDOM_SEED)
    random_indices = np.random.choice(len(y_test), n_budget, replace=False)
    recall, precision = evaluate_policy(y_test, random_indices)
    results.append({"dataset": dataset_name, "policy": "Random", "recall": recall, "precision": precision})
    
    # --- Policy 2: Probability Uncertainty ---
    # Uncertainty is highest when prob is closest to 0.5
    uncertainty = np.abs(model_probs - 0.5)
    uncertain_indices = np.argsort(uncertainty)[:n_budget]  # Ascending order (closest to 0)
    recall, precision = evaluate_policy(y_test, uncertain_indices)
    results.append({"dataset": dataset_name, "policy": "Probability Uncertainty", "recall": recall, "precision": precision})
    
    # --- Policy 3: Model Disagreement ---
    # Disagreement is highest when LR and XGB differ the most
    disagreement = np.abs(prob_lr - prob_xgb)
    disagree_indices = np.argsort(disagreement)[-n_budget:]  # Descending order (highest disagreement)
    recall, precision = evaluate_policy(y_test, disagree_indices)
    results.append({"dataset": dataset_name, "policy": "Model Disagreement", "recall": recall, "precision": precision})
    
    return results

if __name__ == "__main__":
    all_results = []
    
    for dataset in ["credit_card", "south_german"]:
        for model in ["logistic_regression", "xgboost"]:
            all_results.extend(run_policy_experiment(dataset, model))
            
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("results/csv/review_policy_results.csv", index=False)
    
    logger.info("\n--- BASIC REVIEW POLICY RESULTS (10% Budget) ---")
    print(results_df.to_string(index=False))