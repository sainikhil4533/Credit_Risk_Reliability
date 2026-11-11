import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from sklearn.metrics import recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
BUDGET = 0.10

def evaluate_policy(y_true, selected_indices):
    y_pred = np.zeros(len(y_true), dtype=int)
    y_pred[selected_indices] = 1
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    return recall, precision

def run_advanced_policy_experiment(dataset_name, model_name):
    logger.info(f"--- Running Advanced Policies for {dataset_name} ({model_name}) ---")
    
    if dataset_name == "credit_card":
        data_path = "datasets/processed/credit_card_clients_processed.csv"
    else:
        data_path = "datasets/processed/south_german_credit_processed.csv"
        
    df = pd.read_csv(data_path)
    X = df.drop(columns=['target'])
    y = df['target']
    
    with open(f"datasets/processed/splits_{dataset_name}.json", "r") as f:
        splits = json.load(f)
        
    X_test = X.loc[splits['test_indices']].reset_index(drop=True)
    y_test = y.loc[splits['test_indices']].reset_index(drop=True)
    X_train = X.loc[splits['train_indices']]
    y_train = y.loc[splits['train_indices']]
    
    # Train models
    lr_pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_SEED))
    lr_pipeline.fit(X_train, y_train)
    prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]
    
    xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, eval_metric='logloss', random_state=RANDOM_SEED)
    xgb.fit(X_train, y_train)
    prob_xgb = xgb.predict_proba(X_test)[:, 1]
    
    if model_name == "logistic_regression":
        model_probs = prob_lr
    else:
        model_probs = prob_xgb
        
    n_budget = int(len(y_test) * BUDGET)
    results = []
    
    # --- Policy 4: SCROSS (Expected Risk / Highest Probability) ---
    # SCROSS in a fixed-budget review context prioritizes the most likely positive cases.
    # So we pick the customers with the highest predicted probability of default.
    scross_indices = np.argsort(model_probs)[-n_budget:] 
    recall, precision = evaluate_policy(y_test, scross_indices)
    results.append({"dataset": dataset_name, "policy": "SCROSS (Expected Risk)", "recall": recall, "precision": precision})
    
    # --- Policy 5: AUCROSS (Ranking Optimization) ---
    # AUCROSS aims to maximize AUC. In a fixed budget, this is practically identical 
    # to picking the highest scores. We will implement it as a rank-based ensemble.
    # We combine the probabilities from both models (rank average).
    rank_lr = pd.Series(prob_lr).rank(pct=True)
    rank_xgb = pd.Series(prob_xgb).rank(pct=True)
    aucross_scores = (rank_lr + rank_xgb) / 2
    
    aucross_indices = np.argsort(aucross_scores.values)[-n_budget:]
    recall, precision = evaluate_policy(y_test, aucross_indices)
    results.append({"dataset": dataset_name, "policy": "AUCROSS (Rank Ensemble)", "recall": recall, "precision": precision})
    
    return results

if __name__ == "__main__":
    all_results = []
    
    for dataset in ["credit_card", "south_german"]:
        for model in ["logistic_regression", "xgboost"]:
            all_results.extend(run_advanced_policy_experiment(dataset, model))
            
    new_results_df = pd.DataFrame(all_results)
    
    # Combine with previous results to give a full picture
    try:
        old_results_df = pd.read_csv("results/csv/review_policy_results.csv")
        combined_df = pd.concat([old_results_df, new_results_df], ignore_index=True)
    except FileNotFoundError:
        combined_df = new_results_df
        
    combined_df.to_csv("results/csv/review_policy_results.csv", index=False)
    
    logger.info("\n--- FULL REVIEW POLICY RESULTS (10% Budget) ---")
    # Drop duplicates for cleaner viewing (since we evaluated both LR and XGB for Random/Disagreement earlier)
    clean_df = combined_df.drop_duplicates(subset=['dataset', 'policy']).copy()
    print(clean_df.to_string(index=False))