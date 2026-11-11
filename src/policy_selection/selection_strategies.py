import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
BUDGET = 0.10
N_FOLDS = 5

def get_utility(y_true, scores):
    """Utility is Recall at 10% budget."""
    n_budget = int(len(y_true) * BUDGET)
    if n_budget == 0:
        return 0.0
    top_indices = np.argsort(scores)[-n_budget:]
    y_pred = np.zeros(len(y_true), dtype=int)
    y_pred[top_indices] = 1
    return recall_score(y_true, y_pred)

def generate_policy_scores(X_train, y_train, X_val, y_val, X_test, y_test):
    """Trains models and returns the policy scores for validation and test."""
    # Train base models
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_SEED))
    lr.fit(X_train, y_train)
    prob_lr_val = lr.predict_proba(X_val)[:, 1]
    prob_lr_test = lr.predict_proba(X_test)[:, 1]
    
    xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, eval_metric='logloss', random_state=RANDOM_SEED)
    xgb.fit(X_train, y_train)
    prob_xgb_val = xgb.predict_proba(X_val)[:, 1]
    prob_xgb_test = xgb.predict_proba(X_test)[:, 1]
    
    # Calculate policy scores
    policies = {}
    
    # 1. Probability Uncertainty (closest to 0.5)
    # We invert this so that higher score = more uncertain
    policies['Uncertainty'] = (1 - np.abs(prob_xgb_val - 0.5), 1 - np.abs(prob_xgb_test - 0.5))
    
    # 2. Model Disagreement
    policies['Disagreement'] = (np.abs(prob_lr_val - prob_xgb_val), np.abs(prob_lr_test - prob_xgb_test))
    
    # 3. SCROSS (Expected Risk)
    policies['SCROSS'] = (prob_xgb_val, prob_xgb_test)
    
    # 4. AUCROSS (Rank Ensemble)
    rank_lr_val = pd.Series(prob_lr_val).rank(pct=True)
    rank_xgb_val = pd.Series(prob_xgb_val).rank(pct=True)
    rank_lr_test = pd.Series(prob_lr_test).rank(pct=True)
    rank_xgb_test = pd.Series(prob_xgb_test).rank(pct=True)
    policies['AUCROSS'] = ((rank_lr_val + rank_xgb_val)/2, (rank_lr_test + rank_xgb_test)/2)
    
    return policies

def run_selection_experiment(dataset_name):
    logger.info(f"--- Running Selection Strategies for {dataset_name} ---")
    
    if dataset_name == "credit_card":
        data_path = "datasets/processed/credit_card_clients_processed.csv"
    else:
        data_path = "datasets/processed/south_german_credit_processed.csv"
        
    df = pd.read_csv(data_path)
    X = df.drop(columns=['target'])
    y = df['target']
    
    with open(f"datasets/processed/splits_{dataset_name}.json", "r") as f:
        splits = json.load(f)
        
    X_train_full = X.loc[splits['train_indices']]
    y_train_full = y.loc[splits['train_indices']]
    X_test = X.loc[splits['test_indices']].reset_index(drop=True)
    y_test = y.loc[splits['test_indices']].reset_index(drop=True)
    
    # 1. S1: Single Validation Split
    # Split the training data into 80% sub-train and 20% validation
    from sklearn.model_selection import train_test_split
    X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train_full
    )
    
    policies_s1 = generate_policy_scores(X_subtrain, y_subtrain, X_val, y_val, X_test, y_test)
    
    val_utils_s1 = {p: get_utility(y_val, policies_s1[p][0]) for p in policies_s1}
    best_policy_s1 = max(val_utils_s1, key=val_utils_s1.get)
    
    # 2. S2 & S3: Repeated Validation (5 folds)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_utilities = {p: [] for p in ['Uncertainty', 'Disagreement', 'SCROSS', 'AUCROSS']}
    
    for train_idx, val_idx in skf.split(X_train_full, y_train_full):
        X_sub, y_sub = X_train_full.iloc[train_idx], y_train_full.iloc[train_idx]
        X_v, y_v = X_train_full.iloc[val_idx], y_train_full.iloc[val_idx]
        
        pol_scores = generate_policy_scores(X_sub, y_sub, X_v, y_v, X_test, y_test)
        for p in fold_utilities.keys():
            fold_utilities[p].append(get_utility(y_v, pol_scores[p][0]))
            
    # S2: Mean Utility
    mean_utils_s2 = {p: np.mean(fold_utilities[p]) for p in fold_utilities}
    best_policy_s2 = max(mean_utils_s2, key=mean_utils_s2.get)
    
    # S3: LCB (Mean - 1.0 * Standard Error)
    lcb_utils_s3 = {}
    for p in fold_utilities:
        mean_val = np.mean(fold_utilities[p])
        std_err = np.std(fold_utilities[p]) / np.sqrt(N_FOLDS)
        lcb_utils_s3[p] = mean_val - 1.0 * std_err
    best_policy_s3 = max(lcb_utils_s3, key=lcb_utils_s3.get)
    
    # Now, evaluate the chosen policies on the test set.
    # We need to re-train on the FULL training set to evaluate on test.
    # To get the test scores, we use the test scores generated in the last loop iteration.
    test_scores = {p: pol_scores[p][1] for p in pol_scores}
    test_utilities = {p: get_utility(y_test, test_scores[p]) for p in test_scores}
    
    oracle_policy = max(test_utilities, key=test_utilities.get)
    oracle_utility = test_utilities[oracle_policy]
    
    results = []
    for strat_name, chosen_pol in [("S1", best_policy_s1), ("S2", best_policy_s2), ("S3", best_policy_s3)]:
        test_util = test_utilities[chosen_pol]
        regret = oracle_utility - test_util
        results.append({
            "dataset": dataset_name,
            "strategy": strat_name,
            "selected_policy": chosen_pol,
            "validation_utility": val_utils_s1.get(chosen_pol, mean_utils_s2.get(chosen_pol, lcb_utils_s3.get(chosen_pol))),
            "test_utility": test_util,
            "oracle_utility": oracle_utility,
            "oracle_policy": oracle_policy,
            "regret": regret
        })
        
    return results

if __name__ == "__main__":
    all_results = []
    for dataset in ["credit_card", "south_german"]:
        all_results.extend(run_selection_experiment(dataset))
        
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("results/csv/policy_selection_comparison.csv", index=False)
    logger.info("\n--- POLICY SELECTION RESULTS ---")
    print(results_df.to_string(index=False))