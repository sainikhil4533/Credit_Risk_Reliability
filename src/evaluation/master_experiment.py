import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BUDGETS = [0.05, 0.10, 0.20]
N_FOLDS = 5
N_REPEATS = 10

def get_utility(y_true, scores, budget):
    n_budget = int(len(y_true) * budget)
    if n_budget == 0: return 0.0
    top_indices = np.argsort(scores)[-n_budget:]
    y_pred = np.zeros(len(y_true), dtype=int)
    y_pred[top_indices] = 1
    return recall_score(y_true, y_pred)

def generate_scores(X_train, y_train, X_eval, y_eval, seed):
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    lr.fit(X_train, y_train)
    prob_lr = lr.predict_proba(X_eval)[:, 1]
    
    xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, eval_metric='logloss', random_state=seed)
    xgb.fit(X_train, y_train)
    prob_xgb = xgb.predict_proba(X_eval)[:, 1]
    
    policies = {}
    policies['Uncertainty'] = 1 - np.abs(prob_xgb - 0.5)
    policies['Disagreement'] = np.abs(prob_lr - prob_xgb)
    policies['SCROSS'] = prob_xgb
    rank_lr = pd.Series(prob_lr).rank(pct=True)
    rank_xgb = pd.Series(prob_xgb).rank(pct=True)
    policies['AUCROSS'] = (rank_lr + rank_xgb) / 2
    return policies

def run_master_experiment(dataset_name, data_path):
    logger.info(f"Running Master Experiment for {dataset_name}...")
    df = pd.read_csv(data_path)
    X = df.drop(columns=['target'])
    y = df['target']
    
    all_results = []
    
    for budget in BUDGETS:
        logger.info(f"  -> Budget: {int(budget*100)}%")
        for seed in range(1, N_REPEATS + 1):
            # Outer Split
            X_train_full, X_test, y_train_full, y_test = train_test_split(
                X, y, test_size=0.2, random_state=seed, stratify=y
            )
            
            # S1
            X_subtrain, X_val, y_subtrain, y_val = train_test_split(
                X_train_full, y_train_full, test_size=0.2, random_state=seed, stratify=y_train_full
            )
            val_policies_s1 = generate_scores(X_subtrain, y_subtrain, X_val, y_val, seed)
            val_utils_s1 = {p: get_utility(y_val, val_policies_s1[p], budget) for p in val_policies_s1}
            best_policy_s1 = max(val_utils_s1, key=val_utils_s1.get)
            
            # S2 & S3
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            fold_utils = {p: [] for p in ['Uncertainty', 'Disagreement', 'SCROSS', 'AUCROSS']}
            
            for train_idx, val_idx in skf.split(X_train_full, y_train_full):
                X_sub, y_sub = X_train_full.iloc[train_idx], y_train_full.iloc[train_idx]
                X_v, y_v = X_train_full.iloc[val_idx], y_train_full.iloc[val_idx]
                pol_scores = generate_scores(X_sub, y_sub, X_v, y_v, seed)
                for p in fold_utils:
                    fold_utils[p].append(get_utility(y_v, pol_scores[p], budget))
                    
            mean_utils_s2 = {p: np.mean(fold_utils[p]) for p in fold_utils}
            best_policy_s2 = max(mean_utils_s2, key=mean_utils_s2.get)
            
            lcb_utils_s3 = {}
            for p in fold_utils:
                mean_val = np.mean(fold_utils[p])
                std_err = np.std(fold_utils[p]) / np.sqrt(N_FOLDS)
                lcb_utils_s3[p] = mean_val - 1.0 * std_err
            best_policy_s3 = max(lcb_utils_s3, key=lcb_utils_s3.get)
            
            # Test Evaluation
            test_policies = generate_scores(X_train_full, y_train_full, X_test, y_test, seed)
            test_utils = {p: get_utility(y_test, test_policies[p], budget) for p in test_policies}
            oracle_policy = max(test_utils, key=test_utils.get)
            oracle_utility = test_utils[oracle_policy]
            
            for strat, chosen in [("S1", best_policy_s1), ("S2", best_policy_s2), ("S3", best_policy_s3)]:
                regret = oracle_utility - test_utils[chosen]
                all_results.append({
                    "dataset": dataset_name, "budget": budget, "seed": seed, 
                    "strategy": strat, "regret": regret, "selected_policy": chosen, "oracle_policy": oracle_policy
                })
                
    return all_results

if __name__ == "__main__":
    Path("figures").mkdir(exist_ok=True)
    Path("results/csv").mkdir(parents=True, exist_ok=True)
    
    all_results = []
    all_results.extend(run_master_experiment("credit_card", "datasets/processed/credit_card_clients_processed.csv"))
    all_results.extend(run_master_experiment("south_german", "datasets/processed/south_german_credit_processed.csv"))
    
    results_df = pd.DataFrame(all_results)
    
    # 1. Save Raw Results
    results_df.to_csv("results/csv/selection_regret_results.csv", index=False)
    
    # 2. Summary Statistics
    summary_df = results_df.groupby(['dataset', 'budget', 'strategy'])['regret'].agg(['mean', 'std', 'median', 'max']).reset_index()
    summary_df.to_csv("results/csv/policy_selection_comparison.csv", index=False)
    logger.info("\n--- SUMMARY STATISTICS ---")
    print(summary_df.to_string(index=False))
    
    # 3. Statistical Tests (Paired t-test S1 vs S2)
    logger.info("\n--- STATISTICAL TESTS (S1 vs S2) ---")
    stat_results = []
    for dataset in ["credit_card", "south_german"]:
        for budget in BUDGETS:
            s1_vals = results_df[(results_df['dataset']==dataset) & (results_df['budget']==budget) & (results_df['strategy']=='S1')]['regret'].values
            s2_vals = results_df[(results_df['dataset']==dataset) & (results_df['budget']==budget) & (results_df['strategy']=='S2')]['regret'].values
            
            t_stat, p_val = stats.ttest_rel(s1_vals, s2_vals)
            stat_results.append({"dataset": dataset, "budget": budget, "t_stat": t_stat, "p_value": p_val, "significant": p_val < 0.05})
            print(f"{dataset} | Budget {int(budget*100)}% | p-value: {p_val:.4f} | Significant: {p_val < 0.05}")
            
    pd.DataFrame(stat_results).to_csv("results/csv/statistical_comparison.csv", index=False)
    
    # 4. Generate Boxplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for ax, budget in zip(axes, BUDGETS):
        # Only plot South German, as it's the one with variance
        subset = results_df[(results_df['dataset'] == 'south_german') & (results_df['budget'] == budget)]
        subset.boxplot(column='regret', by='strategy', ax=ax)
        ax.set_title(f"South German (Budget: {int(budget*100)}%)")
        ax.set_ylabel("Selection Regret")
        ax.set_xlabel("Strategy")
    plt.suptitle("Selection Regret Across Budgets and Strategies (South German)")
    plt.tight_layout()
    plt.savefig("figures/regret_distributions.png")
    plt.close()
    
    logger.info("\nAll final results, statistics, and figures saved successfully!")