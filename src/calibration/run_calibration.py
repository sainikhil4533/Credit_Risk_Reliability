import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RANDOM_SEED = 42

def get_uncalibrated_model(model_name):
    if model_name == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_SEED))
    else:
        return XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, eval_metric='logloss', random_state=RANDOM_SEED)

def plot_calibration_curve(y_true, y_prob, model_label, dataset_name, model_name, ax):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
    ax.plot(prob_pred, prob_true, marker='.', label=model_label)

def run_calibration_experiment(dataset_name):
    logger.info(f"--- Running Calibration for {dataset_name} ---")
    
    # 1. Load Data
    if dataset_name == "credit_card":
        data_path = "datasets/processed/credit_card_clients_processed.csv"
    else:
        data_path = "datasets/processed/south_german_credit_processed.csv"
        
    df = pd.read_csv(data_path)
    X = df.drop(columns=['target'])
    y = df['target']
    
    # 2. Load Train/Test Indices from Stage 3
    with open(f"datasets/processed/splits_{dataset_name}.json", "r") as f:
        splits = json.load(f)
        
    X_train_full = X.loc[splits['train_indices']]
    y_train_full = y.loc[splits['train_indices']]
    
    X_test = X.loc[splits['test_indices']]
    y_test = y.loc[splits['test_indices']]
    
    # 3. Split Train into Train (80%) and Calibration (20%)
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train_full
    )
    
    results = []
    
    # 4. Iterate over models
    for model_name in ["logistic_regression", "xgboost"]:
        logger.info(f"Processing {model_name}...")
        
        # Get base model and FIT IT FIRST (Fixes the NotFittedError)
        base_model = get_uncalibrated_model(model_name)
        base_model.fit(X_train, y_train)
        
        # A. Uncalibrated
        prob_uncal = base_model.predict_proba(X_test)[:, 1]
        brier_uncal = brier_score_loss(y_test, prob_uncal)
        results.append({"dataset": dataset_name, "model": model_name, "calibration": "uncalibrated", "brier_score": brier_uncal})
        
        # B. Sigmoid Calibration (pass the FITTED base_model)
        sig_model = CalibratedClassifierCV(estimator=base_model, method='sigmoid', cv="prefit")
        sig_model.fit(X_cal, y_cal)
        prob_sig = sig_model.predict_proba(X_test)[:, 1]
        brier_sig = brier_score_loss(y_test, prob_sig)
        results.append({"dataset": dataset_name, "model": model_name, "calibration": "sigmoid", "brier_score": brier_sig})
        
        # C. Isotonic Calibration (pass the FITTED base_model)
        iso_model = CalibratedClassifierCV(estimator=base_model, method='isotonic', cv="prefit")
        iso_model.fit(X_cal, y_cal)
        prob_iso = iso_model.predict_proba(X_test)[:, 1]
        brier_iso = brier_score_loss(y_test, prob_iso)
        results.append({"dataset": dataset_name, "model": model_name, "calibration": "isotonic", "brier_score": brier_iso})
        
        # Plotting
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plot_calibration_curve(y_test, prob_uncal, "Uncalibrated", dataset_name, model_name, ax)
        plot_calibration_curve(y_test, prob_sig, "Sigmoid", dataset_name, model_name, ax)
        plot_calibration_curve(y_test, prob_iso, "Isotonic", dataset_name, model_name, ax)
        ax.set_ylabel("Fraction of positives (Actual)")
        ax.set_xlabel("Mean predicted probability (Predicted)")
        ax.set_title(f"Calibration Curve: {dataset_name.replace('_', ' ').title()} - {model_name.replace('_', ' ').title()}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"figures/calibration_curve_{dataset_name}_{model_name}.png")
        plt.close()
        
    return results

if __name__ == "__main__":
    Path("figures").mkdir(exist_ok=True)
    Path("results/csv").mkdir(parents=True, exist_ok=True)
    
    all_results = []
    all_results.extend(run_calibration_experiment("credit_card"))
    all_results.extend(run_calibration_experiment("south_german"))
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("results/csv/calibration_comparison.csv", index=False)
    logger.info("\n--- CALIBRATION RESULTS ---")
    print(results_df.to_string(index=False))