Credit-Risk Review-Policy Reliability (v2)
Validation-to-Test Reliability of Credit-Risk Review Policy Selection Under Fixed Review Budgets.

📌 Project Overview
Financial institutions use machine learning to identify customers for manual review but operate under strict budget constraints. This project investigates whether a review policy selected using validation data remains optimal on unseen test data.

We evaluate 5 Review Policies (Random, Probability Uncertainty, Model Disagreement, SCROSS, AUCROSS) and 3 Policy-Selection Strategies (S1: Single Validation, S2: Repeated Validation, S3: Uncertainty-Aware).

📂 Folder Structure
text
credit_risk_reliability_v2/
│
├── datasets/
│   ├── raw/                 # Original UCI datasets (downloaded automatically)
│   └── processed/           # Cleaned, one-hot encoded CSVs ready for modeling
│
├── src/                     # All Python source code
│   ├── data/                # Scripts to download and preprocess data
│   ├── models/              # Baseline model training (LR & XGBoost)
│   ├── calibration/         # Sigmoid and Isotonic calibration experiments
│   ├── review_policies/     # Level 1: Customer selection policies
│   ├── policy_selection/    # Level 2: S1, S2, S3 selection strategies
│   ├── evaluation/          # Master experiment script & statistical tests
│   └── utils/               # Helper functions
│
├── results/
│   ├── csv/                 # Output metrics, tables, and statistical tests
│   └── json/                # Saved train/test split indices (for reproducibility)
│
├── figures/                 # Generated calibration curves and regret boxplots
├── requirements.txt         # Pinned Python dependencies
└── config.yaml              # Experiment configuration (seeds, budgets, lambda)
⚙️ Setup & Installation
This project uses Python 3.13. To reproduce the environment:

Clone the repository and navigate to the folder:

powershell
cd credit_risk_reliability_v2
Create and activate a virtual environment:

powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
Install pinned dependencies:

powershell
pip install -r requirements.txt
pip install xlrd==2.0.1  # Required for reading the legacy .xls format
🚀 Execution Pipeline
To reproduce the results from scratch, run the scripts in this exact order:

powershell
# 1. Download the official UCI datasets
python src/data/download_data.py

# 2. Audit the data (check duplicates, missing values)
python src/data/audit_data.py

# 3. Clean, align targets, and preprocess (One-Hot Encoding)
python src/data/preprocess.py

# 4. Train baseline models (Logistic Regression & XGBoost)
python src/models/train_models.py

# 5. Evaluate model calibration (Sigmoid vs Isotonic)
python src/calibration/run_calibration.py

# 6. Evaluate basic review policies at 10% budget
python src/review_policies/basic_policies.py

# 7. Evaluate advanced review policies (SCROSS, AUCROSS)
python src/review_policies/advanced_policies.py

# 8. Run the Master Experiment (Budget Sweep, 10 Seeds, Stats, Boxplots)
python src/evaluation/master_experiment.py
📊 Key Research Findings
The Size Dependence: On the large Credit Card dataset (30,000 rows), all validation strategies achieved near-zero selection regret. However, on the small South German dataset (1,000 rows), S1 (Single Validation) suffered catastrophic worst-case regret (11.6%).

The Safety Net of S2: S2 (Repeated Validation) slashed the worst-case regret on the small dataset from 11.6% down to 3.3%, acting as a necessary safeguard against noisy data splits.

Statistical Honesty: Paired t-tests across 10 random seeds yielded p-values > 0.05 for all budgets. This indicates that while S2 is practically safer, the average improvement over S1 is not statistically significant due to the inherent noise of credit-risk data.

Budget Effects: At a 20% review budget, all policies converged to the exact same performance, resulting in zero regret.

📝 Outputs Generated
CSV Files (results/csv/): dataset_summary.csv, base_model_results.csv, calibration_comparison.csv, review_policy_results.csv, policy_selection_comparison.csv, selection_regret_results.csv, statistical_comparison.csv.

Figures (figures/): Reliability diagrams (calibration curves) for both datasets, and a boxplot showing the distribution of selection regret across 10 seeds.

⚠️ Limitations & Integrity Note
The SCROSS and AUCROSS algorithms in this repository are adapted as "Expected Risk" and "Rank Ensemble" heuristics for fixed-budget queues, rather than exact mathematical reproductions of the original selective classification frameworks.

The UCI datasets lack independent chronological deployment periods; all splits are random and stratified.

