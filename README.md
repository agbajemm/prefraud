# Pre-Fraud Indication Factors: Behavioural Drift Analysis

**MSc Research Project** — Investigating whether indirect behavioural indicators in financial transaction data can predict fraud *before* it occurs, once direct fraud indicators have been systematically removed.

## Research Question

> Can structured analysis of indirect behavioural signals partially recover predictive power lost when direct fraud indicators are unavailable — i.e., before fraud has actually been committed?

## Methodology

1. **Baseline Model** — Build a high-accuracy fraud detector using all available features (target: AUC-ROC > 0.95)
2. **Feature Ablation** — Systematically remove direct fraud indicators in 7 progressive stages
3. **Drift Decomposition** — Decompose behavioural change into 5 measurable drift dimensions (temporal, device, amount, email/identity, velocity)
4. **Pre-Fraud Model** — Train prediction models using only indirect features and drift scores
5. **Evaluation** — Compare performance across all stages with statistical rigour

## Project Structure

```
pre-fraud-detection/
├── data/
│   ├── raw/              # Original datasets (not committed)
│   ├── processed/        # Preprocessed train/val/test splits
│   └── features/         # Feature classifications and metadata
├── src/
│   ├── data_loader.py          # Data loading, preprocessing, chronological splitting
│   ├── feature_engineering.py  # Derived features and feature taxonomy
│   ├── baseline_model.py       # Full-feature XGBoost fraud detector
│   ├── feature_ablation.py     # Progressive feature removal engine
│   ├── drift_analysis.py       # Behavioural drift decomposition
│   ├── pre_fraud_model.py      # LightGBM / Neural Net / Ensemble models
│   ├── evaluation.py           # Comprehensive evaluation framework
│   └── visualisation.py        # Publication-quality plotting
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_feature_ablation.ipynb
│   ├── 04_drift_decomposition.ipynb
│   └── 05_pre_fraud_evaluation.ipynb
├── tests/
│   ├── test_feature_engineering.py
│   ├── test_models.py
│   └── test_drift_metrics.py
├── results/
│   ├── figures/          # All generated plots
│   ├── tables/           # CSV/JSON result tables
│   └── logs/             # Training logs
├── config.yaml           # All hyperparameters and paths
├── requirements.txt      # Python dependencies
└── README.md
```

## Datasets

### Primary: IEEE-CIS Fraud Detection (Kaggle)
- **URL:** https://www.kaggle.com/c/ieee-fraud-detection/data
- **Size:** ~1.2 GB, 590,540 transactions, 394 features
- **Files:** `train_transaction.csv`, `train_identity.csv`
- Place in `data/raw/`

### Secondary: Credit Card Fraud Detection (MLG-ULB)
- **URL:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Size:** ~150 MB, 284,807 transactions, 30 features
- **File:** `creditcard.csv`
- Place in `data/raw/`

## Setup and Reproduction

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download datasets to data/raw/ (see above)

# 3. Preprocess data (chronological split)
python src/data_loader.py

# 4. Train baseline model
python src/baseline_model.py

# 5. Run feature ablation study
python src/feature_ablation.py

# 6. Compute behavioural drift scores
python src/drift_analysis.py

# 7. Train and evaluate pre-fraud models
python src/pre_fraud_model.py

# 8. Run comprehensive evaluation
python src/evaluation.py

# 9. Run tests
pytest tests/ -v
```

Alternatively, run the Jupyter notebooks in order (01 through 05) for an interactive walkthrough.

## Expected Results

| Model | AUC-ROC | Notes |
|-------|---------|-------|
| Baseline (full features) | > 0.95 | Performance ceiling |
| After full ablation | 0.60–0.75 | Direct detection fails |
| Pre-fraud (with drift) | 0.70–0.85 | Drift recovers signal |

The gap between ablated and pre-fraud models represents the contribution of behavioural drift analysis — the core thesis finding.

## Key Design Decisions

- **Chronological splitting** (not random) to simulate real deployment
- **Progressive ablation** rather than random feature removal
- **Five drift dimensions** capturing distinct behavioural aspects
- **Lead time analysis** to verify signals precede fraud
- **Three model architectures** compared with statistical tests
- **Cross-dataset validation** on the secondary Credit Card dataset

## Reproducibility

- All random seeds set to 42
- Hyperparameters stored in `config.yaml`
- Chronological data splitting ensures temporal validity
- 5-fold stratified cross-validation for all evaluations
- Statistical significance tests (paired t-test, McNemar's) reported
