# Pre-Fraud Indication Factors: Behavioural Drift Analysis

**MSc Final Project** -- Michael Okonkwo (23090303), University of Hertfordshire

Investigating whether indirect behavioural indicators in financial transaction data can predict fraud *before* it occurs, once direct fraud indicators have been systematically removed.

## Research Question

> Can structured analysis of indirect behavioural signals partially recover predictive power lost when direct fraud indicators are unavailable -- i.e., before fraud has actually been committed?

---

## Technology Stack

| Layer | Technology | Why it was chosen |
|-------|-----------|-------------------|
| Language | Python 3.12 | Standard for ML research, rich ecosystem |
| ML Framework | scikit-learn 1.6 | Provides pipelines, CV, stacking, MLP, metrics |
| Gradient Boosting | XGBoost 3.2 | Baseline fraud detector -- fast, handles missing values natively |
| Gradient Boosting | LightGBM 4.6 | Pre-fraud model -- faster than XGBoost on large data, `is_unbalance` for class weights |
| Hyperparameter Tuning | Optuna 4.3 | Bayesian optimisation with TPE sampler -- more efficient than grid/random search |
| Explainability | SHAP 0.46 | TreeExplainer for feature importance -- model-agnostic, theoretically grounded |
| Data Format | Parquet (via pyarrow) | Columnar storage -- 10x faster read than CSV for 500K+ rows |
| Visualisation | matplotlib + seaborn | Publication-quality figures at 300 DPI |
| Configuration | YAML | Human-readable, keeps all hyperparameters in one place |
| Testing | pytest | Standard Python test runner |

---

## How to Read and Understand the Codebase

The project follows a **pipeline architecture** -- each module does one job, and they run in sequence. Start here:

### 1. `config.yaml` -- Read this first

Every hyperparameter, file path, and feature group name lives here. When you see `config["data"]["processed_path"]` in the code, this is where it comes from. Understanding the config means you understand what knobs the system has.

### 2. `src/data_loader.py` (222 lines) -- Data pipeline

**Key class:** `DataLoader`

What it does:
- Loads the two IEEE-CIS CSV files and joins them on `TransactionID`
- Fills missing values (median for numbers, mode for categories)
- Encodes categorical columns as integers (label encoding)
- Splits the data **chronologically** (by `TransactionDT`), not randomly -- this is critical because in production you train on past data and predict the future

**Key method to understand:** `chronological_split()` -- this is why the test AUC-ROC is always lower than CV AUC-ROC; the test set comes from a later time period.

### 3. `src/feature_engineering.py` (194 lines) -- Feature taxonomy

**Not a model** -- this is a reference file that defines which features belong to which group:
- `DIRECT_INDICATOR_GROUPS`: features the ablation removes (TransactionAmt, card IDs, addresses, etc.)
- `BEHAVIOURAL_GROUPS`: features that remain after full ablation (timedeltas, device info, email domains)

Read the dictionaries at the top to understand the ablation stages.

### 4. `src/baseline_model.py` (871 lines) -- Full-feature fraud detector

**Key class:** `BaselineModel`

This trains an XGBoost classifier on ALL 432 features to establish the performance ceiling. Key concepts:
- `scale_pos_weight`: handles the 27:1 class imbalance (27 legit transactions per fraud)
- `train_cv()`: runs 5-fold stratified cross-validation, records per-fold metrics
- `compute_feature_importances()`: combines 4 importance sources (XGBoost gain, SHAP values, permutation importance, mutual information) into a single composite score -- this ranking drives the ablation order
- `evaluate()`: tests the model on the held-out chronological test set

### 5. `src/feature_ablation.py` (479 lines) -- The core experiment

**Key class:** `FeatureAblationEngine`

This is the heart of the methodology. It progressively removes feature groups in 8 stages:

```
Stage 0: All 432 features (baseline)
Stage 1: Remove TransactionAmt
Stage 2: Remove card1-card6
Stage 3: Remove addr1, addr2, dist1, dist2
Stage 4: Remove M1-M9 (match features)
Stage 5: Remove C1-C14 (counting features)
Stage 6: Remove top 50% of Vesta features (by SHAP rank)
Stage 7: Keep only 49 behavioural features
```

At each stage, it retrains XGBoost on the remaining features and measures the drop in AUC-ROC. A paired t-test checks if each drop is statistically significant.

**Key method:** `run_ablation()` -- the main loop. `_resolve_features_to_remove()` decides which columns to drop at each stage (Stage 6 uses dynamic SHAP-based selection).

### 6. `src/drift_analysis.py` (436 lines) -- Behavioural drift decomposition

**Key class:** `BehaviouralDriftAnalyser`

Computes 5 independent "drift scores" for each transaction, measuring how much its behavioural patterns differ from normal:

| Method | What it measures | Statistical metric |
|--------|-----------------|-------------------|
| `compute_temporal_drift()` | Time-of-day pattern shift | Jensen-Shannon divergence |
| `compute_device_drift()` | Device/browser diversity | Entropy-based index |
| `compute_amount_drift()` | Counting feature anomalies | Z-score |
| `compute_email_drift()` | Email domain rarity | Rarity score + missingness |
| `compute_velocity_drift()` | Transaction timing gaps | Wasserstein distance |

`compute_all_drift_scores()` runs all 5 and returns a DataFrame with one row per transaction. These scores become extra features for the pre-fraud model.

### 7. `src/pre_fraud_model.py` (1,848 lines) -- Pre-fraud prediction

**Key classes:** `PreFraudLightGBM`, `PreFraudNeuralNet`, `PreFraudEnsemble`, `PreFraudModel`

Takes the 49 behavioural features (from Stage 7) plus the 6 drift scores = 55 features total, and trains 3 models:
- **LightGBM** with Optuna tuning (15 trials)
- **MLP neural network** with Optuna tuning (15 trials, subsampled for speed)
- **Stacking ensemble** combining the above with logistic regression

`PreFraudModel` is the orchestrator that trains all three, evaluates on the test set, runs McNemar's pairwise significance tests, and computes SHAP values.

The `main()` function at the bottom shows the full pipeline: load data -> load drift scores -> load ablation boundary -> build feature matrix -> train -> evaluate -> save.

### 8. `src/evaluation.py` (1,672 lines) -- Cross-dataset validation

**Key class:** `ComprehensiveEvaluator`

Runs the evaluation report: loads trained models, computes metrics, generates comparison plots, analyses false positives/negatives.

**Important method:** `validate_on_credit_card()` -- replicates the ablation experiment on the ULB Credit Card dataset to check if the findings generalise to a different dataset with different features (PCA-transformed V1-V28).

### 9. `src/visualisation.py` (937 lines) -- Plotting

**Key class:** `Visualiser`

14 plotting methods, all generating 300 DPI PNGs. Each method takes data + a save path and produces one figure. Used by the other modules via `vis.plot_roc_curve(...)` etc.

### 10. `tests/` -- Unit tests

3 test files covering feature engineering, model training, and drift metrics. Run with `pytest tests/ -v`.

### 11. `notebooks/` -- Interactive walkthroughs

Numbered 01-05, matching the pipeline stages. These are for exploration and presentation -- the real pipeline runs through the `src/` modules.

---

## Project Structure

```
prefraud/
  config.yaml                    # All hyperparameters and paths
  requirements.txt               # Python dependencies
  src/
    __init__.py
    data_loader.py               # Data loading, merging, chronological splitting
    feature_engineering.py       # Feature taxonomy (direct vs behavioural)
    baseline_model.py            # XGBoost baseline with composite importance
    feature_ablation.py          # 8-stage progressive feature removal
    drift_analysis.py            # 5-dimension behavioural drift scores
    pre_fraud_model.py           # LightGBM / MLP / Ensemble comparison
    evaluation.py                # Cross-dataset validation and analysis
    visualisation.py             # Publication-quality plots (300 DPI)
  tests/
    test_feature_engineering.py  # 14 tests
    test_models.py               # 13 tests
    test_drift_metrics.py        # 14 tests
  notebooks/
    01_exploratory_analysis.ipynb
    02_baseline_model.ipynb
    03_feature_ablation.ipynb
    04_drift_decomposition.ipynb
    05_pre_fraud_evaluation.ipynb
  results/
    figures/                     # Generated plots (28 PNGs)
    tables/                      # CSV/JSON result tables (23 files)
  data/
    raw/                         # Source datasets (not in repo)
    processed/                   # Parquet files (not in repo)
```

---

## Datasets

### Primary: IEEE-CIS Fraud Detection (Kaggle)
- https://www.kaggle.com/c/ieee-fraud-detection/data
- 590,540 transactions, 434 features after merge
- Place `train_transaction.csv` and `train_identity.csv` in `data/raw/`

### Secondary: Credit Card Fraud Detection (MLG-ULB)
- https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- 284,807 transactions, 30 PCA-transformed features
- Place `creditcard.csv` in `data/raw/`

---

## Running the Pipeline

```bash
# Install dependencies
pip install -r requirements.txt

# Run each stage in order
python -m src.data_loader
python -m src.baseline_model
python -m src.feature_ablation
python -m src.drift_analysis
python -m src.pre_fraud_model
python -m src.evaluation

# Run tests
pytest tests/ -v
```

Note: use `python -m src.<module>` (not `python src/<module>.py`) so that Python resolves the package imports correctly.

---

## Key Design Decisions

- **Chronological splitting** (not random) to simulate real deployment -- the model never sees future data during training
- **Progressive ablation** rather than random feature removal -- mirrors how fraud indicators become available in practice
- **Composite feature importance** from 4 sources -- no single importance metric is reliable alone
- **Five drift dimensions** capturing distinct behavioural aspects -- not a single monolithic "anomaly score"
- **Three model architectures** compared with McNemar's test -- to determine whether architecture choice matters for this feature set
- **Cross-dataset validation** on Credit Card data -- to check if the degradation pattern is dataset-specific or general

## Reproducibility

- All random seeds set to 42
- All hyperparameters in `config.yaml`
- Chronological data splitting ensures temporal validity
- 5-fold stratified cross-validation for all evaluations
- Statistical significance tests (paired t-test, McNemar's) for all comparisons
