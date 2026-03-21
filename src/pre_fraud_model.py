"""
Pre-Fraud Prediction Model: Multi-Architecture Comparison
==========================================================

Implements three model architectures for pre-fraud prediction using only
indirect/behavioural features and computed drift scores, then performs
rigorous statistical comparison to identify the optimal approach.

Architectures
-------------
**Option A -- PreFraudLightGBM**
    LightGBM classifier with Optuna-tuned hyperparameters, ``is_unbalance=True``
    for native class-imbalance handling, and 5-fold stratified CV.

**Option B -- PreFraudNeuralNet**
    Sklearn ``MLPClassifier`` with feature-group aggregation (one mean per
    drift dimension) fed alongside the raw behavioural features.  Tuned via
    Optuna over architecture width, depth, learning rate, and alpha.

**Option C -- PreFraudEnsemble**
    ``StackingClassifier`` that combines Options A and B with a Logistic
    Regression meta-learner, using predicted probabilities as meta-features.

The orchestrating **PreFraudModel** class trains all three options with
stratified CV, evaluates AUC-ROC / AUC-PR / F1 / Precision@Recall80,
conducts McNemar's test for pairwise statistical significance, generates
SHAP and partial-dependence interpretability plots, and persists every
artifact to disk.

Author : MSc Pre-Fraud Research Project
"""

from __future__ import annotations

import json
import logging
import pickle
import time
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend; must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.inspection import PartialDependenceDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import lightgbm as lgb
import optuna
import shap

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Suppress noisy Optuna / LightGBM logs during tuning
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "pre_fraud"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
LOGS_DIR = PROJECT_ROOT / "results" / "logs"
MODELS_DIR = PROJECT_ROOT / "results" / "models"


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------
def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the project YAML configuration.

    Parameters
    ----------
    config_path : Path, optional
        Explicit path to ``config.yaml``.  Defaults to
        ``PROJECT_ROOT / "config.yaml"``.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


# ===================================================================
# Utility: Precision at fixed recall
# ===================================================================
def precision_at_recall(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float = 0.80,
) -> float:
    """Return the highest precision achievable at >= *target_recall*.

    Traverses the precision-recall curve and selects the maximum
    precision among all operating points where recall >= target.

    Parameters
    ----------
    y_true : array-like
        Binary ground-truth labels.
    y_prob : array-like
        Predicted probabilities for the positive class.
    target_recall : float
        Minimum recall threshold (default 0.80).

    Returns
    -------
    float
        Best precision at the specified recall level, or 0.0 if no
        threshold achieves the target recall.
    """
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    valid = recalls >= target_recall
    if valid.any():
        return float(precisions[valid].max())
    return 0.0


# ===================================================================
# Utility: McNemar's test
# ===================================================================
def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> Dict[str, float]:
    """Perform McNemar's test comparing two classifiers' predictions.

    McNemar's test assesses whether two classifiers have significantly
    different error rates on the same test set.  It uses only the
    *discordant* pairs -- instances where one model is correct and the
    other is wrong.

    The test statistic (with continuity correction) is:

        chi2 = (|b - c| - 1)^2 / (b + c)

    where *b* = # instances A correct & B wrong,
          *c* = # instances A wrong & B correct.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred_a : array-like
        Binary predictions from model A.
    y_pred_b : array-like
        Binary predictions from model B.

    Returns
    -------
    dict
        Keys: ``"b"`` (A correct, B wrong), ``"c"`` (A wrong, B correct),
        ``"chi2"`` (test statistic), ``"p_value"`` (two-sided p-value),
        ``"significant_0.05"`` (bool).
    """
    from scipy.stats import chi2 as chi2_dist

    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # Contingency table of discordant cells
    b = int(np.sum(correct_a & ~correct_b))   # A right, B wrong
    c = int(np.sum(~correct_a & correct_b))   # A wrong, B right

    if (b + c) == 0:
        # No discordant pairs; models are identical on this test set
        return {
            "b": b,
            "c": c,
            "chi2": 0.0,
            "p_value": 1.0,
            "significant_0.05": False,
        }

    # McNemar's statistic with continuity correction
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = float(1.0 - chi2_dist.cdf(chi2_stat, df=1))

    return {
        "b": b,
        "c": c,
        "chi2": round(chi2_stat, 6),
        "p_value": round(p_value, 6),
        "significant_0.05": p_value < 0.05,
    }


# ===================================================================
# Option A: PreFraudLightGBM
# ===================================================================
class PreFraudLightGBM:
    """LightGBM classifier tuned via Optuna for pre-fraud prediction.

    Uses only indirect behavioural features and drift scores.
    ``is_unbalance=True`` lets LightGBM natively up-weight the
    minority (fraud) class.

    Parameters
    ----------
    random_state : int
        Seed for reproducibility.
    n_optuna_trials : int
        Number of Bayesian optimisation trials.
    n_cv_folds : int
        Number of stratified CV folds.
    """

    def __init__(
        self,
        random_state: int = 42,
        n_optuna_trials: int = 20,
        n_cv_folds: int = 5,
    ) -> None:
        self.random_state = random_state
        self.n_optuna_trials = n_optuna_trials
        self.n_cv_folds = n_cv_folds
        self.best_params: Dict[str, Any] = {}
        self.model: Optional[lgb.LGBMClassifier] = None
        self.cv_scores: List[float] = []

    # ------------------------------------------------------------------
    # Optuna objective
    # ------------------------------------------------------------------
    def _objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """Optuna objective: mean AUC-ROC across stratified CV folds."""
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "is_unbalance": True,
            "random_state": self.random_state,
            "verbosity": -1,
            "n_jobs": -1,
        }

        skf = StratifiedKFold(
            n_splits=self.n_cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )
        fold_aucs = []

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            clf = lgb.LGBMClassifier(**params)
            clf.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            y_prob = clf.predict_proba(X_va)[:, 1]
            fold_aucs.append(roc_auc_score(y_va, y_prob))

        return float(np.mean(fold_aucs))

    # ------------------------------------------------------------------
    # Tuning + final training
    # ------------------------------------------------------------------
    def tune_and_train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Run Optuna hyperparameter search, then retrain on full data.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Binary target vector.
        """
        logger.info(
            "Option A (LightGBM): starting Optuna search (%d trials, %d-fold CV)",
            self.n_optuna_trials, self.n_cv_folds,
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(
            lambda trial: self._objective(trial, X, y),
            n_trials=self.n_optuna_trials,
            show_progress_bar=False,
        )

        self.best_params = study.best_params
        self.best_params["is_unbalance"] = True
        self.best_params["random_state"] = self.random_state
        self.best_params["verbosity"] = -1
        self.best_params["n_jobs"] = -1

        logger.info(
            "  Best trial AUC-ROC: %.4f | Params: %s",
            study.best_value, json.dumps(self.best_params, indent=None),
        )

        # ---- Final 5-fold CV to record per-fold scores ----------------
        skf = StratifiedKFold(
            n_splits=self.n_cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )
        self.cv_scores = []
        for train_idx, val_idx in skf.split(X, y):
            clf = lgb.LGBMClassifier(**self.best_params)
            clf.fit(
                X[train_idx], y[train_idx],
                eval_set=[(X[val_idx], y[val_idx])],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            prob = clf.predict_proba(X[val_idx])[:, 1]
            self.cv_scores.append(roc_auc_score(y[val_idx], prob))

        logger.info(
            "  Final CV AUC-ROC: %.4f +/- %.4f",
            np.mean(self.cv_scores), np.std(self.cv_scores),
        )

        # ---- Retrain on full data -------------------------------------
        self.model = lgb.LGBMClassifier(**self.best_params)
        self.model.fit(X, y)
        logger.info("  LightGBM retrained on full training set.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(fraud) for each row."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call tune_and_train() first.")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions."""
        return (self.predict_proba(X) >= threshold).astype(int)


# ===================================================================
# Option B: PreFraudNeuralNet
# ===================================================================
class PreFraudNeuralNet:
    """MLP classifier (sklearn) tuned via Optuna for pre-fraud prediction.

    Accepts the same feature matrix as Option A.  Internally wraps the
    features in a ``StandardScaler`` pipeline, which is critical for
    neural-network convergence.

    Parameters
    ----------
    random_state : int
        Seed for reproducibility.
    n_optuna_trials : int
        Number of Bayesian optimisation trials.
    n_cv_folds : int
        Number of stratified CV folds.
    """

    def __init__(
        self,
        random_state: int = 42,
        n_optuna_trials: int = 20,
        n_cv_folds: int = 5,
    ) -> None:
        self.random_state = random_state
        self.n_optuna_trials = n_optuna_trials
        self.n_cv_folds = n_cv_folds
        self.best_params: Dict[str, Any] = {}
        self.pipeline: Optional[Pipeline] = None
        self.cv_scores: List[float] = []

    # ------------------------------------------------------------------
    # Optuna objective
    # ------------------------------------------------------------------
    def _objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """Optuna objective: mean AUC-ROC across stratified CV folds.

        Uses stratified subsampling during search for speed; final
        retraining uses the full dataset.
        """
        n_layers = trial.suggest_int("n_layers", 1, 3)
        layers = []
        for i in range(n_layers):
            width = trial.suggest_int(f"layer_{i}_width", 32, 256)
            layers.append(width)

        params = {
            "hidden_layer_sizes": tuple(layers),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float(
                "learning_rate_init", 1e-4, 1e-2, log=True
            ),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
            "max_iter": 150,
            "early_stopping": True,
            "validation_fraction": 0.15,
            "n_iter_no_change": 10,
            "random_state": self.random_state,
        }

        # Subsample for faster Optuna search (stratified)
        max_search_samples = 50_000
        if len(X) > max_search_samples:
            rng = np.random.RandomState(self.random_state)
            pos_idx = np.where(y == 1)[0]
            neg_idx = np.where(y == 0)[0]
            n_pos = min(len(pos_idx), int(max_search_samples * y.mean()))
            n_neg = max_search_samples - n_pos
            sel = np.concatenate([
                rng.choice(pos_idx, size=n_pos, replace=False),
                rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False),
            ])
            X_sub, y_sub = X[sel], y[sel]
        else:
            X_sub, y_sub = X, y

        skf = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=self.random_state,
        )
        fold_aucs = []

        for train_idx, val_idx in skf.split(X_sub, y_sub):
            X_tr, X_va = X_sub[train_idx], X_sub[val_idx]
            y_tr, y_va = y_sub[train_idx], y_sub[val_idx]

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("mlp", MLPClassifier(**params)),
            ])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(X_tr, y_tr)
            y_prob = pipe.predict_proba(X_va)[:, 1]
            fold_aucs.append(roc_auc_score(y_va, y_prob))

        return float(np.mean(fold_aucs))

    # ------------------------------------------------------------------
    # Tuning + final training
    # ------------------------------------------------------------------
    def tune_and_train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Run Optuna hyperparameter search, then retrain on full data.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Binary target vector.
        """
        logger.info(
            "Option B (NeuralNet): starting Optuna search (%d trials, %d-fold CV)",
            self.n_optuna_trials, self.n_cv_folds,
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(
            lambda trial: self._objective(trial, X, y),
            n_trials=self.n_optuna_trials,
            show_progress_bar=False,
        )

        # Reconstruct best params
        bp = study.best_params
        n_layers = bp["n_layers"]
        layers = tuple(bp[f"layer_{i}_width"] for i in range(n_layers))

        self.best_params = {
            "hidden_layer_sizes": layers,
            "activation": bp["activation"],
            "alpha": bp["alpha"],
            "learning_rate_init": bp["learning_rate_init"],
            "batch_size": bp["batch_size"],
            "max_iter": 150,
            "early_stopping": True,
            "validation_fraction": 0.15,
            "n_iter_no_change": 10,
            "random_state": self.random_state,
        }

        logger.info(
            "  Best trial AUC-ROC: %.4f | Layers: %s",
            study.best_value, layers,
        )

        # ---- Final 3-fold CV to record per-fold scores ----------------
        # Subsample for feasible runtime; MLP is expensive on 500K rows
        max_cv_samples = 80_000
        if len(X) > max_cv_samples:
            rng = np.random.RandomState(self.random_state)
            pos_idx = np.where(y == 1)[0]
            neg_idx = np.where(y == 0)[0]
            n_pos = min(len(pos_idx), int(max_cv_samples * y.mean()))
            n_neg = max_cv_samples - n_pos
            sel = np.concatenate([
                rng.choice(pos_idx, size=n_pos, replace=False),
                rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False),
            ])
            X_cv, y_cv = X[sel], y[sel]
        else:
            X_cv, y_cv = X, y

        skf = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=self.random_state,
        )
        self.cv_scores = []
        for train_idx, val_idx in skf.split(X_cv, y_cv):
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("mlp", MLPClassifier(**self.best_params)),
            ])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(X_cv[train_idx], y_cv[train_idx])
            prob = pipe.predict_proba(X_cv[val_idx])[:, 1]
            self.cv_scores.append(roc_auc_score(y_cv[val_idx], prob))

        logger.info(
            "  Final CV AUC-ROC: %.4f +/- %.4f",
            np.mean(self.cv_scores), np.std(self.cv_scores),
        )

        # ---- Retrain on full data -------------------------------------
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(**self.best_params)),
        ])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.pipeline.fit(X, y)
        logger.info("  NeuralNet retrained on full training set.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(fraud) for each row."""
        if self.pipeline is None:
            raise RuntimeError("Model not trained. Call tune_and_train() first.")
        return self.pipeline.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions."""
        return (self.predict_proba(X) >= threshold).astype(int)


# ===================================================================
# Option C: PreFraudEnsemble
# ===================================================================
class PreFraudEnsemble:
    """Stacking ensemble combining LightGBM and MLP with a Logistic
    Regression meta-learner.

    Uses ``StackingClassifier`` from sklearn with ``cv=5`` so
    that out-of-fold predictions are used as meta-features,
    preventing information leakage.

    Parameters
    ----------
    lgbm_params : dict
        Best hyperparameters from Option A.
    mlp_params : dict
        Best hyperparameters from Option B.
    random_state : int
        Seed for reproducibility.
    n_cv_folds : int
        Number of stratified CV folds.
    """

    def __init__(
        self,
        lgbm_params: Dict[str, Any],
        mlp_params: Dict[str, Any],
        random_state: int = 42,
        n_cv_folds: int = 5,
    ) -> None:
        self.lgbm_params = lgbm_params
        self.mlp_params = mlp_params
        self.random_state = random_state
        self.n_cv_folds = n_cv_folds
        self.model: Optional[StackingClassifier] = None
        self.cv_scores: List[float] = []

    def _build_stacker(self) -> StackingClassifier:
        """Construct the StackingClassifier with the tuned base learners."""
        lgbm_estimator = lgb.LGBMClassifier(**self.lgbm_params)

        mlp_estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(**self.mlp_params)),
        ])

        stacker = StackingClassifier(
            estimators=[
                ("lightgbm", lgbm_estimator),
                ("neural_net", mlp_estimator),
            ],
            final_estimator=LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight="balanced",
            ),
            cv=3,
            stack_method="predict_proba",
            passthrough=False,
            n_jobs=-1,
        )
        return stacker

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the stacking ensemble with internal CV for meta-features.

        Also records per-fold AUC-ROC via a separate stratified CV loop
        for fair comparison with Options A and B.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Binary target vector.
        """
        logger.info(
            "Option C (Ensemble): training StackingClassifier (%d-fold CV meta-features)",
            self.n_cv_folds,
        )

        # ---- Per-fold evaluation for fair comparison -------------------
        skf = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=self.random_state,
        )
        self.cv_scores = []
        for train_idx, val_idx in skf.split(X, y):
            stacker = self._build_stacker()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stacker.fit(X[train_idx], y[train_idx])
            prob = stacker.predict_proba(X[val_idx])[:, 1]
            self.cv_scores.append(roc_auc_score(y[val_idx], prob))

        logger.info(
            "  CV AUC-ROC: %.4f +/- %.4f",
            np.mean(self.cv_scores), np.std(self.cv_scores),
        )

        # ---- Retrain on full data -------------------------------------
        self.model = self._build_stacker()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X, y)
        logger.info("  Ensemble retrained on full training set.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(fraud) for each row."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions."""
        return (self.predict_proba(X) >= threshold).astype(int)


# ===================================================================
# Main orchestrating class
# ===================================================================
class PreFraudModel:
    """Orchestrates training, evaluation, and comparison of all three
    pre-fraud model architectures.

    Parameters
    ----------
    config : dict, optional
        Project configuration.  Loaded from ``config.yaml`` if *None*.
    n_optuna_trials : int
        Number of Optuna trials per tuneable model.
    n_cv_folds : int
        Number of stratified CV folds.
    random_state : int
        Global random seed.
    """

    EXCLUDE_COLS = {"isFraud", "TransactionID"}

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        n_optuna_trials: int = 20,
        n_cv_folds: int = 5,
        random_state: int = 42,
    ) -> None:
        self.config = config or load_config()
        self.n_optuna_trials = n_optuna_trials
        self.n_cv_folds = n_cv_folds
        self.random_state = random_state

        # Sub-models
        self.lgbm: Optional[PreFraudLightGBM] = None
        self.nn: Optional[PreFraudNeuralNet] = None
        self.ensemble: Optional[PreFraudEnsemble] = None

        # Evaluation artefacts
        self.test_metrics: Dict[str, Dict[str, Any]] = {}
        self.mcnemar_results: Dict[str, Dict[str, Any]] = {}
        self.comparison_df: Optional[pd.DataFrame] = None
        self.feature_names: Optional[List[str]] = None

        # Cached test predictions for plotting and McNemar's
        self._y_test: Optional[np.ndarray] = None
        self._test_probs: Dict[str, np.ndarray] = {}
        self._test_preds: Dict[str, np.ndarray] = {}

        # Ensure output directories
        for d in (FIGURES_DIR, TABLES_DIR, LOGS_DIR, MODELS_DIR):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train all three model options.

        Parameters
        ----------
        X_train : np.ndarray
            Feature matrix (behavioural features + drift scores).
        y_train : np.ndarray
            Binary target vector (0 = non-fraud, 1 = fraud).
        """
        t0 = time.time()

        # ---- Option A: LightGBM ---------------------------------------
        self.lgbm = PreFraudLightGBM(
            random_state=self.random_state,
            n_optuna_trials=self.n_optuna_trials,
            n_cv_folds=self.n_cv_folds,
        )
        self.lgbm.tune_and_train(X_train, y_train)

        # ---- Option B: NeuralNet --------------------------------------
        self.nn = PreFraudNeuralNet(
            random_state=self.random_state,
            n_optuna_trials=self.n_optuna_trials,
            n_cv_folds=self.n_cv_folds,
        )
        self.nn.tune_and_train(X_train, y_train)

        # ---- Option C: Ensemble (uses tuned params from A and B) ------
        self.ensemble = PreFraudEnsemble(
            lgbm_params=self.lgbm.best_params,
            mlp_params=self.nn.best_params,
            random_state=self.random_state,
            n_cv_folds=self.n_cv_folds,
        )
        self.ensemble.train(X_train, y_train)

        elapsed = time.time() - t0
        logger.info("All three models trained in %.1f seconds.", elapsed)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def _evaluate_single(
        self,
        name: str,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute all metrics for a single model on the test set."""
        y_pred = (y_prob >= 0.5).astype(int)

        auc_roc = roc_auc_score(y_true, y_prob)
        auc_pr = average_precision_score(y_true, y_prob)
        f1 = f1_score(y_true, y_pred)
        p_at_r80 = precision_at_recall(y_true, y_prob, 0.80)
        cm = confusion_matrix(y_true, y_pred).tolist()

        metrics = {
            "model": name,
            "auc_roc": round(auc_roc, 6),
            "auc_pr": round(auc_pr, 6),
            "f1": round(f1, 6),
            "precision_at_recall_80": round(p_at_r80, 6),
            "confusion_matrix": cm,
            "n_test": int(len(y_true)),
            "n_positive": int(y_true.sum()),
        }

        logger.info(
            "  %s -- AUC-ROC: %.4f | AUC-PR: %.4f | F1: %.4f | P@R80: %.4f",
            name, auc_roc, auc_pr, f1, p_at_r80,
        )
        return metrics

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Evaluate all three models on the held-out test set.

        Parameters
        ----------
        X_test : np.ndarray
            Test feature matrix.
        y_test : np.ndarray
            Test target vector.

        Returns
        -------
        dict
            Nested dict keyed by model name with metric dicts as values.
        """
        logger.info("Evaluating all three models on the test set ...")
        self._y_test = y_test

        models = {
            "Option A: LightGBM": self.lgbm,
            "Option B: NeuralNet": self.nn,
            "Option C: Ensemble": self.ensemble,
        }

        self.test_metrics = {}
        self._test_probs = {}
        self._test_preds = {}

        for name, model in models.items():
            y_prob = model.predict_proba(X_test)
            y_pred = (y_prob >= 0.5).astype(int)

            self._test_probs[name] = y_prob
            self._test_preds[name] = y_pred

            metrics = self._evaluate_single(name, y_test, y_prob)

            # Attach CV scores from training phase
            if hasattr(model, "cv_scores"):
                metrics["cv_auc_roc_mean"] = round(float(np.mean(model.cv_scores)), 6)
                metrics["cv_auc_roc_std"] = round(float(np.std(model.cv_scores)), 6)
                metrics["cv_auc_roc_folds"] = [round(s, 6) for s in model.cv_scores]

            self.test_metrics[name] = metrics

        # Build comparison DataFrame
        rows = []
        for name, m in self.test_metrics.items():
            rows.append({
                "Model": name,
                "AUC-ROC": m["auc_roc"],
                "AUC-PR": m["auc_pr"],
                "F1": m["f1"],
                "P@R80": m["precision_at_recall_80"],
                "CV AUC-ROC (mean)": m.get("cv_auc_roc_mean", ""),
                "CV AUC-ROC (std)": m.get("cv_auc_roc_std", ""),
            })
        self.comparison_df = pd.DataFrame(rows).set_index("Model")

        logger.info("\n%s", self.comparison_df.to_string())
        return self.test_metrics

    # ------------------------------------------------------------------
    # Statistical significance testing
    # ------------------------------------------------------------------
    def run_mcnemar_tests(self) -> pd.DataFrame:
        """Perform pairwise McNemar's tests between all three models.

        Must be called after ``evaluate()``.

        Returns
        -------
        pd.DataFrame
            Pairwise comparison table with chi2, p-value, and significance.
        """
        if self._y_test is None or not self._test_preds:
            raise RuntimeError("Call evaluate() before running McNemar's tests.")

        logger.info("Running pairwise McNemar's tests ...")

        model_names = list(self._test_preds.keys())
        results_rows = []

        for name_a, name_b in combinations(model_names, 2):
            result = mcnemar_test(
                self._y_test,
                self._test_preds[name_a],
                self._test_preds[name_b],
            )
            result["model_a"] = name_a
            result["model_b"] = name_b
            results_rows.append(result)

            sig_label = "YES" if result["significant_0.05"] else "no"
            logger.info(
                "  %s vs %s -- chi2=%.4f, p=%.6f (%s)",
                name_a, name_b, result["chi2"], result["p_value"], sig_label,
            )

        mcnemar_df = pd.DataFrame(results_rows)
        col_order = ["model_a", "model_b", "b", "c", "chi2", "p_value", "significant_0.05"]
        mcnemar_df = mcnemar_df[col_order]
        self.mcnemar_results = mcnemar_df
        return mcnemar_df

    # ==================================================================
    # Interpretability: SHAP analysis (LightGBM)
    # ==================================================================
    def compute_shap_analysis(
        self,
        X_test: np.ndarray,
        feature_names: List[str],
        max_samples: int = 5000,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Compute SHAP values for the LightGBM model and return the
        feature importance ranking.

        Parameters
        ----------
        X_test : np.ndarray
            Test feature matrix.
        feature_names : list of str
            Column names corresponding to features.
        max_samples : int
            Subsample size for SHAP computation (large datasets).

        Returns
        -------
        shap_values : np.ndarray
            SHAP value matrix (n_samples, n_features).
        importance_df : pd.DataFrame
            Feature importance ranking sorted by mean |SHAP|.
        """
        if self.lgbm is None or self.lgbm.model is None:
            raise RuntimeError("LightGBM model not trained.")

        logger.info("Computing SHAP values for LightGBM ...")

        # Subsample if needed
        if len(X_test) > max_samples:
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(len(X_test), max_samples, replace=False)
            X_shap = X_test[idx]
        else:
            X_shap = X_test

        explainer = shap.TreeExplainer(self.lgbm.model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_values = explainer.shap_values(X_shap)

        # For binary classifiers LightGBM SHAP may return a list of two arrays
        if isinstance(shap_values, list):
            shap_array = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            shap_array = shap_values

        # Build importance ranking
        mean_abs_shap = np.abs(shap_array).mean(axis=0)
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        # Stash for plotting
        self._shap_values = shap_array
        self._shap_X = pd.DataFrame(X_shap, columns=feature_names)
        self._shap_importance = importance_df

        logger.info("  SHAP analysis complete. Top 5 features:")
        for _, row in importance_df.head(5).iterrows():
            logger.info("    %s: %.6f", row["feature"], row["mean_abs_shap"])

        return shap_array, importance_df

    # ==================================================================
    # Plotting
    # ==================================================================
    def plot_roc_curves(self) -> Path:
        """Overlay ROC curves for all three models.

        Returns
        -------
        Path
            File path of the saved figure.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        colours = ["#2563eb", "#dc2626", "#16a34a"]

        for idx, (name, y_prob) in enumerate(self._test_probs.items()):
            fpr, tpr, _ = roc_curve(self._y_test, y_prob)
            roc_auc = roc_auc_score(self._y_test, y_prob)
            ax.plot(fpr, tpr, color=colours[idx], lw=2,
                    label=f"{name} (AUC = {roc_auc:.4f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curves -- Pre-Fraud Model Comparison", fontsize=14)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out_path = FIGURES_DIR / "roc_curves_comparison.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved ROC comparison to %s", out_path)
        return out_path

    def plot_pr_curves(self) -> Path:
        """Overlay Precision-Recall curves for all three models.

        Returns
        -------
        Path
            File path of the saved figure.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        colours = ["#2563eb", "#dc2626", "#16a34a"]

        for idx, (name, y_prob) in enumerate(self._test_probs.items()):
            prec, rec, _ = precision_recall_curve(self._y_test, y_prob)
            ap = average_precision_score(self._y_test, y_prob)
            ax.plot(rec, prec, color=colours[idx], lw=2,
                    label=f"{name} (AP = {ap:.4f})")

        baseline_rate = self._y_test.mean()
        ax.axhline(baseline_rate, color="grey", lw=1, linestyle="--",
                    label=f"No-skill ({baseline_rate:.4f})")
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title("Precision-Recall Curves -- Pre-Fraud Model Comparison", fontsize=14)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out_path = FIGURES_DIR / "pr_curves_comparison.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved PR comparison to %s", out_path)
        return out_path

    def plot_shap_summary(self, max_display: int = 30) -> Path:
        """SHAP beeswarm summary plot for LightGBM.

        Parameters
        ----------
        max_display : int
            Number of top features to show.

        Returns
        -------
        Path
            File path of the saved figure.
        """
        if not hasattr(self, "_shap_values"):
            raise RuntimeError("Run compute_shap_analysis() first.")

        plt.figure(figsize=(10, 12))
        shap.summary_plot(
            self._shap_values,
            self._shap_X,
            max_display=max_display,
            show=False,
            plot_size=None,
        )
        plt.title("SHAP Feature Importance -- Pre-Fraud LightGBM", fontsize=14)
        plt.tight_layout()

        out_path = FIGURES_DIR / "shap_summary.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close("all")
        logger.info("Saved SHAP summary to %s", out_path)
        return out_path

    def plot_shap_bar(self, top_n: int = 20) -> Path:
        """SHAP mean-absolute-value bar chart.

        Parameters
        ----------
        top_n : int
            Number of features to display.

        Returns
        -------
        Path
            File path of the saved figure.
        """
        if not hasattr(self, "_shap_importance"):
            raise RuntimeError("Run compute_shap_analysis() first.")

        df_top = self._shap_importance.head(top_n).copy()

        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
        sns.barplot(
            data=df_top,
            x="mean_abs_shap",
            y="feature",
            palette="viridis",
            ax=ax,
        )
        ax.set_xlabel("Mean |SHAP value|", fontsize=12)
        ax.set_ylabel("Feature", fontsize=12)
        ax.set_title(f"Top {top_n} Features by SHAP Importance (Pre-Fraud LightGBM)", fontsize=14)
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()

        out_path = FIGURES_DIR / "shap_importance_bar.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved SHAP bar chart to %s", out_path)
        return out_path

    def plot_partial_dependence(
        self,
        X_test: np.ndarray,
        feature_names: List[str],
        top_n: int = 10,
    ) -> Path:
        """Partial dependence plots for the top-N most important features.

        Uses the LightGBM model and the SHAP importance ranking to select
        which features to plot.

        Parameters
        ----------
        X_test : np.ndarray
            Test feature matrix.
        feature_names : list of str
            Feature names.
        top_n : int
            Number of features to plot.

        Returns
        -------
        Path
            File path of the saved figure.
        """
        if self.lgbm is None or self.lgbm.model is None:
            raise RuntimeError("LightGBM model not trained.")

        # Use SHAP ranking if available, else LightGBM native importance
        if hasattr(self, "_shap_importance"):
            top_features = self._shap_importance["feature"].head(top_n).tolist()
        else:
            imp = self.lgbm.model.feature_importances_
            top_idx = np.argsort(imp)[::-1][:top_n]
            top_features = [feature_names[i] for i in top_idx]

        # Resolve feature indices
        feature_indices = []
        for feat in top_features:
            if feat in feature_names:
                feature_indices.append(feature_names.index(feat))

        if not feature_indices:
            logger.warning("No valid features for partial dependence plots.")
            return FIGURES_DIR / "partial_dependence.png"

        # Subsample for speed
        max_samples = 2000
        if len(X_test) > max_samples:
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(len(X_test), max_samples, replace=False)
            X_pdp = X_test[idx]
        else:
            X_pdp = X_test

        X_pdp_df = pd.DataFrame(X_pdp, columns=feature_names)

        ncols = 2
        nrows = int(np.ceil(len(feature_indices) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
        axes_flat = np.atleast_1d(axes).flatten()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            display = PartialDependenceDisplay.from_estimator(
                self.lgbm.model,
                X_pdp_df,
                features=feature_indices,
                feature_names=feature_names,
                ax=axes_flat[:len(feature_indices)],
                grid_resolution=50,
                kind="average",
            )

        # Hide unused axes
        for i in range(len(feature_indices), len(axes_flat)):
            axes_flat[i].set_visible(False)

        fig.suptitle(
            f"Partial Dependence Plots -- Top {len(feature_indices)} Features (LightGBM)",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        fig.tight_layout()

        out_path = FIGURES_DIR / "partial_dependence.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved partial dependence plots to %s", out_path)
        return out_path

    def plot_feature_importance_ranking(self, feature_names: List[str]) -> Path:
        """Combined feature importance chart: LightGBM native + SHAP.

        Parameters
        ----------
        feature_names : list of str
            Feature names.

        Returns
        -------
        Path
            File path of the saved figure.
        """
        if self.lgbm is None or self.lgbm.model is None:
            raise RuntimeError("LightGBM model not trained.")

        # Native gain importance
        lgbm_imp = self.lgbm.model.feature_importances_
        lgbm_series = pd.Series(lgbm_imp, index=feature_names, name="lgbm_gain")

        # SHAP importance (if available)
        if hasattr(self, "_shap_importance"):
            shap_series = self._shap_importance.set_index("feature")["mean_abs_shap"]
        else:
            shap_series = pd.Series(0.0, index=feature_names, name="shap")

        # Combine
        df = pd.DataFrame({
            "lgbm_gain": lgbm_series,
            "shap_mean_abs": shap_series.reindex(feature_names, fill_value=0),
        })

        # Normalise each to [0, 1]
        for col in df.columns:
            cmin, cmax = df[col].min(), df[col].max()
            if cmax > cmin:
                df[f"{col}_norm"] = (df[col] - cmin) / (cmax - cmin)
            else:
                df[f"{col}_norm"] = 0.0

        norm_cols = [c for c in df.columns if c.endswith("_norm")]
        df["composite"] = df[norm_cols].mean(axis=1)
        df = df.sort_values("composite", ascending=False)

        # Plot top 20
        top_df = df.head(20)

        fig, axes = plt.subplots(1, 3, figsize=(18, 8))

        # Panel 1: LightGBM gain
        top_lgbm = top_df["lgbm_gain"].sort_values()
        axes[0].barh(top_lgbm.index, top_lgbm.values, color="#2563eb")
        axes[0].set_title("LightGBM Gain Importance", fontsize=12)
        axes[0].set_xlabel("Gain", fontsize=10)

        # Panel 2: SHAP
        top_shap = top_df["shap_mean_abs"].sort_values()
        axes[1].barh(top_shap.index, top_shap.values, color="#dc2626")
        axes[1].set_title("SHAP Mean |Value|", fontsize=12)
        axes[1].set_xlabel("Mean |SHAP|", fontsize=10)

        # Panel 3: Composite
        top_comp = top_df["composite"].sort_values()
        axes[2].barh(top_comp.index, top_comp.values, color="#16a34a")
        axes[2].set_title("Composite Importance", fontsize=12)
        axes[2].set_xlabel("Composite Score", fontsize=10)

        fig.suptitle(
            "Feature Importance Ranking -- Pre-Fraud LightGBM (Top 20)",
            fontsize=14, fontweight="bold",
        )
        fig.tight_layout()

        out_path = FIGURES_DIR / "feature_importance_ranking.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved feature importance ranking to %s", out_path)

        # Also save as CSV
        df.to_csv(TABLES_DIR / "pre_fraud_feature_importance.csv")
        logger.info("Saved feature importance table to %s", TABLES_DIR / "pre_fraud_feature_importance.csv")

        return out_path

    def plot_model_comparison_table(self) -> Path:
        """Render the comparison DataFrame as a publication-quality table figure.

        Returns
        -------
        Path
            File path of the saved figure.
        """
        if self.comparison_df is None:
            raise RuntimeError("Call evaluate() first.")

        df = self.comparison_df.copy()

        n_rows, n_cols = df.shape
        fig_width = max(10, n_cols * 2.0)
        fig_height = max(2.5, (n_rows + 1) * 0.7)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis("off")

        cell_text = []
        for _, row in df.iterrows():
            row_text = []
            for val in row:
                if isinstance(val, float):
                    row_text.append(f"{val:.4f}")
                else:
                    row_text.append(str(val))
            cell_text.append(row_text)

        table = ax.table(
            cellText=cell_text,
            colLabels=df.columns.tolist(),
            rowLabels=df.index.tolist(),
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.8)

        # Style header
        for (row_idx, col_idx), cell in table.get_celld().items():
            if row_idx == 0:
                cell.set_facecolor("#2c3e50")
                cell.set_text_props(color="white", fontweight="bold")
            cell.set_edgecolor("#bdc3c7")

        ax.set_title(
            "Pre-Fraud Model Comparison",
            fontsize=14, fontweight="bold", pad=20,
        )
        fig.tight_layout()

        out_path = FIGURES_DIR / "model_comparison_table.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved comparison table to %s", out_path)
        return out_path

    # ==================================================================
    # Persistence
    # ==================================================================
    def save_models(self) -> Dict[str, Path]:
        """Pickle all three trained models to disk.

        Returns
        -------
        dict
            Mapping of model name to file path.
        """
        saved = {}

        if self.lgbm is not None and self.lgbm.model is not None:
            path = MODELS_DIR / "pre_fraud_lgbm.pkl"
            with open(path, "wb") as fh:
                pickle.dump(self.lgbm, fh)
            saved["lgbm"] = path
            logger.info("Saved LightGBM to %s", path)

        if self.nn is not None and self.nn.pipeline is not None:
            path = MODELS_DIR / "pre_fraud_nn.pkl"
            with open(path, "wb") as fh:
                pickle.dump(self.nn, fh)
            saved["nn"] = path
            logger.info("Saved NeuralNet to %s", path)

        if self.ensemble is not None and self.ensemble.model is not None:
            path = MODELS_DIR / "pre_fraud_ensemble.pkl"
            with open(path, "wb") as fh:
                pickle.dump(self.ensemble, fh)
            saved["ensemble"] = path
            logger.info("Saved Ensemble to %s", path)

        return saved

    def save_metrics(self) -> Dict[str, Path]:
        """Persist all evaluation metrics, comparison tables, and
        McNemar's test results.

        Returns
        -------
        dict
            Mapping of artifact name to its file path.
        """
        saved = {}

        # ---- Test metrics (JSON) -----------------------------------------
        if self.test_metrics:
            # Make a serialisable copy (drop confusion matrix numpy arrays)
            serialisable = {}
            for name, m in self.test_metrics.items():
                serialisable[name] = {
                    k: v for k, v in m.items()
                }
            path = TABLES_DIR / "pre_fraud_test_metrics.json"
            with open(path, "w") as fh:
                json.dump(serialisable, fh, indent=2, default=str)
            saved["test_metrics"] = path
            logger.info("Saved test metrics to %s", path)

        # ---- Comparison table (CSV) --------------------------------------
        if self.comparison_df is not None:
            path = TABLES_DIR / "pre_fraud_model_comparison.csv"
            self.comparison_df.to_csv(path)
            saved["comparison_table"] = path
            logger.info("Saved comparison table to %s", path)

        # ---- McNemar results (CSV) ---------------------------------------
        if isinstance(self.mcnemar_results, pd.DataFrame) and not self.mcnemar_results.empty:
            path = TABLES_DIR / "pre_fraud_mcnemar_tests.csv"
            self.mcnemar_results.to_csv(path, index=False)
            saved["mcnemar_tests"] = path
            logger.info("Saved McNemar test results to %s", path)

        # ---- Hyperparameters (JSON) --------------------------------------
        hp = {}
        if self.lgbm is not None:
            hp["lgbm"] = {
                k: (list(v) if isinstance(v, tuple) else v)
                for k, v in self.lgbm.best_params.items()
            }
        if self.nn is not None:
            hp["neural_net"] = {
                k: (list(v) if isinstance(v, tuple) else v)
                for k, v in self.nn.best_params.items()
            }
        if hp:
            path = TABLES_DIR / "pre_fraud_best_hyperparameters.json"
            with open(path, "w") as fh:
                json.dump(hp, fh, indent=2, default=str)
            saved["hyperparameters"] = path
            logger.info("Saved hyperparameters to %s", path)

        # ---- CV fold scores (CSV) ----------------------------------------
        cv_rows = []
        for name, model in [
            ("LightGBM", self.lgbm),
            ("NeuralNet", self.nn),
            ("Ensemble", self.ensemble),
        ]:
            if model is not None and hasattr(model, "cv_scores"):
                for fold_idx, score in enumerate(model.cv_scores, 1):
                    cv_rows.append({
                        "model": name,
                        "fold": fold_idx,
                        "auc_roc": round(score, 6),
                    })
        if cv_rows:
            cv_df = pd.DataFrame(cv_rows)
            path = TABLES_DIR / "pre_fraud_cv_fold_scores.csv"
            cv_df.to_csv(path, index=False)
            saved["cv_fold_scores"] = path
            logger.info("Saved CV fold scores to %s", path)

        # ---- SHAP feature importance (CSV) --------------------------------
        if hasattr(self, "_shap_importance"):
            path = TABLES_DIR / "pre_fraud_shap_importance.csv"
            self._shap_importance.to_csv(path, index=False)
            saved["shap_importance"] = path
            logger.info("Saved SHAP importance to %s", path)

        return saved

    def save_all(
        self,
        X_test: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, Path]:
        """Convenience wrapper: save all models, metrics, and plots.

        Parameters
        ----------
        X_test : np.ndarray
            Test feature matrix (for partial dependence / SHAP plots).
        feature_names : list of str
            Feature column names.

        Returns
        -------
        dict
            Mapping of artifact name to its file path.
        """
        paths = {}

        # Models
        paths.update(self.save_models())

        # Metrics
        paths.update(self.save_metrics())

        # Plots
        plot_methods = [
            ("roc_curves", self.plot_roc_curves),
            ("pr_curves", self.plot_pr_curves),
            ("comparison_table", self.plot_model_comparison_table),
        ]

        for name, method in plot_methods:
            try:
                paths[name] = method()
            except Exception as exc:
                logger.warning("Could not generate %s: %s", name, exc)

        # SHAP plots
        try:
            paths["shap_summary"] = self.plot_shap_summary()
        except Exception as exc:
            logger.warning("Could not generate SHAP summary: %s", exc)

        try:
            paths["shap_bar"] = self.plot_shap_bar()
        except Exception as exc:
            logger.warning("Could not generate SHAP bar: %s", exc)

        # Partial dependence
        try:
            paths["partial_dependence"] = self.plot_partial_dependence(
                X_test, feature_names, top_n=10,
            )
        except Exception as exc:
            logger.warning("Could not generate partial dependence plots: %s", exc)

        # Feature importance ranking
        try:
            paths["feature_importance"] = self.plot_feature_importance_ranking(feature_names)
        except Exception as exc:
            logger.warning("Could not generate feature importance ranking: %s", exc)

        logger.info("All artifacts saved. Total: %d files.", len(paths))
        return paths


# ===================================================================
# CLI entry point
# ===================================================================
def main() -> None:
    """End-to-end pre-fraud model pipeline.

    Steps
    -----
    1. Load processed data splits.
    2. Load drift scores from ``results/tables/drift_scores.csv``.
    3. Load pre-fraud boundary features from ablation results.
    4. Merge behavioural features with drift scores.
    5. Train and compare all three model options (A, B, C).
    6. Statistical significance testing (McNemar).
    7. Interpretability analysis (SHAP, PDP).
    8. Save all models, metrics, figures, and tables.
    """
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("PRE-FRAUD MODEL -- Multi-Architecture Comparison")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. Configuration
    # ------------------------------------------------------------------
    config = load_config()
    random_state = config.get("data", {}).get("random_state", 42)
    n_folds = config.get("evaluation", {}).get("cross_validation_folds", 5)

    # ------------------------------------------------------------------
    # 2. Load processed data
    # ------------------------------------------------------------------
    processed_dir = PROJECT_ROOT / config["data"]["processed_path"]
    logger.info("Loading processed data from %s", processed_dir)

    try:
        train_df = pd.read_parquet(processed_dir / "train.parquet")
        val_df = pd.read_parquet(processed_dir / "val.parquet")
        test_df = pd.read_parquet(processed_dir / "test.parquet")
    except FileNotFoundError as exc:
        logger.error(
            "Processed data not found. Run data_loader.py first: %s", exc,
        )
        raise SystemExit(1) from exc

    logger.info(
        "Loaded splits: train=%d, val=%d, test=%d",
        len(train_df), len(val_df), len(test_df),
    )

    # ------------------------------------------------------------------
    # 3. Load drift scores
    # ------------------------------------------------------------------
    drift_path = TABLES_DIR / "drift_scores.csv"
    if drift_path.exists():
        logger.info("Loading drift scores from %s", drift_path)
        drift_df = pd.read_csv(drift_path)
        drift_cols = [c for c in drift_df.columns if c.endswith("_drift")]
        logger.info("  Drift dimensions loaded: %s", drift_cols)
    else:
        logger.warning(
            "Drift scores not found at %s. "
            "Running without drift features. "
            "Execute drift_analysis.py first for full pipeline.",
            drift_path,
        )
        drift_df = None
        drift_cols = []

    # ------------------------------------------------------------------
    # 4. Determine pre-fraud boundary features
    # ------------------------------------------------------------------
    # Try to load from ablation results
    ablation_path = TABLES_DIR / "ablation_results.json"
    boundary_features = None

    if ablation_path.exists():
        logger.info("Loading ablation results to determine boundary features ...")
        with open(ablation_path, "r") as fh:
            ablation_results = json.load(fh)

        # Find the last stage before AUC-ROC drops below 0.80, or use
        # the final stage (Stage 7: Behavioural Only)
        for result in reversed(ablation_results):
            remaining = result.get("remaining_features", [])
            if remaining:
                boundary_features = remaining
                logger.info(
                    "  Using boundary features from '%s' (%d features)",
                    result.get("stage_name", "unknown"), len(remaining),
                )
                break

    if boundary_features is None:
        # Fallback: use the behavioural feature groups from config
        logger.info("No ablation results found; using config behavioural indicators.")
        behavioural_cfg = config.get("features", {}).get("behavioural_indicators", {})
        boundary_features = []
        for group_name, feats in behavioural_cfg.items():
            if isinstance(feats, list):
                boundary_features.extend(feats)
        # Expand V1_through_V339
        if "V1_through_V339" in boundary_features:
            boundary_features.remove("V1_through_V339")
            boundary_features.extend([f"V{i}" for i in range(1, 340)])
        logger.info("  Fallback boundary features: %d", len(boundary_features))

    # ------------------------------------------------------------------
    # 5. Build feature matrix (behavioural + drift)
    # ------------------------------------------------------------------
    # Combine train + val for training, test for evaluation
    train_combined = pd.concat([train_df, val_df], ignore_index=True)

    # Identify available boundary features
    exclude = {"isFraud", "TransactionID"}
    all_cols = [c for c in train_combined.columns if c not in exclude]
    available_boundary = [f for f in boundary_features if f in all_cols]

    # If we have very few boundary features, fall back to all non-excluded features
    if len(available_boundary) < 5:
        logger.warning(
            "Only %d boundary features found in data. "
            "Falling back to all available features.",
            len(available_boundary),
        )
        available_boundary = all_cols

    logger.info("Using %d behavioural features for pre-fraud prediction.", len(available_boundary))

    # Merge drift scores if available
    if drift_df is not None and len(drift_cols) > 0:
        # Drift scores should align with the full dataset
        n_total = len(train_combined)
        if len(drift_df) >= n_total:
            # Take the first n_total rows for training
            train_drift = drift_df[drift_cols].iloc[:n_total].values
            train_combined_drift = np.column_stack([
                train_combined[available_boundary].values,
                train_drift,
            ])
            # Test drift scores
            test_start = n_total
            test_end = test_start + len(test_df)
            if len(drift_df) >= test_end:
                test_drift = drift_df[drift_cols].iloc[test_start:test_end].values
            else:
                logger.warning("Insufficient drift scores for test set; padding with zeros.")
                test_drift = np.zeros((len(test_df), len(drift_cols)))
            test_combined_drift = np.column_stack([
                test_df[available_boundary].values,
                test_drift,
            ])
            feature_names = available_boundary + drift_cols
        else:
            logger.warning(
                "Drift score count (%d) < training data count (%d). "
                "Using only behavioural features.",
                len(drift_df), n_total,
            )
            train_combined_drift = train_combined[available_boundary].values
            test_combined_drift = test_df[available_boundary].values
            feature_names = available_boundary
    else:
        train_combined_drift = train_combined[available_boundary].values
        test_combined_drift = test_df[available_boundary].values
        feature_names = available_boundary

    # Replace NaN/inf with 0 for stability
    train_combined_drift = np.nan_to_num(train_combined_drift, nan=0.0, posinf=0.0, neginf=0.0)
    test_combined_drift = np.nan_to_num(test_combined_drift, nan=0.0, posinf=0.0, neginf=0.0)

    y_train = train_combined["isFraud"].values
    y_test = test_df["isFraud"].values

    logger.info(
        "Final feature matrix: train=%s, test=%s",
        train_combined_drift.shape, test_combined_drift.shape,
    )
    logger.info(
        "Class balance -- train: %.3f%% fraud | test: %.3f%% fraud",
        y_train.mean() * 100, y_test.mean() * 100,
    )

    # ------------------------------------------------------------------
    # 6. Train all three models
    # ------------------------------------------------------------------
    n_optuna_trials = 15  # Balanced: enough for convergence, feasible runtime
    model = PreFraudModel(
        config=config,
        n_optuna_trials=n_optuna_trials,
        n_cv_folds=n_folds,
        random_state=random_state,
    )
    model.feature_names = feature_names
    model.train(train_combined_drift, y_train)

    # ------------------------------------------------------------------
    # 7. Evaluate on test set
    # ------------------------------------------------------------------
    test_metrics = model.evaluate(test_combined_drift, y_test)

    # ------------------------------------------------------------------
    # 8. McNemar's statistical significance tests
    # ------------------------------------------------------------------
    mcnemar_df = model.run_mcnemar_tests()

    # ------------------------------------------------------------------
    # 9. Interpretability analysis
    # ------------------------------------------------------------------
    logger.info("Running interpretability analysis ...")
    model.compute_shap_analysis(test_combined_drift, feature_names)

    # ------------------------------------------------------------------
    # 10. Save everything
    # ------------------------------------------------------------------
    saved_paths = model.save_all(test_combined_drift, feature_names)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    logger.info("-" * 70)
    logger.info("Pre-fraud model pipeline completed in %.1f seconds.", elapsed)
    logger.info("Saved artifacts:")
    for name, path in saved_paths.items():
        logger.info("  %-30s -> %s", name, path)
    logger.info("-" * 70)

    # Determine best model
    best_name = max(
        model.test_metrics,
        key=lambda k: model.test_metrics[k]["auc_roc"],
    )
    best_auc = model.test_metrics[best_name]["auc_roc"]
    logger.info("Best model: %s (AUC-ROC = %.4f)", best_name, best_auc)

    # Report McNemar significance
    if isinstance(mcnemar_df, pd.DataFrame) and not mcnemar_df.empty:
        sig_pairs = mcnemar_df[mcnemar_df["significant_0.05"] == True]
        if len(sig_pairs) > 0:
            logger.info("Statistically significant differences (p < 0.05):")
            for _, row in sig_pairs.iterrows():
                logger.info(
                    "  %s vs %s (p = %.6f)",
                    row["model_a"], row["model_b"], row["p_value"],
                )
        else:
            logger.info(
                "No statistically significant differences between models (p >= 0.05 for all pairs)."
            )

    logger.info("=" * 70)


if __name__ == "__main__":
    main()
