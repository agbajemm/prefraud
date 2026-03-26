"""
Baseline XGBoost Model for Fraud Detection
===========================================

Trains an XGBoost classifier on ALL available features (excluding isFraud
and TransactionID) to establish an upper-bound baseline for the pre-fraud
detection research.  This baseline quantifies how well fraud can be detected
when every feature -- including direct transaction-level indicators -- is
available.  Subsequent ablation experiments will systematically remove direct
indicators to isolate the predictive contribution of behavioural signals.

Key capabilities
----------------
- Automatic class-weight calculation via ``scale_pos_weight``
- 5-fold stratified cross-validation with per-fold metric tracking
- Composite feature importance ranking (XGBoost gain, SHAP, permutation,
  mutual information)
- Publication-quality diagnostic plots (ROC, PR, SHAP summary, feature
  importance bar chart)
- Full metric persistence (JSON, CSV, pickle)

Target performance: AUC-ROC > 0.95 with full feature set.

Author : Michael Okonkwo (23090303)
"""

from __future__ import annotations

import json
import logging
import pickle
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend; must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
import yaml
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "baseline"
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
# BaselineModel
# ===================================================================
class BaselineModel:
    """XGBoost baseline trained on the full feature set.

    This class encapsulates the complete training, evaluation, and
    reporting workflow for the baseline fraud-detection model.

    Parameters
    ----------
    config : dict, optional
        Project configuration.  Loaded from ``config.yaml`` if *None*.
    """

    # Columns that must never be used as features
    EXCLUDE_COLS = {"isFraud", "TransactionID"}

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or load_config()
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: Optional[List[str]] = None

        # Metric storage
        self.cv_metrics: List[Dict[str, Any]] = []
        self.test_metrics: Dict[str, Any] = {}
        self.feature_importances: Optional[pd.DataFrame] = None

        # Ensure output directories exist
        for d in (FIGURES_DIR, TABLES_DIR, LOGS_DIR, MODELS_DIR):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _get_feature_cols(df: pd.DataFrame) -> List[str]:
        """Return all columns except the target and TransactionID."""
        return [
            c for c in df.columns
            if c not in BaselineModel.EXCLUDE_COLS
        ]

    @staticmethod
    def _compute_scale_pos_weight(y: pd.Series) -> float:
        """Compute class weight ratio for imbalanced binary targets.

        Returns ``n_negative / n_positive`` so the minority (fraud)
        class is up-weighted.
        """
        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)
        if n_pos == 0:
            raise ValueError("Target vector contains no positive samples.")
        weight = n_neg / n_pos
        logger.info(
            "Class distribution: neg=%d, pos=%d, scale_pos_weight=%.2f",
            n_neg, n_pos, weight,
        )
        return weight

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------
    def _build_model(self, scale_pos_weight: float) -> xgb.XGBClassifier:
        """Instantiate an XGBClassifier from the config, overriding
        ``scale_pos_weight`` with the value computed from data.

        Returns
        -------
        xgb.XGBClassifier
            Unfitted classifier instance.
        """
        baseline_cfg = self.config.get("model", {}).get("baseline", {})
        params = dict(baseline_cfg.get("params", {}))

        # Override auto scale_pos_weight
        params["scale_pos_weight"] = scale_pos_weight

        early_stopping = params.pop("early_stopping_rounds", 50)

        # Ensure reproducibility
        params.setdefault("random_state", self.config.get("data", {}).get("random_state", 42))
        params.setdefault("use_label_encoder", False)
        params.setdefault("verbosity", 0)
        params.setdefault("tree_method", "hist")
        params["early_stopping_rounds"] = int(early_stopping)
        params["n_jobs"] = -1

        clf = xgb.XGBClassifier(**params)
        self._early_stopping_rounds = int(early_stopping)
        return clf

    # ------------------------------------------------------------------
    # Precision at fixed recall
    # ------------------------------------------------------------------
    @staticmethod
    def _precision_at_recall(y_true: np.ndarray, y_prob: np.ndarray,
                             target_recall: float = 0.80) -> float:
        """Return the highest precision achievable at >= *target_recall*.

        Walks the precision-recall curve from the high-recall end and
        picks the first threshold where recall >= target.
        """
        precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
        # recalls is sorted descending; precisions is aligned
        valid = recalls >= target_recall
        if valid.any():
            return float(precisions[valid].max())
        return 0.0

    # ------------------------------------------------------------------
    # Single-fold evaluation
    # ------------------------------------------------------------------
    def _evaluate_fold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        fold_idx: int,
    ) -> Dict[str, Any]:
        """Compute all metrics for one CV fold.

        Returns a dict with scalar metrics and the confusion matrix.
        """
        y_pred = (y_prob >= 0.5).astype(int)

        auc_roc = roc_auc_score(y_true, y_prob)
        auc_pr = average_precision_score(y_true, y_prob)
        f1 = f1_score(y_true, y_pred)
        p_at_r80 = self._precision_at_recall(y_true, y_prob, 0.80)
        cm = confusion_matrix(y_true, y_pred).tolist()

        metrics = {
            "fold": fold_idx,
            "auc_roc": round(auc_roc, 6),
            "auc_pr": round(auc_pr, 6),
            "f1": round(f1, 6),
            "precision_at_recall_80": round(p_at_r80, 6),
            "confusion_matrix": cm,
        }

        logger.info(
            "  Fold %d -- AUC-ROC: %.4f | AUC-PR: %.4f | F1: %.4f | P@R80: %.4f",
            fold_idx, auc_roc, auc_pr, f1, p_at_r80,
        )
        return metrics

    # ------------------------------------------------------------------
    # Cross-validated training
    # ------------------------------------------------------------------
    def train_cv(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        n_folds: int = 5,
    ) -> None:
        """Run stratified *n_folds*-fold cross-validation, then retrain
        on the full training set.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data (features + ``isFraud``).
        val_df : pd.DataFrame, optional
            Hold-out validation set used for early stopping during the
            final full-training pass.
        n_folds : int
            Number of CV folds (default 5).
        """
        feature_cols = self._get_feature_cols(train_df)
        self.feature_names = feature_cols

        X = train_df[feature_cols].values
        y = train_df["isFraud"].values

        scale_pos_weight = self._compute_scale_pos_weight(
            train_df["isFraud"]
        )

        logger.info(
            "Starting %d-fold stratified CV on %d samples, %d features",
            n_folds, X.shape[0], X.shape[1],
        )

        skf = StratifiedKFold(
            n_splits=n_folds, shuffle=True,
            random_state=self.config.get("data", {}).get("random_state", 42),
        )

        self.cv_metrics = []
        oof_probs = np.zeros(len(y), dtype=np.float64)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            clf = self._build_model(scale_pos_weight)
            clf.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                verbose=False,
            )

            y_prob = clf.predict_proba(X_va)[:, 1]
            oof_probs[val_idx] = y_prob

            fold_metrics = self._evaluate_fold(y_va, y_prob, fold_idx)
            self.cv_metrics.append(fold_metrics)

        # Aggregate CV metrics
        agg = self._aggregate_cv_metrics()
        logger.info(
            "CV summary -- AUC-ROC: %.4f +/- %.4f | AUC-PR: %.4f +/- %.4f",
            agg["auc_roc_mean"], agg["auc_roc_std"],
            agg["auc_pr_mean"], agg["auc_pr_std"],
        )

        # ----------------------------------------------------------
        # Retrain on full training set
        # ----------------------------------------------------------
        logger.info("Retraining on full training set (%d samples)", X.shape[0])
        self.model = self._build_model(scale_pos_weight)

        fit_kwargs: Dict[str, Any] = {"verbose": False}
        if val_df is not None:
            X_val_es = val_df[feature_cols].values
            y_val_es = val_df["isFraud"].values
            fit_kwargs["eval_set"] = [(X_val_es, y_val_es)]

        self.model.fit(X, y, **fit_kwargs)
        logger.info("Final model trained (n_estimators used: %d)", self.model.n_estimators)

    # ------------------------------------------------------------------
    # CV metric aggregation
    # ------------------------------------------------------------------
    def _aggregate_cv_metrics(self) -> Dict[str, float]:
        """Compute mean and std for each scalar metric across folds."""
        scalar_keys = ["auc_roc", "auc_pr", "f1", "precision_at_recall_80"]
        agg: Dict[str, float] = {}
        for key in scalar_keys:
            values = [m[key] for m in self.cv_metrics]
            agg[f"{key}_mean"] = float(np.mean(values))
            agg[f"{key}_std"] = float(np.std(values))
        return agg

    # ------------------------------------------------------------------
    # Test-set evaluation
    # ------------------------------------------------------------------
    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate the trained model on a held-out test set.

        Parameters
        ----------
        test_df : pd.DataFrame
            Test data (features + ``isFraud``).

        Returns
        -------
        dict
            Dictionary of test-set metrics.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call train_cv() first.")

        feature_cols = self._get_feature_cols(test_df)
        X_test = test_df[feature_cols].values
        y_test = test_df["isFraud"].values

        y_prob = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        auc_roc = roc_auc_score(y_test, y_prob)
        auc_pr = average_precision_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        p_at_r80 = self._precision_at_recall(y_test, y_prob, 0.80)
        cm = confusion_matrix(y_test, y_pred).tolist()

        self.test_metrics = {
            "auc_roc": round(auc_roc, 6),
            "auc_pr": round(auc_pr, 6),
            "f1": round(f1, 6),
            "precision_at_recall_80": round(p_at_r80, 6),
            "confusion_matrix": cm,
            "n_test_samples": int(len(y_test)),
            "n_test_positive": int(y_test.sum()),
        }

        logger.info(
            "Test -- AUC-ROC: %.4f | AUC-PR: %.4f | F1: %.4f | P@R80: %.4f",
            auc_roc, auc_pr, f1, p_at_r80,
        )

        # Stash arrays for plotting
        self._y_test = y_test
        self._y_prob = y_prob
        self._y_pred = y_pred

        return self.test_metrics

    # ==================================================================
    # Feature importance: composite ranking
    # ==================================================================
    def compute_feature_importances(
        self,
        test_df: pd.DataFrame,
        n_permutation_repeats: int = 5,
    ) -> pd.DataFrame:
        """Build a composite feature-importance ranking from four sources.

        Sources
        -------
        1. XGBoost gain importance
        2. SHAP mean |value|
        3. Permutation importance (AUC-ROC)
        4. Mutual information with target

        Each source is min-max normalised to [0, 1] before averaging
        into a single ``composite_rank`` score.

        Parameters
        ----------
        test_df : pd.DataFrame
            Data used for permutation importance and SHAP.
        n_permutation_repeats : int
            Number of repeats for permutation importance.

        Returns
        -------
        pd.DataFrame
            Feature importance table sorted by composite score.
        """
        if self.model is None:
            raise RuntimeError("Model not trained.")

        feature_cols = self._get_feature_cols(test_df)
        X_test = test_df[feature_cols]
        y_test = test_df["isFraud"]

        logger.info("Computing composite feature importances ...")

        # 1. XGBoost gain importance -----------------------------------
        logger.info("  [1/4] XGBoost gain importance")
        booster = self.model.get_booster()
        gain_raw = booster.get_score(importance_type="gain")
        # Map internal feature names (f0, f1, ...) back to real names
        gain_series = pd.Series(0.0, index=feature_cols)
        for internal_name, score in gain_raw.items():
            idx = int(internal_name.replace("f", ""))
            if idx < len(feature_cols):
                gain_series.iloc[idx] = score
        gain_series.name = "xgb_gain"

        # 2. SHAP values -----------------------------------------------
        logger.info("  [2/4] SHAP values (TreeExplainer)")
        explainer = shap.TreeExplainer(self.model)
        # Use a subsample if the test set is very large to keep runtime
        # manageable (SHAP on >50 k rows can be slow).
        max_shap_samples = 10_000
        if len(X_test) > max_shap_samples:
            shap_sample = X_test.sample(
                n=max_shap_samples,
                random_state=42,
            )
        else:
            shap_sample = X_test

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_values = explainer.shap_values(shap_sample)

        # shap_values may be a list (one per class) or a 2-D array
        if isinstance(shap_values, list):
            # For binary classification take the positive-class array
            shap_array = np.abs(shap_values[1]) if len(shap_values) > 1 else np.abs(shap_values[0])
        else:
            shap_array = np.abs(shap_values)

        shap_mean = pd.Series(
            shap_array.mean(axis=0), index=feature_cols, name="shap_mean_abs"
        )

        # Stash for the SHAP summary plot
        self._shap_values_for_plot = shap_values
        self._shap_sample_df = shap_sample

        # 3. Permutation importance ------------------------------------
        logger.info("  [3/4] Permutation importance (%d repeats)", n_permutation_repeats)
        perm_result = permutation_importance(
            self.model, X_test, y_test,
            n_repeats=n_permutation_repeats,
            random_state=42,
            scoring="roc_auc",
            n_jobs=-1,
        )
        perm_series = pd.Series(
            perm_result.importances_mean, index=feature_cols, name="perm_importance"
        )

        # 4. Mutual information ----------------------------------------
        logger.info("  [4/4] Mutual information with target")
        X_mi = X_test.fillna(0).select_dtypes(include=[np.number])
        mi_scores = mutual_info_classif(
            X_mi, y_test, random_state=42, n_neighbors=5,
        )
        mi_series = pd.Series(mi_scores, index=X_mi.columns, name="mutual_info")
        # Re-index to full feature list (non-numeric features get 0)
        mi_series = mi_series.reindex(feature_cols, fill_value=0.0)

        # ----- Combine into a DataFrame --------------------------------
        df_imp = pd.DataFrame({
            "feature": feature_cols,
            "xgb_gain": gain_series.values,
            "shap_mean_abs": shap_mean.values,
            "perm_importance": perm_series.values,
            "mutual_info": mi_series.values,
        })

        # Min-max normalise each column to [0, 1]
        for col in ["xgb_gain", "shap_mean_abs", "perm_importance", "mutual_info"]:
            cmin, cmax = df_imp[col].min(), df_imp[col].max()
            if cmax - cmin > 0:
                df_imp[f"{col}_norm"] = (df_imp[col] - cmin) / (cmax - cmin)
            else:
                df_imp[f"{col}_norm"] = 0.0

        norm_cols = [c for c in df_imp.columns if c.endswith("_norm")]
        df_imp["composite_score"] = df_imp[norm_cols].mean(axis=1)
        df_imp = df_imp.sort_values("composite_score", ascending=False).reset_index(drop=True)

        self.feature_importances = df_imp
        logger.info("Composite feature importance computed for %d features.", len(df_imp))
        return df_imp

    # ==================================================================
    # Plotting
    # ==================================================================
    def plot_roc_curve(self) -> Path:
        """Generate and save a ROC curve from the most recent test evaluation.

        Returns
        -------
        Path
            File path of the saved figure.
        """
        fpr, tpr, _ = roc_curve(self._y_test, self._y_prob)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color="#2563eb", lw=2,
                label=f"Baseline XGBoost (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--",
                label="Random classifier")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curve -- Baseline Model (Full Features)", fontsize=14)
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out_path = FIGURES_DIR / "roc_curve.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        logger.info("Saved ROC curve to %s", out_path)
        return out_path

    def plot_pr_curve(self) -> Path:
        """Generate and save a Precision-Recall curve.

        Returns
        -------
        Path
            File path of the saved figure.
        """
        precisions, recalls, _ = precision_recall_curve(self._y_test, self._y_prob)
        pr_auc = average_precision_score(self._y_test, self._y_prob)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recalls, precisions, color="#dc2626", lw=2,
                label=f"Baseline XGBoost (AP = {pr_auc:.4f})")
        baseline_rate = self._y_test.mean()
        ax.axhline(baseline_rate, color="grey", lw=1, linestyle="--",
                    label=f"No-skill baseline ({baseline_rate:.4f})")
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title("Precision-Recall Curve -- Baseline Model", fontsize=14)
        ax.legend(loc="upper right", fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out_path = FIGURES_DIR / "pr_curve.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        logger.info("Saved PR curve to %s", out_path)
        return out_path

    def plot_shap_summary(self, max_display: int = 30) -> Path:
        """Generate and save a SHAP summary (beeswarm) plot.

        Parameters
        ----------
        max_display : int
            Number of top features to display.

        Returns
        -------
        Path
            File path of the saved figure.
        """
        if not hasattr(self, "_shap_values_for_plot"):
            raise RuntimeError(
                "SHAP values not available. Run compute_feature_importances() first."
            )

        shap_vals = self._shap_values_for_plot
        # Pick the positive-class SHAP values for plotting
        if isinstance(shap_vals, list):
            plot_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        else:
            plot_vals = shap_vals

        fig, ax = plt.subplots(figsize=(10, 12))
        shap.summary_plot(
            plot_vals,
            self._shap_sample_df,
            max_display=max_display,
            show=False,
            plot_size=None,
        )
        plt.title("SHAP Feature Importance (Top 30) -- Baseline", fontsize=14)
        plt.tight_layout()

        out_path = FIGURES_DIR / "shap_summary.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close("all")
        logger.info("Saved SHAP summary plot to %s", out_path)
        return out_path

    def plot_feature_importance_bar(self, top_n: int = 30) -> Path:
        """Bar chart of the composite feature importance scores.

        Parameters
        ----------
        top_n : int
            Number of top features to display.

        Returns
        -------
        Path
            File path of the saved figure.
        """
        if self.feature_importances is None:
            raise RuntimeError(
                "Feature importances not computed. "
                "Call compute_feature_importances() first."
            )

        df_top = self.feature_importances.head(top_n).copy()

        fig, ax = plt.subplots(figsize=(10, 10))
        sns.barplot(
            data=df_top,
            x="composite_score",
            y="feature",
            palette="viridis",
            ax=ax,
        )
        ax.set_xlabel("Composite Importance Score", fontsize=12)
        ax.set_ylabel("Feature", fontsize=12)
        ax.set_title(
            f"Top {top_n} Features -- Composite Importance (Baseline)",
            fontsize=14,
        )
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()

        out_path = FIGURES_DIR / "feature_importance_bar.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved feature importance bar chart to %s", out_path)
        return out_path

    # ==================================================================
    # Persistence
    # ==================================================================
    def save_model(self) -> Path:
        """Pickle the trained XGBClassifier to disk.

        Returns
        -------
        Path
            File path of the saved model.
        """
        if self.model is None:
            raise RuntimeError("No model to save.")
        out_path = MODELS_DIR / "baseline_xgb.pkl"
        with open(out_path, "wb") as fh:
            pickle.dump(self.model, fh)
        logger.info("Saved trained model to %s", out_path)
        return out_path

    def save_metrics(self) -> Dict[str, Path]:
        """Persist all metrics (CV + test) and feature importances.

        Returns
        -------
        dict
            Mapping of artifact name to its file path.
        """
        saved: Dict[str, Path] = {}

        # --- CV fold metrics (JSON) -----------------------------------
        cv_path = TABLES_DIR / "baseline_cv_metrics.json"
        with open(cv_path, "w") as fh:
            json.dump(self.cv_metrics, fh, indent=2)
        saved["cv_metrics"] = cv_path
        logger.info("Saved CV metrics to %s", cv_path)

        # --- Aggregated CV summary ------------------------------------
        agg = self._aggregate_cv_metrics()
        agg_path = TABLES_DIR / "baseline_cv_summary.json"
        with open(agg_path, "w") as fh:
            json.dump(agg, fh, indent=2)
        saved["cv_summary"] = agg_path

        # --- Test metrics (JSON) --------------------------------------
        if self.test_metrics:
            test_path = TABLES_DIR / "baseline_test_metrics.json"
            with open(test_path, "w") as fh:
                json.dump(self.test_metrics, fh, indent=2)
            saved["test_metrics"] = test_path
            logger.info("Saved test metrics to %s", test_path)

        # --- Feature importances (CSV) --------------------------------
        if self.feature_importances is not None:
            fi_path = TABLES_DIR / "baseline_feature_importances.csv"
            self.feature_importances.to_csv(fi_path, index=False)
            saved["feature_importances"] = fi_path
            logger.info("Saved feature importances to %s", fi_path)

        # --- Confusion matrix (CSV) -----------------------------------
        if self.test_metrics and "confusion_matrix" in self.test_metrics:
            cm_array = np.array(self.test_metrics["confusion_matrix"])
            cm_df = pd.DataFrame(
                cm_array,
                index=["Actual_Negative", "Actual_Positive"],
                columns=["Pred_Negative", "Pred_Positive"],
            )
            cm_path = TABLES_DIR / "baseline_confusion_matrix.csv"
            cm_df.to_csv(cm_path)
            saved["confusion_matrix"] = cm_path
            logger.info("Saved confusion matrix to %s", cm_path)

        return saved

    def save_all(self, test_df: pd.DataFrame) -> Dict[str, Path]:
        """Convenience wrapper: save model, metrics, and all plots.

        Parameters
        ----------
        test_df : pd.DataFrame
            Test data (needed only to verify plots can be generated).

        Returns
        -------
        dict
            Mapping of artifact name to its file path.
        """
        paths = self.save_metrics()
        paths["model"] = self.save_model()

        # Plots
        try:
            paths["roc_curve"] = self.plot_roc_curve()
        except Exception as exc:
            logger.warning("Could not save ROC curve: %s", exc)

        try:
            paths["pr_curve"] = self.plot_pr_curve()
        except Exception as exc:
            logger.warning("Could not save PR curve: %s", exc)

        try:
            paths["shap_summary"] = self.plot_shap_summary(max_display=30)
        except Exception as exc:
            logger.warning("Could not save SHAP summary: %s", exc)

        try:
            paths["feature_importance_bar"] = self.plot_feature_importance_bar(top_n=30)
        except Exception as exc:
            logger.warning("Could not save feature importance bar: %s", exc)

        logger.info("All artifacts saved. Total: %d files.", len(paths))
        return paths


# ===================================================================
# CLI entry point
# ===================================================================
def main() -> None:
    """End-to-end baseline pipeline: load data, train, evaluate, save."""
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("BASELINE MODEL -- Full Feature XGBoost")
    logger.info("=" * 70)

    # ---- Configuration ------------------------------------------------
    config = load_config()

    # ---- Load processed data ------------------------------------------
    processed_dir = PROJECT_ROOT / config["data"]["processed_path"]
    logger.info("Loading processed data from %s", processed_dir)

    try:
        train_df = pd.read_parquet(processed_dir / "train.parquet")
        val_df = pd.read_parquet(processed_dir / "val.parquet")
        test_df = pd.read_parquet(processed_dir / "test.parquet")
    except FileNotFoundError as exc:
        logger.error(
            "Processed data not found. Run data_loader.py first: %s", exc
        )
        raise SystemExit(1) from exc

    logger.info(
        "Loaded splits: train=%d, val=%d, test=%d",
        len(train_df), len(val_df), len(test_df),
    )
    logger.info(
        "Fraud rates -- train=%.3f%%, val=%.3f%%, test=%.3f%%",
        train_df["isFraud"].mean() * 100,
        val_df["isFraud"].mean() * 100,
        test_df["isFraud"].mean() * 100,
    )

    # ---- Train --------------------------------------------------------
    model = BaselineModel(config)
    n_folds = config.get("evaluation", {}).get("cross_validation_folds", 5)
    model.train_cv(train_df, val_df=val_df, n_folds=n_folds)

    # ---- Evaluate on test set -----------------------------------------
    test_metrics = model.evaluate(test_df)

    # ---- Feature importances -----------------------------------------
    model.compute_feature_importances(test_df, n_permutation_repeats=5)

    # ---- Persist everything ------------------------------------------
    saved_paths = model.save_all(test_df)

    elapsed = time.time() - t0
    logger.info("-" * 70)
    logger.info("Baseline pipeline completed in %.1f seconds.", elapsed)
    logger.info("Saved artifacts:")
    for name, path in saved_paths.items():
        logger.info("  %-25s -> %s", name, path)
    logger.info("-" * 70)

    # ---- Performance gate --------------------------------------------
    roc = test_metrics["auc_roc"]
    if roc >= 0.95:
        logger.info("PASS: AUC-ROC %.4f meets target (>= 0.95)", roc)
    else:
        logger.warning(
            "BELOW TARGET: AUC-ROC %.4f < 0.95. "
            "Review feature engineering or hyperparameters.", roc
        )


if __name__ == "__main__":
    main()
