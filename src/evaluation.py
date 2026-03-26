"""
Comprehensive Evaluation Framework
====================================

Provides a unified evaluation pipeline for the Pre-Fraud Detection research
project.  Brings together model comparison, error analysis, cross-validation
stability, computational cost profiling, and secondary-dataset validation
into a single reproducible workflow.

Key capabilities
----------------
- Side-by-side comparison of all model variants (AUC-ROC, AUC-PR, F1,
  Precision@Recall=80%, detection lead time)
- Overlaid ROC and Precision-Recall curves (publication quality)
- False-positive and false-negative characterisation
- Cross-validation stability analysis across folds
- Computational cost comparison (training and inference)
- Credit Card dataset replication experiment
- McNemar's statistical significance test between model pairs
- Full report generation

Author : Michael Okonkwo (23090303)
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend; must precede pyplot import

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy import stats
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
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

FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "comparison"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
LOGS_DIR = PROJECT_ROOT / "results" / "logs"

# ---------------------------------------------------------------------------
# Global plot style (consistent with visualisation.py)
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

MODEL_COLOURS = sns.color_palette("husl", 12)

FIGURE_DPI = 300
FONT_SIZES = {
    "title": 14,
    "axis_label": 12,
    "tick": 10,
    "legend": 10,
    "annotation": 9,
}


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
# Helper functions
# ===================================================================

def precision_at_recall(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float = 0.80,
) -> float:
    """Return the highest precision achievable at >= *target_recall*.

    Walks the precision-recall curve and selects the maximum precision
    among all operating points where recall meets or exceeds the target.

    Parameters
    ----------
    y_true : array-like
        Binary ground-truth labels (0 or 1).
    y_prob : array-like
        Predicted probabilities for the positive class.
    target_recall : float
        Minimum recall threshold (default 0.80).

    Returns
    -------
    float
        Best precision at the required recall level, or 0.0 if the
        target recall is never reached.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    valid = recalls >= target_recall
    if valid.any():
        return float(precisions[valid].max())
    return 0.0


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute the standard evaluation metric suite for this project.

    Parameters
    ----------
    y_true : array-like
        Binary ground-truth labels.
    y_prob : array-like
        Predicted probabilities for the positive class.
    y_pred : array-like
        Hard binary predictions (0 or 1).

    Returns
    -------
    dict
        Dictionary with keys ``auc_roc``, ``auc_pr``, ``f1``,
        ``precision``, ``recall``, ``p_at_r80``.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = np.asarray(y_pred)

    return {
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "auc_pr": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "p_at_r80": float(precision_at_recall(y_true, y_prob, 0.80)),
    }


def statistical_significance_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> Dict[str, Any]:
    """Perform McNemar's test to compare two classifiers.

    McNemar's test assesses whether two models make significantly
    different errors on the same data.  It is appropriate when both
    models are evaluated on an identical test set.

    Parameters
    ----------
    y_true : array-like
        Binary ground-truth labels.
    y_pred_a : array-like
        Hard predictions from model A.
    y_pred_b : array-like
        Hard predictions from model B.

    Returns
    -------
    dict
        ``statistic`` -- McNemar chi-squared statistic.
        ``p_value``   -- Two-sided p-value.
        ``significant`` -- *True* if p < 0.05.
        ``n_a_correct_b_wrong`` -- Discordant count (A right, B wrong).
        ``n_b_correct_a_wrong`` -- Discordant count (B right, A wrong).
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # Discordant cells of the 2x2 contingency table
    n_a_correct_b_wrong = int((correct_a & ~correct_b).sum())   # b
    n_b_correct_a_wrong = int((correct_b & ~correct_a).sum())   # c

    b = n_a_correct_b_wrong
    c = n_b_correct_a_wrong

    # McNemar's test with continuity correction
    if b + c == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        statistic = float((abs(b - c) - 1) ** 2 / (b + c))
        p_value = float(stats.chi2.sf(statistic, df=1))

    return {
        "statistic": round(statistic, 6),
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05,
        "n_a_correct_b_wrong": b,
        "n_b_correct_a_wrong": c,
    }


# ===================================================================
# Internal plotting helpers
# ===================================================================

def _ensure_dir(path: Path) -> Path:
    """Create parent directories for *path* if they do not exist."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_and_close(fig: plt.Figure, save_path: Path) -> None:
    """Save a figure, log the action, and close it."""
    save_path = _ensure_dir(save_path)
    fig.savefig(str(save_path), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved to %s", save_path)


# ===================================================================
# ComprehensiveEvaluator
# ===================================================================

class ComprehensiveEvaluator:
    """Unified evaluation pipeline for the pre-fraud detection study.

    Orchestrates model comparison, error analysis, cross-validation
    stability, computational cost profiling, and secondary-dataset
    validation.

    Parameters
    ----------
    config : dict, optional
        Project configuration.  Loaded from ``config.yaml`` if *None*.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or load_config()

        # Ensure output directories exist
        for d in (FIGURES_DIR, TABLES_DIR, LOGS_DIR):
            d.mkdir(parents=True, exist_ok=True)

        # Internal caches populated by the various analysis methods
        self._model_results: Dict[str, Dict[str, Any]] = {}
        self._comparison_df: Optional[pd.DataFrame] = None
        self._fp_analysis: Optional[pd.DataFrame] = None
        self._fn_analysis: Optional[pd.DataFrame] = None
        self._cv_stability: Optional[pd.DataFrame] = None
        self._cost_df: Optional[pd.DataFrame] = None
        self._credit_card_results: Optional[Dict[str, Any]] = None

    # ----------------------------------------------------------------
    # 1. compare_all_models
    # ----------------------------------------------------------------
    def compare_all_models(
        self,
        model_results: Dict[str, Dict[str, Any]],
    ) -> pd.DataFrame:
        """Produce a side-by-side comparison table for all models.

        Parameters
        ----------
        model_results : dict
            Mapping of ``model_name`` to a dictionary containing:

            - ``y_true``  -- array-like ground-truth labels
            - ``y_prob``  -- array-like predicted probabilities
            - ``y_pred``  -- array-like hard predictions
            - ``metrics`` -- dict with pre-computed metrics (optional;
              recomputed if missing)

        Returns
        -------
        pd.DataFrame
            Comparison table with columns Model, AUC-ROC, AUC-PR, F1,
            P@R80, Lead Time.  Also saved to
            ``results/tables/model_comparison.csv``.
        """
        self._model_results = model_results
        rows: List[Dict[str, Any]] = []

        for name, data in model_results.items():
            y_true = np.asarray(data["y_true"])
            y_prob = np.asarray(data["y_prob"])
            y_pred = np.asarray(data["y_pred"])

            # Recompute metrics to guarantee consistency
            metrics = compute_all_metrics(y_true, y_prob, y_pred)

            # Retrieve optional lead time (supplied externally)
            lead_time = data.get("metrics", {}).get("lead_time", np.nan)

            rows.append({
                "Model": name,
                "AUC-ROC": round(metrics["auc_roc"], 4),
                "AUC-PR": round(metrics["auc_pr"], 4),
                "F1": round(metrics["f1"], 4),
                "P@R80": round(metrics["p_at_r80"], 4),
                "Lead Time": lead_time,
            })

        comparison_df = pd.DataFrame(rows)
        comparison_df = comparison_df.sort_values("AUC-ROC", ascending=False)
        comparison_df = comparison_df.reset_index(drop=True)

        # Persist
        out_path = TABLES_DIR / "model_comparison.csv"
        comparison_df.to_csv(out_path, index=False)
        logger.info("Model comparison table saved to %s", out_path)

        self._comparison_df = comparison_df

        # Log summary
        logger.info("\n%s", comparison_df.to_string(index=False))

        return comparison_df

    # ----------------------------------------------------------------
    # 2. plot_overlaid_roc
    # ----------------------------------------------------------------
    def plot_overlaid_roc(
        self,
        model_results: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Path:
        """Overlay ROC curves for all models on a single figure.

        Parameters
        ----------
        model_results : dict, optional
            Same structure as :meth:`compare_all_models`.  If *None*,
            uses the results from the most recent ``compare_all_models``
            call.

        Returns
        -------
        Path
            File path of the saved figure
            (``results/figures/comparison/roc_overlay.png``).
        """
        results = model_results or self._model_results
        if not results:
            raise ValueError(
                "No model results available. Call compare_all_models() first "
                "or pass model_results directly."
            )

        fig, ax = plt.subplots(figsize=(8, 7))

        for idx, (name, data) in enumerate(results.items()):
            y_true = np.asarray(data["y_true"])
            y_prob = np.asarray(data["y_prob"])

            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)

            colour = MODEL_COLOURS[idx % len(MODEL_COLOURS)]
            ax.plot(
                fpr, tpr,
                color=colour,
                linewidth=2,
                label=f"{name} (AUC = {roc_auc:.4f})",
            )

        # Diagonal reference
        ax.plot(
            [0, 1], [0, 1],
            color="grey", linewidth=1, linestyle="--",
            label="Random classifier",
        )

        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        ax.set_xlabel("False Positive Rate", fontsize=FONT_SIZES["axis_label"])
        ax.set_ylabel("True Positive Rate", fontsize=FONT_SIZES["axis_label"])
        ax.set_title(
            "Receiver Operating Characteristic (ROC) Curves",
            fontsize=FONT_SIZES["title"],
            fontweight="bold",
        )
        ax.legend(
            loc="lower right",
            fontsize=FONT_SIZES["legend"],
            frameon=True,
        )
        ax.tick_params(labelsize=FONT_SIZES["tick"])

        fig.tight_layout()
        out_path = FIGURES_DIR / "roc_overlay.png"
        _save_and_close(fig, out_path)
        return out_path

    # ----------------------------------------------------------------
    # 3. plot_overlaid_pr
    # ----------------------------------------------------------------
    def plot_overlaid_pr(
        self,
        model_results: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Path:
        """Overlay Precision-Recall curves for all models on one figure.

        Parameters
        ----------
        model_results : dict, optional
            Same structure as :meth:`compare_all_models`.

        Returns
        -------
        Path
            File path of the saved figure
            (``results/figures/comparison/pr_overlay.png``).
        """
        results = model_results or self._model_results
        if not results:
            raise ValueError(
                "No model results available. Call compare_all_models() first "
                "or pass model_results directly."
            )

        fig, ax = plt.subplots(figsize=(8, 7))

        first_y_true = None

        for idx, (name, data) in enumerate(results.items()):
            y_true = np.asarray(data["y_true"])
            y_prob = np.asarray(data["y_prob"])

            if first_y_true is None:
                first_y_true = y_true

            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)

            colour = MODEL_COLOURS[idx % len(MODEL_COLOURS)]
            ax.plot(
                rec, prec,
                color=colour,
                linewidth=2,
                label=f"{name} (AP = {ap:.4f})",
            )

        # No-skill baseline (prevalence)
        if first_y_true is not None:
            baseline_rate = float(np.mean(first_y_true))
            ax.axhline(
                y=baseline_rate,
                color="grey", linewidth=1, linestyle="--",
                alpha=0.6,
                label=f"No-skill baseline ({baseline_rate:.4f})",
            )

        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        ax.set_xlabel("Recall", fontsize=FONT_SIZES["axis_label"])
        ax.set_ylabel("Precision", fontsize=FONT_SIZES["axis_label"])
        ax.set_title(
            "Precision-Recall Curves",
            fontsize=FONT_SIZES["title"],
            fontweight="bold",
        )
        ax.legend(
            loc="upper right",
            fontsize=FONT_SIZES["legend"],
            frameon=True,
        )
        ax.tick_params(labelsize=FONT_SIZES["tick"])

        fig.tight_layout()
        out_path = FIGURES_DIR / "pr_overlay.png"
        _save_and_close(fig, out_path)
        return out_path

    # ----------------------------------------------------------------
    # 4. analyse_false_positives
    # ----------------------------------------------------------------
    def analyse_false_positives(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        X: pd.DataFrame,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """Identify and characterise false-positive predictions.

        A false positive is a legitimate transaction that the model
        incorrectly flagged as fraud.  Understanding these cases is
        critical for operational deployment (customer friction).

        Parameters
        ----------
        y_true : array-like
            Binary ground-truth labels.
        y_prob : array-like
            Predicted probabilities for the positive class.
        X : pd.DataFrame
            Feature matrix aligned with *y_true*.
        threshold : float
            Classification threshold (default 0.5).

        Returns
        -------
        pd.DataFrame
            Summary statistics for false-positive samples, including
            per-feature mean, standard deviation, and deviation from the
            global population mean.
        """
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        y_pred = (y_prob >= threshold).astype(int)

        fp_mask = (y_pred == 1) & (y_true == 0)
        n_fp = int(fp_mask.sum())

        logger.info(
            "False positives: %d / %d negative samples (%.2f%%)",
            n_fp, int((y_true == 0).sum()),
            n_fp / max(int((y_true == 0).sum()), 1) * 100,
        )

        if n_fp == 0:
            logger.info("No false positives at threshold %.2f.", threshold)
            self._fp_analysis = pd.DataFrame()
            return self._fp_analysis

        fp_data = X.loc[fp_mask]
        numeric_cols = fp_data.select_dtypes(include=[np.number]).columns.tolist()

        # Summary statistics for FP samples vs global population
        global_means = X[numeric_cols].mean()
        global_stds = X[numeric_cols].std().replace(0, np.nan)

        fp_means = fp_data[numeric_cols].mean()
        fp_stds = fp_data[numeric_cols].std()

        summary = pd.DataFrame({
            "feature": numeric_cols,
            "global_mean": global_means.values,
            "fp_mean": fp_means.values,
            "fp_std": fp_stds.values,
            "deviation": ((fp_means - global_means) / global_stds).values,
        })
        summary["abs_deviation"] = summary["deviation"].abs()
        summary = summary.sort_values("abs_deviation", ascending=False)
        summary = summary.reset_index(drop=True)

        # Add metadata
        summary.attrs["n_false_positives"] = n_fp
        summary.attrs["threshold"] = threshold
        summary.attrs["fp_mean_prob"] = float(y_prob[fp_mask].mean())

        # Persist
        out_path = TABLES_DIR / "false_positive_analysis.csv"
        summary.to_csv(out_path, index=False)
        logger.info("False-positive analysis saved to %s", out_path)

        # Plot: top deviating features
        self._plot_error_analysis(
            summary.head(20),
            title="False Positives: Top Feature Deviations from Population",
            save_name="fp_feature_deviations.png",
        )

        # Plot: predicted probability distribution for FPs
        self._plot_error_prob_distribution(
            y_prob, fp_mask,
            title="Predicted Probability Distribution (False Positives)",
            save_name="fp_prob_distribution.png",
        )

        self._fp_analysis = summary
        return summary

    # ----------------------------------------------------------------
    # 5. analyse_false_negatives
    # ----------------------------------------------------------------
    def analyse_false_negatives(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        X: pd.DataFrame,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """Identify and characterise false-negative predictions (missed frauds).

        A false negative is a fraudulent transaction that the model
        failed to flag.  These represent the most dangerous failure mode
        in fraud detection.

        Parameters
        ----------
        y_true : array-like
            Binary ground-truth labels.
        y_prob : array-like
            Predicted probabilities for the positive class.
        X : pd.DataFrame
            Feature matrix aligned with *y_true*.
        threshold : float
            Classification threshold (default 0.5).

        Returns
        -------
        pd.DataFrame
            Summary statistics for false-negative samples, with the same
            schema as :meth:`analyse_false_positives`.
        """
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        y_pred = (y_prob >= threshold).astype(int)

        fn_mask = (y_pred == 0) & (y_true == 1)
        n_fn = int(fn_mask.sum())

        logger.info(
            "False negatives (missed frauds): %d / %d positive samples (%.2f%%)",
            n_fn, int(y_true.sum()),
            n_fn / max(int(y_true.sum()), 1) * 100,
        )

        if n_fn == 0:
            logger.info("No false negatives at threshold %.2f.", threshold)
            self._fn_analysis = pd.DataFrame()
            return self._fn_analysis

        fn_data = X.loc[fn_mask]
        numeric_cols = fn_data.select_dtypes(include=[np.number]).columns.tolist()

        # Compare FN samples to confirmed fraud (TP) samples
        tp_mask = (y_pred == 1) & (y_true == 1)
        tp_data = X.loc[tp_mask]

        global_fraud_means = X.loc[y_true == 1, numeric_cols].mean()
        global_fraud_stds = X.loc[y_true == 1, numeric_cols].std().replace(0, np.nan)

        fn_means = fn_data[numeric_cols].mean()
        fn_stds = fn_data[numeric_cols].std()

        summary = pd.DataFrame({
            "feature": numeric_cols,
            "fraud_mean": global_fraud_means.values,
            "fn_mean": fn_means.values,
            "fn_std": fn_stds.values,
            "deviation_from_fraud": (
                (fn_means - global_fraud_means) / global_fraud_stds
            ).values,
        })
        summary["abs_deviation"] = summary["deviation_from_fraud"].abs()
        summary = summary.sort_values("abs_deviation", ascending=False)
        summary = summary.reset_index(drop=True)

        # Metadata
        summary.attrs["n_false_negatives"] = n_fn
        summary.attrs["threshold"] = threshold
        summary.attrs["fn_mean_prob"] = float(y_prob[fn_mask].mean())
        summary.attrs["tp_count"] = int(tp_mask.sum())

        # Persist
        out_path = TABLES_DIR / "false_negative_analysis.csv"
        summary.to_csv(out_path, index=False)
        logger.info("False-negative analysis saved to %s", out_path)

        # Plot: top deviating features
        self._plot_error_analysis(
            summary.head(20),
            title="False Negatives: Feature Deviation from Detected Fraud",
            save_name="fn_feature_deviations.png",
        )

        # Plot: predicted probability distribution for FNs
        self._plot_error_prob_distribution(
            y_prob, fn_mask,
            title="Predicted Probability Distribution (False Negatives)",
            save_name="fn_prob_distribution.png",
        )

        self._fn_analysis = summary
        return summary

    # ----------------------------------------------------------------
    # Error analysis plotting helpers
    # ----------------------------------------------------------------
    def _plot_error_analysis(
        self,
        summary_df: pd.DataFrame,
        title: str,
        save_name: str,
    ) -> None:
        """Horizontal bar chart of the top feature deviations."""
        if summary_df.empty:
            return

        # Pick the deviation column (works for both FP and FN summaries)
        dev_col = (
            "deviation" if "deviation" in summary_df.columns
            else "deviation_from_fraud"
        )

        plot_df = summary_df.head(20).copy()
        plot_df = plot_df.sort_values(dev_col, ascending=True)

        fig, ax = plt.subplots(figsize=(9, max(4, len(plot_df) * 0.38)))

        colours = [
            "#e74c3c" if v > 0 else "#3498db"
            for v in plot_df[dev_col]
        ]
        ax.barh(
            plot_df["feature"],
            plot_df[dev_col],
            color=colours,
            edgecolor="white",
            height=0.7,
        )

        ax.set_xlabel(
            "Deviation (standard deviations from reference)",
            fontsize=FONT_SIZES["axis_label"],
        )
        ax.set_title(title, fontsize=FONT_SIZES["title"], fontweight="bold")
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.tick_params(labelsize=FONT_SIZES["tick"])

        fig.tight_layout()
        _save_and_close(fig, FIGURES_DIR / save_name)

    def _plot_error_prob_distribution(
        self,
        y_prob: np.ndarray,
        error_mask: np.ndarray,
        title: str,
        save_name: str,
    ) -> None:
        """Histogram of predicted probabilities for misclassified samples."""
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.hist(
            y_prob[~error_mask],
            bins=50, alpha=0.5, density=True,
            color="#3498db", label="Correctly classified",
            edgecolor="white", linewidth=0.5,
        )
        ax.hist(
            y_prob[error_mask],
            bins=50, alpha=0.7, density=True,
            color="#e74c3c", label="Misclassified",
            edgecolor="white", linewidth=0.5,
        )

        ax.set_xlabel("Predicted Probability", fontsize=FONT_SIZES["axis_label"])
        ax.set_ylabel("Density", fontsize=FONT_SIZES["axis_label"])
        ax.set_title(title, fontsize=FONT_SIZES["title"], fontweight="bold")
        ax.legend(fontsize=FONT_SIZES["legend"], frameon=True)
        ax.tick_params(labelsize=FONT_SIZES["tick"])

        fig.tight_layout()
        _save_and_close(fig, FIGURES_DIR / save_name)

    # ----------------------------------------------------------------
    # 6. cross_validation_stability
    # ----------------------------------------------------------------
    def cross_validation_stability(
        self,
        cv_results_dict: Dict[str, Dict[str, List[float]]],
    ) -> pd.DataFrame:
        """Analyse variance of metrics across CV folds for each model.

        Parameters
        ----------
        cv_results_dict : dict
            Mapping of ``model_name`` to a dict of metric lists, e.g.::

                {
                    "XGBoost Baseline": {
                        "auc_roc": [0.95, 0.94, 0.96, ...],
                        "auc_pr":  [0.72, 0.71, 0.73, ...],
                        "f1":      [0.65, 0.64, 0.66, ...],
                    },
                    ...
                }

        Returns
        -------
        pd.DataFrame
            Table with mean, std, min, max, and coefficient of variation
            for each metric-model pair.
        """
        rows: List[Dict[str, Any]] = []

        for model_name, metric_dict in cv_results_dict.items():
            for metric_name, fold_values in metric_dict.items():
                values = np.asarray(fold_values, dtype=float)
                mean_val = float(np.mean(values))
                std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                cv_coeff = (std_val / mean_val) if mean_val > 0 else 0.0

                rows.append({
                    "Model": model_name,
                    "Metric": metric_name,
                    "Mean": round(mean_val, 6),
                    "Std": round(std_val, 6),
                    "Min": round(float(np.min(values)), 6),
                    "Max": round(float(np.max(values)), 6),
                    "CV (%)": round(cv_coeff * 100, 2),
                    "n_folds": len(values),
                })

        stability_df = pd.DataFrame(rows)

        # Persist
        out_path = TABLES_DIR / "cv_stability_analysis.csv"
        stability_df.to_csv(out_path, index=False)
        logger.info("CV stability analysis saved to %s", out_path)

        # Plot: box plots per metric
        self._plot_cv_stability(cv_results_dict)

        self._cv_stability = stability_df
        return stability_df

    def _plot_cv_stability(
        self,
        cv_results_dict: Dict[str, Dict[str, List[float]]],
    ) -> None:
        """Generate box-and-whisker plots comparing CV fold distributions."""
        # Determine which metrics are available
        all_metrics: set = set()
        for metric_dict in cv_results_dict.values():
            all_metrics.update(metric_dict.keys())
        all_metrics_sorted = sorted(all_metrics)

        if not all_metrics_sorted:
            return

        n_metrics = len(all_metrics_sorted)
        fig, axes = plt.subplots(
            1, n_metrics,
            figsize=(5 * n_metrics, 6),
            squeeze=False,
        )

        for col_idx, metric_name in enumerate(all_metrics_sorted):
            ax = axes[0, col_idx]

            box_data: List[List[float]] = []
            labels: List[str] = []

            for model_name, metric_dict in cv_results_dict.items():
                if metric_name in metric_dict:
                    box_data.append(list(metric_dict[metric_name]))
                    labels.append(model_name)

            if box_data:
                bp = ax.boxplot(
                    box_data,
                    labels=labels,
                    patch_artist=True,
                    widths=0.6,
                )
                for patch_idx, patch in enumerate(bp["boxes"]):
                    colour = MODEL_COLOURS[patch_idx % len(MODEL_COLOURS)]
                    patch.set_facecolor(colour)
                    patch.set_alpha(0.6)

            ax.set_title(
                metric_name.upper().replace("_", " "),
                fontsize=FONT_SIZES["axis_label"],
                fontweight="bold",
            )
            ax.tick_params(axis="x", rotation=30, labelsize=FONT_SIZES["tick"])
            ax.tick_params(axis="y", labelsize=FONT_SIZES["tick"])
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            "Cross-Validation Stability Across Folds",
            fontsize=FONT_SIZES["title"],
            fontweight="bold",
            y=1.02,
        )
        fig.tight_layout()
        _save_and_close(fig, FIGURES_DIR / "cv_stability_boxplots.png")

    # ----------------------------------------------------------------
    # 7. computational_cost_comparison
    # ----------------------------------------------------------------
    def computational_cost_comparison(
        self,
        timing_dict: Dict[str, Dict[str, float]],
    ) -> pd.DataFrame:
        """Compare training and inference times across models.

        Parameters
        ----------
        timing_dict : dict
            Mapping of ``model_name`` to a dict with keys:

            - ``train_time_s``     -- Training time in seconds.
            - ``inference_time_s`` -- Inference time in seconds.
            - ``n_train_samples``  -- Number of training samples (optional).
            - ``n_test_samples``   -- Number of test samples (optional).

        Returns
        -------
        pd.DataFrame
            Cost comparison table with per-sample throughput estimates.
        """
        rows: List[Dict[str, Any]] = []

        for model_name, timings in timing_dict.items():
            train_t = timings.get("train_time_s", np.nan)
            infer_t = timings.get("inference_time_s", np.nan)
            n_train = timings.get("n_train_samples", np.nan)
            n_test = timings.get("n_test_samples", np.nan)

            # Throughput (samples per second)
            train_throughput = (
                n_train / train_t if (
                    not np.isnan(n_train) and not np.isnan(train_t) and train_t > 0
                ) else np.nan
            )
            infer_throughput = (
                n_test / infer_t if (
                    not np.isnan(n_test) and not np.isnan(infer_t) and infer_t > 0
                ) else np.nan
            )

            rows.append({
                "Model": model_name,
                "Train Time (s)": round(train_t, 3) if not np.isnan(train_t) else np.nan,
                "Inference Time (s)": round(infer_t, 3) if not np.isnan(infer_t) else np.nan,
                "Train Samples": int(n_train) if not np.isnan(n_train) else np.nan,
                "Test Samples": int(n_test) if not np.isnan(n_test) else np.nan,
                "Train Throughput (samples/s)": (
                    round(train_throughput, 1) if not np.isnan(train_throughput) else np.nan
                ),
                "Inference Throughput (samples/s)": (
                    round(infer_throughput, 1) if not np.isnan(infer_throughput) else np.nan
                ),
            })

        cost_df = pd.DataFrame(rows)

        # Persist
        out_path = TABLES_DIR / "computational_cost_comparison.csv"
        cost_df.to_csv(out_path, index=False)
        logger.info("Computational cost comparison saved to %s", out_path)

        # Plot: grouped bar chart of training vs inference time
        self._plot_computational_cost(cost_df)

        self._cost_df = cost_df
        return cost_df

    def _plot_computational_cost(self, cost_df: pd.DataFrame) -> None:
        """Grouped bar chart of training and inference times."""
        if cost_df.empty:
            return

        fig, ax = plt.subplots(figsize=(max(8, len(cost_df) * 1.5), 6))

        x = np.arange(len(cost_df))
        bar_width = 0.35

        train_times = cost_df["Train Time (s)"].values
        infer_times = cost_df["Inference Time (s)"].values

        bars_train = ax.bar(
            x - bar_width / 2, train_times,
            bar_width,
            label="Training",
            color="#2563eb",
            edgecolor="white",
        )
        bars_infer = ax.bar(
            x + bar_width / 2, infer_times,
            bar_width,
            label="Inference",
            color="#dc2626",
            edgecolor="white",
        )

        # Value labels
        for bar_set in (bars_train, bars_infer):
            for bar in bar_set:
                height = bar.get_height()
                if not np.isnan(height) and height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height,
                        f"{height:.1f}s",
                        ha="center", va="bottom",
                        fontsize=FONT_SIZES["annotation"],
                        fontweight="bold",
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(
            cost_df["Model"].values,
            rotation=25, ha="right",
            fontsize=FONT_SIZES["tick"],
        )
        ax.set_ylabel("Time (seconds)", fontsize=FONT_SIZES["axis_label"])
        ax.set_title(
            "Computational Cost Comparison",
            fontsize=FONT_SIZES["title"],
            fontweight="bold",
        )
        ax.legend(fontsize=FONT_SIZES["legend"], frameon=True)
        ax.tick_params(labelsize=FONT_SIZES["tick"])
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        _save_and_close(fig, FIGURES_DIR / "computational_cost.png")

    # ----------------------------------------------------------------
    # 8. validate_on_credit_card
    # ----------------------------------------------------------------
    def validate_on_credit_card(
        self,
        credit_card_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Replicate the core experiment on the Credit Card Fraud dataset.

        This method validates the project's central hypothesis on an
        independent dataset with a different feature representation
        (PCA-transformed V1-V28 plus Amount and Time).

        The experiment:

        1. Train a baseline XGBoost on all 30 features.
        2. Progressively ablate features by importance (mimicking the
           IEEE-CIS ablation protocol).
        3. Evaluate pre-fraud prediction capability at each stage.
        4. Compare degradation patterns with the IEEE-CIS results.

        Parameters
        ----------
        credit_card_df : pd.DataFrame
            Raw Credit Card Fraud dataset with columns ``Time``,
            ``V1``-``V28``, ``Amount``, ``Class``.

        Returns
        -------
        dict
            Results dictionary containing per-stage metrics and
            comparison notes.
        """
        import xgboost as xgb

        logger.info("=" * 70)
        logger.info("CREDIT CARD VALIDATION EXPERIMENT")
        logger.info("=" * 70)

        # ----- Prepare data -----------------------------------------------
        target_col = "Class"
        feature_cols = [c for c in credit_card_df.columns if c != target_col]
        pca_features = [c for c in feature_cols if c.startswith("V")]
        other_features = [c for c in feature_cols if c in ("Time", "Amount")]

        logger.info(
            "Dataset: %d samples, %d features, fraud rate: %.3f%%",
            len(credit_card_df),
            len(feature_cols),
            credit_card_df[target_col].mean() * 100,
        )

        # Chronological split on Time
        credit_card_df = credit_card_df.sort_values("Time").reset_index(drop=True)
        n = len(credit_card_df)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        train_df = credit_card_df.iloc[:train_end]
        val_df = credit_card_df.iloc[train_end:val_end]
        test_df = credit_card_df.iloc[val_end:]

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_val = val_df[feature_cols]
        y_val = val_df[target_col]
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]

        logger.info(
            "Splits: train=%d, val=%d, test=%d",
            len(X_train), len(X_val), len(X_test),
        )

        # ----- Helper: train & evaluate -----------------------------------
        def _train_evaluate(
            X_tr: pd.DataFrame,
            y_tr: pd.Series,
            X_va: pd.DataFrame,
            y_va: pd.Series,
            X_te: pd.DataFrame,
            y_te: pd.Series,
            stage_label: str,
        ) -> Dict[str, Any]:
            n_neg = int((y_tr == 0).sum())
            n_pos = int((y_tr == 1).sum())
            spw = n_neg / max(n_pos, 1)

            model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=spw,
                eval_metric="auc",
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                tree_method="hist",
            )
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                verbose=False,
            )

            y_prob = model.predict_proba(X_te)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            metrics = compute_all_metrics(y_te.values, y_prob, y_pred)

            # Feature importance for ablation ordering
            booster = model.get_booster()
            gain_raw = booster.get_score(importance_type="gain")
            gain_map = {}
            cols = list(X_tr.columns)
            for internal_name, score in gain_raw.items():
                # XGBoost may return feature names directly or "fN" indices
                if internal_name in cols:
                    gain_map[internal_name] = score
                elif internal_name.startswith("f"):
                    try:
                        idx = int(internal_name[1:])
                        if idx < len(cols):
                            gain_map[cols[idx]] = score
                    except ValueError:
                        pass

            logger.info(
                "  %s -- AUC-ROC: %.4f | AUC-PR: %.4f | F1: %.4f",
                stage_label, metrics["auc_roc"], metrics["auc_pr"], metrics["f1"],
            )

            return {
                "stage": stage_label,
                "metrics": metrics,
                "n_features": len(X_tr.columns),
                "feature_importance": gain_map,
                "y_true": y_te.values.tolist(),
                "y_prob": y_prob.tolist(),
            }

        # ----- Stage 0: Full model ----------------------------------------
        all_results: List[Dict[str, Any]] = []

        stage0 = _train_evaluate(
            X_train, y_train, X_val, y_val, X_test, y_test,
            "Stage 0: Full Model (V1-V28 + Amount + Time)",
        )
        all_results.append(stage0)

        # Rank features by importance for progressive removal
        imp = stage0["feature_importance"]
        sorted_features = sorted(imp.keys(), key=lambda f: imp[f], reverse=True)
        # Include features that did not appear in gain (zero importance)
        remaining_set = set(feature_cols)
        for f in sorted_features:
            remaining_set.discard(f)
        sorted_features.extend(sorted(remaining_set))

        # ----- Progressive ablation stages --------------------------------
        ablation_schedules = [
            ("Stage 1: Remove Top-1 Feature", 1),
            ("Stage 2: Remove Top-3 Features", 3),
            ("Stage 3: Remove Top-5 Features", 5),
            ("Stage 4: Remove Top-10 Features", 10),
            ("Stage 5: Remove Top-15 Features", 15),
            ("Stage 6: Remove Top-20 Features", 20),
            ("Stage 7: Retain Only V1-V5 + Time", None),  # special
        ]

        for stage_label, n_remove in ablation_schedules:
            if n_remove is not None:
                features_to_remove = sorted_features[:n_remove]
                cols_remaining = [
                    c for c in feature_cols if c not in features_to_remove
                ]
            else:
                # Retain only the weakest PCA features plus Time
                cols_remaining = ["V1", "V2", "V3", "V4", "V5", "Time"]
                cols_remaining = [c for c in cols_remaining if c in feature_cols]

            if len(cols_remaining) == 0:
                logger.warning("  %s: no features remaining, skipping.", stage_label)
                continue

            stage_result = _train_evaluate(
                X_train[cols_remaining], y_train,
                X_val[cols_remaining], y_val,
                X_test[cols_remaining], y_test,
                stage_label,
            )
            all_results.append(stage_result)

        # ----- Summarise --------------------------------------------------
        summary_rows = []
        for r in all_results:
            summary_rows.append({
                "Stage": r["stage"],
                "N Features": r["n_features"],
                "AUC-ROC": round(r["metrics"]["auc_roc"], 4),
                "AUC-PR": round(r["metrics"]["auc_pr"], 4),
                "F1": round(r["metrics"]["f1"], 4),
                "P@R80": round(r["metrics"]["p_at_r80"], 4),
            })

        summary_df = pd.DataFrame(summary_rows)
        out_path = TABLES_DIR / "credit_card_validation.csv"
        summary_df.to_csv(out_path, index=False)
        logger.info("Credit Card validation summary saved to %s", out_path)

        # ----- Comparison plot: ablation curves ---------------------------
        fig, ax = plt.subplots(figsize=(10, 6))
        stages = [r["stage"] for r in all_results]
        auc_rocs = [r["metrics"]["auc_roc"] for r in all_results]
        auc_prs = [r["metrics"]["auc_pr"] for r in all_results]
        x = np.arange(len(stages))

        ax.plot(x, auc_rocs, "o-", color="#2563eb", linewidth=2, markersize=7, label="AUC-ROC")
        ax.plot(x, auc_prs, "s--", color="#dc2626", linewidth=2, markersize=7, label="AUC-PR")

        for i, (roc_v, pr_v) in enumerate(zip(auc_rocs, auc_prs)):
            ax.annotate(
                f"{roc_v:.3f}",
                (x[i], roc_v),
                textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=FONT_SIZES["annotation"],
            )

        ax.axhline(y=0.80, color="grey", linestyle=":", alpha=0.6, label="Threshold (0.80)")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [s.split(":")[0] for s in stages],
            rotation=30, ha="right",
            fontsize=FONT_SIZES["tick"],
        )
        ax.set_ylabel("Score", fontsize=FONT_SIZES["axis_label"])
        ax.set_title(
            "Credit Card Dataset: Ablation Curve",
            fontsize=FONT_SIZES["title"],
            fontweight="bold",
        )
        ax.legend(fontsize=FONT_SIZES["legend"], frameon=True)
        ax.tick_params(labelsize=FONT_SIZES["tick"])
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        _save_and_close(fig, FIGURES_DIR / "credit_card_ablation_curve.png")

        # ----- Store results ----------------------------------------------
        results = {
            "stages": all_results,
            "summary": summary_df.to_dict(orient="records"),
            "baseline_auc_roc": all_results[0]["metrics"]["auc_roc"],
            "final_auc_roc": all_results[-1]["metrics"]["auc_roc"],
            "degradation_pct": round(
                (1 - all_results[-1]["metrics"]["auc_roc"] / all_results[0]["metrics"]["auc_roc"]) * 100,
                2,
            ),
        }

        self._credit_card_results = results
        logger.info(
            "Credit Card validation complete. "
            "Baseline AUC-ROC: %.4f -> Final: %.4f (%.1f%% degradation)",
            results["baseline_auc_roc"],
            results["final_auc_roc"],
            results["degradation_pct"],
        )

        return results

    # ----------------------------------------------------------------
    # 9. generate_full_report
    # ----------------------------------------------------------------
    def generate_full_report(
        self,
        model_results: Dict[str, Dict[str, Any]],
        cv_results_dict: Optional[Dict[str, Dict[str, List[float]]]] = None,
        timing_dict: Optional[Dict[str, Dict[str, float]]] = None,
        credit_card_df: Optional[pd.DataFrame] = None,
        primary_model_name: Optional[str] = None,
        X_test: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Run all evaluations and persist every artifact.

        This is the top-level entry point that orchestrates the full
        evaluation pipeline, suitable for end-of-experiment reporting.

        Parameters
        ----------
        model_results : dict
            Model results dict (see :meth:`compare_all_models`).
        cv_results_dict : dict, optional
            CV fold results (see :meth:`cross_validation_stability`).
        timing_dict : dict, optional
            Timing information (see :meth:`computational_cost_comparison`).
        credit_card_df : pd.DataFrame, optional
            Credit Card dataset for secondary validation.
        primary_model_name : str, optional
            Name of the primary model for detailed error analysis.
            Defaults to the first model in *model_results*.
        X_test : pd.DataFrame, optional
            Feature matrix for the primary model's test set (required
            for error analysis).

        Returns
        -------
        dict
            Summary of all generated artifacts and key metrics.
        """
        t0 = time.time()
        artifacts: Dict[str, Any] = {}

        logger.info("=" * 70)
        logger.info("GENERATING FULL EVALUATION REPORT")
        logger.info("=" * 70)

        # --- 1. Model comparison table ------------------------------------
        logger.info("[1/7] Comparing all models ...")
        comparison_df = self.compare_all_models(model_results)
        artifacts["model_comparison"] = TABLES_DIR / "model_comparison.csv"

        # --- 2. Overlaid ROC curves ---------------------------------------
        logger.info("[2/7] Plotting overlaid ROC curves ...")
        try:
            artifacts["roc_overlay"] = self.plot_overlaid_roc()
        except Exception as exc:
            logger.warning("Could not generate ROC overlay: %s", exc)

        # --- 3. Overlaid PR curves ----------------------------------------
        logger.info("[3/7] Plotting overlaid PR curves ...")
        try:
            artifacts["pr_overlay"] = self.plot_overlaid_pr()
        except Exception as exc:
            logger.warning("Could not generate PR overlay: %s", exc)

        # --- 4 & 5. Error analysis ----------------------------------------
        if primary_model_name is None:
            primary_model_name = list(model_results.keys())[0]

        primary_data = model_results.get(primary_model_name)

        if primary_data is not None and X_test is not None:
            y_true = np.asarray(primary_data["y_true"])
            y_prob = np.asarray(primary_data["y_prob"])

            logger.info("[4/7] Analysing false positives (%s) ...", primary_model_name)
            try:
                self.analyse_false_positives(y_true, y_prob, X_test)
                artifacts["fp_analysis"] = TABLES_DIR / "false_positive_analysis.csv"
            except Exception as exc:
                logger.warning("False-positive analysis failed: %s", exc)

            logger.info("[5/7] Analysing false negatives (%s) ...", primary_model_name)
            try:
                self.analyse_false_negatives(y_true, y_prob, X_test)
                artifacts["fn_analysis"] = TABLES_DIR / "false_negative_analysis.csv"
            except Exception as exc:
                logger.warning("False-negative analysis failed: %s", exc)
        else:
            logger.info(
                "[4-5/7] Skipping error analysis (no X_test or primary model data)."
            )

        # --- 6. Cross-validation stability --------------------------------
        if cv_results_dict is not None:
            logger.info("[6/7] Analysing cross-validation stability ...")
            try:
                self.cross_validation_stability(cv_results_dict)
                artifacts["cv_stability"] = TABLES_DIR / "cv_stability_analysis.csv"
            except Exception as exc:
                logger.warning("CV stability analysis failed: %s", exc)
        else:
            logger.info("[6/7] Skipping CV stability (no cv_results_dict provided).")

        # --- 7. Computational cost ----------------------------------------
        if timing_dict is not None:
            logger.info("[7/7] Comparing computational costs ...")
            try:
                self.computational_cost_comparison(timing_dict)
                artifacts["cost_comparison"] = TABLES_DIR / "computational_cost_comparison.csv"
            except Exception as exc:
                logger.warning("Cost comparison failed: %s", exc)
        else:
            logger.info("[7/7] Skipping cost comparison (no timing_dict provided).")

        # --- Bonus: Credit Card validation --------------------------------
        if credit_card_df is not None:
            logger.info("[Bonus] Running Credit Card validation experiment ...")
            try:
                self.validate_on_credit_card(credit_card_df)
                artifacts["credit_card_validation"] = (
                    TABLES_DIR / "credit_card_validation.csv"
                )
            except Exception as exc:
                logger.warning("Credit Card validation failed: %s", exc)

        # --- Statistical significance tests between top models ------------
        model_names = list(model_results.keys())
        if len(model_names) >= 2:
            logger.info("Running pairwise McNemar tests ...")
            sig_rows: List[Dict[str, Any]] = []
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    name_a = model_names[i]
                    name_b = model_names[j]
                    try:
                        result = statistical_significance_test(
                            np.asarray(model_results[name_a]["y_true"]),
                            np.asarray(model_results[name_a]["y_pred"]),
                            np.asarray(model_results[name_b]["y_pred"]),
                        )
                        sig_rows.append({
                            "Model A": name_a,
                            "Model B": name_b,
                            **result,
                        })
                    except Exception as exc:
                        logger.warning(
                            "McNemar test (%s vs %s) failed: %s",
                            name_a, name_b, exc,
                        )

            if sig_rows:
                sig_df = pd.DataFrame(sig_rows)
                sig_path = TABLES_DIR / "mcnemar_significance_tests.csv"
                sig_df.to_csv(sig_path, index=False)
                artifacts["significance_tests"] = sig_path
                logger.info("Pairwise significance tests saved to %s", sig_path)

        # --- Final summary ------------------------------------------------
        elapsed = time.time() - t0
        logger.info("-" * 70)
        logger.info("Full evaluation report generated in %.1f seconds.", elapsed)
        logger.info("Artifacts produced: %d", len(artifacts))
        for name, path in artifacts.items():
            logger.info("  %-30s -> %s", name, path)
        logger.info("-" * 70)

        return artifacts


# ===================================================================
# CLI entry point
# ===================================================================

def main() -> None:
    """Demonstrate loading results and running the full evaluation.

    This entry point assumes that model training has already been
    performed and results are available (either as saved files or
    synthesised here for demonstration).
    """
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("EVALUATION FRAMEWORK -- Comprehensive Report")
    logger.info("=" * 70)

    config = load_config()

    # ------------------------------------------------------------------
    # Attempt to load real results; fall back to synthetic demonstration
    # ------------------------------------------------------------------
    model_results: Dict[str, Dict[str, Any]] = {}
    cv_results_dict: Dict[str, Dict[str, List[float]]] = {}
    timing_dict: Dict[str, Dict[str, float]] = {}
    X_test: Optional[pd.DataFrame] = None
    credit_card_df: Optional[pd.DataFrame] = None

    # Try loading processed test data for real evaluation
    processed_dir = PROJECT_ROOT / config["data"]["processed_path"]
    test_path = processed_dir / "test.parquet"

    if test_path.exists():
        logger.info("Found processed test data. Loading real results ...")
        test_df = pd.read_parquet(test_path)

        exclude_cols = {"TransactionID", "isFraud"}
        feature_cols = [c for c in test_df.columns if c not in exclude_cols]
        X_test = test_df[feature_cols]
        y_test = test_df["isFraud"].values

        # Try loading saved model predictions
        import pickle

        models_to_check = {
            "Baseline XGBoost": PROJECT_ROOT / "results" / "models" / "baseline_xgb.pkl",
        }
        # Also check for ablation stage models
        for stage_idx in range(8):
            model_path = (
                PROJECT_ROOT / "results" / "models" / f"ablation_stage_{stage_idx}.pkl"
            )
            if model_path.exists():
                models_to_check[f"Ablation Stage {stage_idx}"] = model_path

        for model_name, model_path in models_to_check.items():
            if model_path.exists():
                try:
                    with open(model_path, "rb") as fh:
                        model = pickle.load(fh)
                    y_prob = model.predict_proba(X_test)[:, 1]
                    y_pred = (y_prob >= 0.5).astype(int)
                    model_results[model_name] = {
                        "y_true": y_test,
                        "y_prob": y_prob,
                        "y_pred": y_pred,
                        "metrics": {},
                    }
                    logger.info("Loaded model: %s", model_name)
                except Exception as exc:
                    logger.warning("Could not load %s: %s", model_name, exc)

        # Try loading CV metrics
        cv_json_path = TABLES_DIR / "baseline_cv_metrics.json"
        if cv_json_path.exists():
            try:
                with open(cv_json_path, "r") as fh:
                    cv_data = json.load(fh)
                cv_results_dict["Baseline XGBoost"] = {
                    "auc_roc": [fold["auc_roc"] for fold in cv_data],
                    "auc_pr": [fold["auc_pr"] for fold in cv_data],
                    "f1": [fold["f1"] for fold in cv_data],
                }
            except Exception as exc:
                logger.warning("Could not load CV metrics: %s", exc)

    # If no real data available, generate synthetic demonstration data
    if not model_results:
        logger.info("No trained models found. Generating synthetic demonstration ...")
        rng = np.random.default_rng(42)

        n_samples = 10_000
        n_positive = 350
        y_true = np.concatenate([
            np.zeros(n_samples - n_positive),
            np.ones(n_positive),
        ]).astype(int)

        # Create a simple feature matrix for error analysis
        X_test = pd.DataFrame(
            rng.standard_normal((n_samples, 10)),
            columns=[f"feature_{i}" for i in range(10)],
        )

        # Simulate three model variants with progressively weaker signals
        for model_name, noise_scale in [
            ("Baseline XGBoost (Full)", 0.15),
            ("Ablation Stage 3", 0.30),
            ("Behavioural Only (Stage 7)", 0.50),
        ]:
            signal = y_true.astype(float) * 0.8
            noise = rng.normal(0, noise_scale, n_samples)
            y_prob = np.clip(signal + noise + 0.1, 0.01, 0.99)
            y_pred = (y_prob >= 0.5).astype(int)

            model_results[model_name] = {
                "y_true": y_true,
                "y_prob": y_prob,
                "y_pred": y_pred,
                "metrics": {"lead_time": rng.uniform(1.0, 14.0)},
            }

        # Synthetic CV results
        for model_name in model_results:
            cv_results_dict[model_name] = {
                "auc_roc": list(rng.normal(0.92, 0.02, 5)),
                "auc_pr": list(rng.normal(0.70, 0.03, 5)),
                "f1": list(rng.normal(0.65, 0.04, 5)),
            }

        # Synthetic timing
        for idx, model_name in enumerate(model_results):
            timing_dict[model_name] = {
                "train_time_s": float(rng.uniform(30, 300)),
                "inference_time_s": float(rng.uniform(0.5, 5.0)),
                "n_train_samples": 400_000,
                "n_test_samples": n_samples,
            }

    # Try loading Credit Card dataset
    cc_path = PROJECT_ROOT / config["data"]["credit_card_path"]
    if cc_path.exists():
        try:
            credit_card_df = pd.read_csv(cc_path)
            logger.info("Loaded Credit Card dataset (%d rows)", len(credit_card_df))
        except Exception as exc:
            logger.warning("Could not load Credit Card dataset: %s", exc)

    # ------------------------------------------------------------------
    # Run the full evaluation
    # ------------------------------------------------------------------
    evaluator = ComprehensiveEvaluator(config)
    artifacts = evaluator.generate_full_report(
        model_results=model_results,
        cv_results_dict=cv_results_dict if cv_results_dict else None,
        timing_dict=timing_dict if timing_dict else None,
        credit_card_df=credit_card_df,
        X_test=X_test,
    )

    elapsed = time.time() - t0
    logger.info("=" * 70)
    logger.info("Evaluation complete in %.1f seconds. %d artifacts generated.", elapsed, len(artifacts))
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
