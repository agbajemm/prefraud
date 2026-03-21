"""
Visualisation Module
Publication-quality figures for the Pre-Fraud Detection research project.

Provides a Visualiser class with methods for every major plot type needed
across data exploration, model evaluation, ablation analysis, and
interpretability sections of the thesis.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Global style configuration
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

# Consistent colour palette across all figures
PALETTE = {
    "non_fraud": "#3498db",   # blue
    "fraud": "#e74c3c",       # red
    "primary": "#2c3e50",     # dark slate
    "secondary": "#27ae60",   # green
    "accent": "#f39c12",      # orange
    "highlight": "#8e44ad",   # purple
    "ci_band": "#b0c4de",     # light steel blue
}

# Categorical model palette for multi-model comparison plots
MODEL_COLOURS = sns.color_palette("husl", 12)

FIGURE_DPI = 300
FONT_SIZES = {
    "title": 14,
    "axis_label": 12,
    "tick": 10,
    "legend": 10,
    "annotation": 9,
}


def _ensure_parent(path: Path) -> Path:
    """Create parent directories for *path* if they do not exist."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_and_close(fig, save_path: Path) -> None:
    """Save a figure to *save_path*, log the action, and close."""
    save_path = _ensure_parent(save_path)
    fig.savefig(str(save_path), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved to %s", save_path)


class Visualiser:
    """
    Centralised visualisation helper for the pre-fraud detection project.

    Every public method creates a self-contained matplotlib figure, saves it
    to *save_path*, and closes the figure to free memory.  Figures are styled
    for direct inclusion in a thesis or journal manuscript.
    """

    # ------------------------------------------------------------------ #
    # 1. Class distribution bar chart
    # ------------------------------------------------------------------ #
    def plot_class_distribution(self, y, save_path):
        """
        Plot a bar chart showing the count of fraud vs non-fraud samples.

        Parameters
        ----------
        y : array-like
            Binary target vector (0 = legitimate, 1 = fraud).
        save_path : str or Path
            Destination file path for the saved figure.
        """
        y = np.asarray(y)
        counts = [int((y == 0).sum()), int((y == 1).sum())]
        labels = ["Non-Fraud (0)", "Fraud (1)"]
        colours = [PALETTE["non_fraud"], PALETTE["fraud"]]

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(labels, counts, color=colours, edgecolor="white", width=0.55)

        for bar, count in zip(bars, counts):
            pct = count / y.shape[0] * 100
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{count:,}\n({pct:.2f}%)",
                ha="center",
                va="bottom",
                fontsize=FONT_SIZES["annotation"],
                fontweight="bold",
            )

        ax.set_ylabel("Number of Transactions", fontsize=FONT_SIZES["axis_label"])
        ax.set_title("Class Distribution", fontsize=FONT_SIZES["title"], fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.tick_params(labelsize=FONT_SIZES["tick"])

        fig.tight_layout()
        _save_and_close(fig, save_path)

    # ------------------------------------------------------------------ #
    # 2. Feature distributions (fraud vs non-fraud histograms)
    # ------------------------------------------------------------------ #
    def plot_feature_distributions(self, df, features, target_col, save_path):
        """
        Plot overlapping histograms for selected features, separated by class.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the features and target column.
        features : list[str]
            Column names to plot.
        target_col : str
            Name of the binary target column.
        save_path : str or Path
            Destination file path for the saved figure.
        """
        n_features = len(features)
        ncols = min(3, n_features)
        nrows = int(np.ceil(n_features / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.atleast_2d(axes)

        legitimate = df[df[target_col] == 0]
        fraud = df[df[target_col] == 1]

        for idx, feat in enumerate(features):
            row, col = divmod(idx, ncols)
            ax = axes[row, col]

            ax.hist(
                legitimate[feat].dropna(),
                bins=50,
                alpha=0.6,
                color=PALETTE["non_fraud"],
                label="Non-Fraud",
                density=True,
            )
            ax.hist(
                fraud[feat].dropna(),
                bins=50,
                alpha=0.6,
                color=PALETTE["fraud"],
                label="Fraud",
                density=True,
            )

            ax.set_title(feat, fontsize=FONT_SIZES["axis_label"], fontweight="bold")
            ax.set_xlabel(feat, fontsize=FONT_SIZES["tick"])
            ax.set_ylabel("Density", fontsize=FONT_SIZES["tick"])
            ax.tick_params(labelsize=FONT_SIZES["tick"])
            ax.legend(fontsize=FONT_SIZES["legend"], loc="upper right")

        # Hide unused subplots
        for idx in range(n_features, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row, col].set_visible(False)

        fig.suptitle(
            "Feature Distributions by Class",
            fontsize=FONT_SIZES["title"],
            fontweight="bold",
            y=1.02,
        )
        fig.tight_layout()
        _save_and_close(fig, save_path)

    # ------------------------------------------------------------------ #
    # 3. Correlation heatmap (top N features)
    # ------------------------------------------------------------------ #
    def plot_correlation_heatmap(self, df, top_n=20, save_path=None):
        """
        Plot a Pearson correlation heatmap for the *top_n* most varying
        numeric features.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with numeric features.
        top_n : int
            Number of features to include (selected by variance).
        save_path : str or Path
            Destination file path for the saved figure.
        """
        numeric_df = df.select_dtypes(include=[np.number])

        # Select the top_n columns by variance
        variances = numeric_df.var().sort_values(ascending=False)
        selected = variances.head(top_n).index.tolist()
        corr = numeric_df[selected].corr()

        fig, ax = plt.subplots(figsize=(max(10, top_n * 0.6), max(8, top_n * 0.5)))

        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        cmap = sns.diverging_palette(240, 10, as_cmap=True)

        sns.heatmap(
            corr,
            mask=mask,
            cmap=cmap,
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.7, "label": "Pearson r"},
            annot=top_n <= 20,
            fmt=".2f" if top_n <= 20 else "",
            annot_kws={"size": 7},
            ax=ax,
        )

        ax.set_title(
            f"Correlation Heatmap (Top {top_n} Features by Variance)",
            fontsize=FONT_SIZES["title"],
            fontweight="bold",
            pad=12,
        )
        ax.tick_params(labelsize=FONT_SIZES["tick"])

        fig.tight_layout()
        _save_and_close(fig, save_path)

    # ------------------------------------------------------------------ #
    # 4. Missing data heatmap
    # ------------------------------------------------------------------ #
    def plot_missing_data_heatmap(self, df, save_path):
        """
        Plot a heatmap indicating missing-value patterns across features.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to inspect for missing values.
        save_path : str or Path
            Destination file path for the saved figure.
        """
        missing_pct = df.isnull().mean() * 100
        cols_with_missing = missing_pct[missing_pct > 0].sort_values(ascending=False)

        if cols_with_missing.empty:
            logger.info("No missing values found; skipping missing-data heatmap.")
            return

        # Limit to top 50 columns for readability
        cols_to_show = cols_with_missing.head(50).index.tolist()
        missing_matrix = df[cols_to_show].isnull().astype(int)

        # Subsample rows if very large
        if len(missing_matrix) > 2000:
            missing_matrix = missing_matrix.sample(n=2000, random_state=42).sort_index()

        fig, ax = plt.subplots(figsize=(max(12, len(cols_to_show) * 0.3), 8))

        sns.heatmap(
            missing_matrix.T,
            cmap=["#ecf0f1", PALETTE["fraud"]],
            cbar_kws={"label": "Missing", "ticks": [0, 1]},
            yticklabels=True,
            xticklabels=False,
            ax=ax,
        )

        ax.set_title(
            "Missing Data Pattern",
            fontsize=FONT_SIZES["title"],
            fontweight="bold",
        )
        ax.set_xlabel("Samples (rows)", fontsize=FONT_SIZES["axis_label"])
        ax.set_ylabel("Features", fontsize=FONT_SIZES["axis_label"])
        ax.tick_params(axis="y", labelsize=8)

        fig.tight_layout()
        _save_and_close(fig, save_path)

    # ------------------------------------------------------------------ #
    # 5. Temporal patterns (fraud rate over time, by hour, by day)
    # ------------------------------------------------------------------ #
    def plot_temporal_patterns(self, df, time_col, target_col, save_path):
        """
        Plot three panels:
        a) Fraud rate over time (rolling window)
        b) Fraud rate by hour of day
        c) Fraud rate by day of week

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with at least the time and target columns.
        time_col : str
            Name of the time column (seconds from reference, e.g. TransactionDT).
        target_col : str
            Name of the binary target column.
        save_path : str or Path
            Destination file path for the saved figure.
        """
        tmp = df[[time_col, target_col]].copy().sort_values(time_col)

        # Derive hour and day if not present
        tmp["hour_of_day"] = (tmp[time_col] / 3600) % 24
        tmp["day_of_week"] = (tmp[time_col] / 86400) % 7

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # (a) Rolling fraud rate
        window = max(1, len(tmp) // 200)
        tmp["rolling_fraud"] = tmp[target_col].rolling(window, min_periods=1).mean()
        ax = axes[0]
        ax.plot(
            tmp[time_col].values,
            tmp["rolling_fraud"].values,
            color=PALETTE["fraud"],
            linewidth=0.8,
            alpha=0.85,
        )
        ax.set_title("Fraud Rate Over Time", fontsize=FONT_SIZES["axis_label"], fontweight="bold")
        ax.set_xlabel(f"{time_col} (seconds)", fontsize=FONT_SIZES["tick"])
        ax.set_ylabel("Fraud Rate (rolling)", fontsize=FONT_SIZES["tick"])
        ax.tick_params(labelsize=FONT_SIZES["tick"])
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))

        # (b) Fraud rate by hour of day
        hourly = tmp.groupby(tmp["hour_of_day"].astype(int))[target_col].mean()
        ax = axes[1]
        ax.bar(
            hourly.index,
            hourly.values,
            color=PALETTE["accent"],
            edgecolor="white",
        )
        ax.set_title("Fraud Rate by Hour of Day", fontsize=FONT_SIZES["axis_label"], fontweight="bold")
        ax.set_xlabel("Hour", fontsize=FONT_SIZES["tick"])
        ax.set_ylabel("Fraud Rate", fontsize=FONT_SIZES["tick"])
        ax.set_xticks(range(0, 24, 2))
        ax.tick_params(labelsize=FONT_SIZES["tick"])
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))

        # (c) Fraud rate by day of week
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        daily = tmp.groupby(tmp["day_of_week"].astype(int))[target_col].mean()
        ax = axes[2]
        ax.bar(
            [day_names[d] if d < 7 else str(d) for d in daily.index],
            daily.values,
            color=PALETTE["highlight"],
            edgecolor="white",
        )
        ax.set_title("Fraud Rate by Day of Week", fontsize=FONT_SIZES["axis_label"], fontweight="bold")
        ax.set_xlabel("Day", fontsize=FONT_SIZES["tick"])
        ax.set_ylabel("Fraud Rate", fontsize=FONT_SIZES["tick"])
        ax.tick_params(labelsize=FONT_SIZES["tick"])
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))

        fig.suptitle(
            "Temporal Fraud Patterns",
            fontsize=FONT_SIZES["title"],
            fontweight="bold",
            y=1.03,
        )
        fig.tight_layout()
        _save_and_close(fig, save_path)

    # ------------------------------------------------------------------ #
    # 6. ROC curves (multiple models)
    # ------------------------------------------------------------------ #
    def plot_roc_curves(self, results_dict, save_path):
        """
        Overlay ROC curves for multiple models on a single figure.

        Parameters
        ----------
        results_dict : dict[str, tuple(array, array)]
            Mapping of *model_name* -> *(y_true, y_prob)*.
        save_path : str or Path
            Destination file path for the saved figure.
        """
        fig, ax = plt.subplots(figsize=(7, 6))

        for idx, (model_name, (y_true, y_prob)) in enumerate(results_dict.items()):
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            colour = MODEL_COLOURS[idx % len(MODEL_COLOURS)]
            ax.plot(
                fpr, tpr,
                color=colour,
                linewidth=2,
                label=f"{model_name} (AUC = {roc_auc:.4f})",
            )

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        ax.set_xlabel("False Positive Rate", fontsize=FONT_SIZES["axis_label"])
        ax.set_ylabel("True Positive Rate", fontsize=FONT_SIZES["axis_label"])
        ax.set_title(
            "Receiver Operating Characteristic (ROC) Curves",
            fontsize=FONT_SIZES["title"],
            fontweight="bold",
        )
        ax.legend(loc="lower right", fontsize=FONT_SIZES["legend"], frameon=True)
        ax.tick_params(labelsize=FONT_SIZES["tick"])

        fig.tight_layout()
        _save_and_close(fig, save_path)

    # ------------------------------------------------------------------ #
    # 7. Precision-Recall curves (multiple models)
    # ------------------------------------------------------------------ #
    def plot_pr_curves(self, results_dict, save_path):
        """
        Overlay Precision-Recall curves for multiple models.

        Parameters
        ----------
        results_dict : dict[str, tuple(array, array)]
            Mapping of *model_name* -> *(y_true, y_prob)*.
        save_path : str or Path
            Destination file path for the saved figure.
        """
        fig, ax = plt.subplots(figsize=(7, 6))

        for idx, (model_name, (y_true, y_prob)) in enumerate(results_dict.items()):
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)
            colour = MODEL_COLOURS[idx % len(MODEL_COLOURS)]
            ax.plot(
                recall, precision,
                color=colour,
                linewidth=2,
                label=f"{model_name} (AP = {ap:.4f})",
            )

        # Baseline: fraction of positives
        first_y = list(results_dict.values())[0][0]
        baseline = np.mean(first_y)
        ax.axhline(y=baseline, color="grey", linestyle="--", linewidth=1, alpha=0.6, label=f"Baseline ({baseline:.4f})")

        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        ax.set_xlabel("Recall", fontsize=FONT_SIZES["axis_label"])
        ax.set_ylabel("Precision", fontsize=FONT_SIZES["axis_label"])
        ax.set_title(
            "Precision-Recall Curves",
            fontsize=FONT_SIZES["title"],
            fontweight="bold",
        )
        ax.legend(loc="upper right", fontsize=FONT_SIZES["legend"], frameon=True)
        ax.tick_params(labelsize=FONT_SIZES["tick"])

        fig.tight_layout()
        _save_and_close(fig, save_path)

    # ------------------------------------------------------------------ #
    # 8. Ablation curve (AUC-ROC vs stage with CI bands)
    # ------------------------------------------------------------------ #
    def plot_ablation_curve(self, stages, metrics, save_path):
        """
        Plot AUC-ROC at successive ablation stages with confidence interval
        bands.

        Parameters
        ----------
        stages : list[str]
            Ordered stage labels (e.g. ["Full", "-card_id", "-address", ...]).
        metrics : list[dict]
            One dict per stage with keys ``"mean"``, ``"lower"``, ``"upper"``
            representing the AUC-ROC point estimate and 95 % CI bounds.
        save_path : str or Path
            Destination file path for the saved figure.
        """
        means = [m["mean"] for m in metrics]
        lowers = [m["lower"] for m in metrics]
        uppers = [m["upper"] for m in metrics]
        x = np.arange(len(stages))

        fig, ax = plt.subplots(figsize=(max(8, len(stages) * 1.1), 5))

        ax.fill_between(x, lowers, uppers, color=PALETTE["ci_band"], alpha=0.4, label="95% CI")
        ax.plot(x, means, "o-", color=PALETTE["primary"], linewidth=2, markersize=7, label="AUC-ROC")

        # Annotate each point
        for i, (m, lo, hi) in enumerate(zip(means, lowers, uppers)):
            ax.annotate(
                f"{m:.4f}",
                (x[i], m),
                textcoords="offset points",
                xytext=(0, 12),
                ha="center",
                fontsize=FONT_SIZES["annotation"],
                fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(stages, rotation=30, ha="right", fontsize=FONT_SIZES["tick"])
        ax.set_ylabel("AUC-ROC", fontsize=FONT_SIZES["axis_label"])
        ax.set_title(
            "Ablation Study: AUC-ROC by Feature Removal Stage",
            fontsize=FONT_SIZES["title"],
            fontweight="bold",
        )
        ax.legend(fontsize=FONT_SIZES["legend"], loc="best")
        ax.tick_params(labelsize=FONT_SIZES["tick"])

        fig.tight_layout()
        _save_and_close(fig, save_path)

    # ------------------------------------------------------------------ #
    # 9. Ablation waterfall chart
    # ------------------------------------------------------------------ #
    def plot_ablation_waterfall(self, stage_names, deltas, save_path):
        """
        Waterfall chart showing the marginal AUC-ROC change when each
        feature group is removed.

        Parameters
        ----------
        stage_names : list[str]
            Labels for each ablation step.
        deltas : list[float]
            AUC-ROC change (positive = improvement, negative = degradation)
            for each step.
        save_path : str or Path
            Destination file path for the saved figure.
        """
        fig, ax = plt.subplots(figsize=(max(8, len(stage_names) * 1.0), 5))

        cumulative = 0.0
        for i, (name, delta) in enumerate(zip(stage_names, deltas)):
            colour = PALETTE["secondary"] if delta >= 0 else PALETTE["fraud"]
            ax.bar(
                i, delta, bottom=cumulative,
                color=colour, edgecolor="white", width=0.6,
            )
            # Value label
            label_y = cumulative + delta / 2
            ax.text(
                i, label_y,
                f"{delta:+.4f}",
                ha="center", va="center",
                fontsize=FONT_SIZES["annotation"],
                fontweight="bold",
                color="white" if abs(delta) > 0.005 else "black",
            )
            cumulative += delta

        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(stage_names)))
        ax.set_xticklabels(stage_names, rotation=35, ha="right", fontsize=FONT_SIZES["tick"])
        ax.set_ylabel("AUC-ROC Change (delta)", fontsize=FONT_SIZES["axis_label"])
        ax.set_title(
            "Ablation Waterfall: Contribution of Each Feature Group",
            fontsize=FONT_SIZES["title"],
            fontweight="bold",
        )
        ax.tick_params(labelsize=FONT_SIZES["tick"])

        fig.tight_layout()
        _save_and_close(fig, save_path)

    # ------------------------------------------------------------------ #
    # 10. Feature importance (horizontal bar)
    # ------------------------------------------------------------------ #
    def plot_feature_importance(self, importances, top_n=20, save_path=None):
        """
        Horizontal bar chart of the top-N most important features.

        Parameters
        ----------
        importances : pd.Series or dict
            Mapping of feature name to importance score.
        top_n : int
            Number of features to display.
        save_path : str or Path
            Destination file path for the saved figure.
        """
        if isinstance(importances, dict):
            importances = pd.Series(importances)

        importances = importances.sort_values(ascending=False).head(top_n)
        # Reverse so highest is at the top of the horizontal bar chart
        importances = importances.sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))

        colours = sns.color_palette("viridis", n_colors=len(importances))
        ax.barh(
            importances.index,
            importances.values,
            color=colours,
            edgecolor="white",
            height=0.7,
        )

        # Value annotations
        max_val = importances.max()
        for i, (feat, val) in enumerate(importances.items()):
            ax.text(
                val + max_val * 0.01,
                i,
                f"{val:.4f}",
                va="center",
                fontsize=FONT_SIZES["annotation"],
            )

        ax.set_xlabel("Importance Score", fontsize=FONT_SIZES["axis_label"])
        ax.set_title(
            f"Top {top_n} Feature Importances",
            fontsize=FONT_SIZES["title"],
            fontweight="bold",
        )
        ax.tick_params(labelsize=FONT_SIZES["tick"])

        fig.tight_layout()
        _save_and_close(fig, save_path)

    # ------------------------------------------------------------------ #
    # 11. SHAP summary plot
    # ------------------------------------------------------------------ #
    def plot_shap_summary(self, shap_values, X, save_path):
        """
        Generate a SHAP beeswarm summary plot.

        This wraps the ``shap`` library's summary_plot so that the figure
        is saved consistently with the rest of the module's style.

        Parameters
        ----------
        shap_values : np.ndarray
            SHAP values array of shape (n_samples, n_features).
        X : pd.DataFrame
            Feature matrix matching *shap_values*.
        save_path : str or Path
            Destination file path for the saved figure.
        """
        try:
            import shap
        except ImportError:
            logger.warning(
                "shap package not installed; skipping SHAP summary plot. "
                "Install with: pip install shap"
            )
            return

        save_path = _ensure_parent(save_path)

        fig, ax = plt.subplots(figsize=(10, max(6, X.shape[1] * 0.25)))
        shap.summary_plot(
            shap_values,
            X,
            plot_type="dot",
            show=False,
            max_display=30,
        )

        # summary_plot creates its own figure; grab the current one
        current_fig = plt.gcf()
        current_fig.suptitle(
            "SHAP Feature Importance (Summary Plot)",
            fontsize=FONT_SIZES["title"],
            fontweight="bold",
            y=1.02,
        )
        current_fig.tight_layout()
        current_fig.savefig(str(save_path), dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(current_fig)
        # Also close the fig we created if it is different
        if current_fig is not fig:
            plt.close(fig)

        logger.info("Figure saved to %s", save_path)

    # ------------------------------------------------------------------ #
    # 12. Drift score distributions
    # ------------------------------------------------------------------ #
    def plot_drift_scores(self, drift_df, save_path):
        """
        Plot distributions of drift scores for fraud vs non-fraud
        transactions.

        Parameters
        ----------
        drift_df : pd.DataFrame
            Must contain columns ``"drift_score"`` and ``"label"``
            (0 = non-fraud, 1 = fraud).
        save_path : str or Path
            Destination file path for the saved figure.
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        for label_val, label_name, colour in [
            (0, "Non-Fraud", PALETTE["non_fraud"]),
            (1, "Fraud", PALETTE["fraud"]),
        ]:
            subset = drift_df[drift_df["label"] == label_val]["drift_score"]
            ax.hist(
                subset,
                bins=60,
                alpha=0.55,
                color=colour,
                label=label_name,
                density=True,
                edgecolor="white",
                linewidth=0.5,
            )
            # Overlay KDE
            if len(subset) > 2:
                try:
                    sns.kdeplot(subset, ax=ax, color=colour, linewidth=2)
                except Exception:
                    pass  # gracefully degrade if KDE fails

        ax.set_xlabel("Drift Score", fontsize=FONT_SIZES["axis_label"])
        ax.set_ylabel("Density", fontsize=FONT_SIZES["axis_label"])
        ax.set_title(
            "Behavioural Drift Score Distributions",
            fontsize=FONT_SIZES["title"],
            fontweight="bold",
        )
        ax.legend(fontsize=FONT_SIZES["legend"], frameon=True)
        ax.tick_params(labelsize=FONT_SIZES["tick"])

        fig.tight_layout()
        _save_and_close(fig, save_path)

    # ------------------------------------------------------------------ #
    # 13. Lead-time analysis
    # ------------------------------------------------------------------ #
    def plot_lead_time_analysis(self, lead_times, auc_scores, save_path):
        """
        Plot AUC-ROC as a function of prediction lead time.

        Parameters
        ----------
        lead_times : list[int or float]
            Lead-time windows (e.g. number of days or hours ahead).
        auc_scores : list[float]
            Corresponding AUC-ROC values.
        save_path : str or Path
            Destination file path for the saved figure.
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            lead_times, auc_scores,
            "o-",
            color=PALETTE["primary"],
            linewidth=2,
            markersize=8,
            markerfacecolor=PALETTE["accent"],
            markeredgecolor=PALETTE["primary"],
            markeredgewidth=1.5,
        )

        # Annotate each point
        for lt, score in zip(lead_times, auc_scores):
            ax.annotate(
                f"{score:.4f}",
                (lt, score),
                textcoords="offset points",
                xytext=(0, 12),
                ha="center",
                fontsize=FONT_SIZES["annotation"],
                fontweight="bold",
            )

        ax.set_xlabel("Lead Time (units)", fontsize=FONT_SIZES["axis_label"])
        ax.set_ylabel("AUC-ROC", fontsize=FONT_SIZES["axis_label"])
        ax.set_title(
            "Pre-Fraud Detection: AUC-ROC at Different Lead Times",
            fontsize=FONT_SIZES["title"],
            fontweight="bold",
        )
        ax.tick_params(labelsize=FONT_SIZES["tick"])

        fig.tight_layout()
        _save_and_close(fig, save_path)

    # ------------------------------------------------------------------ #
    # 14. Model comparison table rendered as figure
    # ------------------------------------------------------------------ #
    def plot_model_comparison_table(self, comparison_df, save_path):
        """
        Render a DataFrame as a styled table figure suitable for a thesis.

        Parameters
        ----------
        comparison_df : pd.DataFrame
            Table of model comparison metrics (e.g. AUC-ROC, AP, F1, etc.).
            The index or first column should be the model name.
        save_path : str or Path
            Destination file path for the saved figure.
        """
        n_rows, n_cols = comparison_df.shape
        fig_width = max(8, n_cols * 1.8)
        fig_height = max(2, (n_rows + 1) * 0.5)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis("off")

        # Build cell text and colours
        cell_text = []
        cell_colours = []
        for _, row in comparison_df.iterrows():
            row_text = []
            row_colours = []
            for val in row:
                if isinstance(val, float):
                    row_text.append(f"{val:.4f}")
                else:
                    row_text.append(str(val))
                row_colours.append("#ffffff")
            cell_text.append(row_text)
            cell_colours.append(row_colours)

        col_labels = comparison_df.columns.tolist()

        table = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            rowLabels=comparison_df.index.tolist(),
            cellColours=cell_colours,
            rowColours=[PALETTE["ci_band"]] * n_rows,
            colColours=[PALETTE["primary"]] * n_cols,
            cellLoc="center",
            loc="center",
        )

        table.auto_set_font_size(False)
        table.set_fontsize(FONT_SIZES["tick"])
        table.scale(1.0, 1.6)

        # Style header cells
        for (row_idx, col_idx), cell in table.get_celld().items():
            if row_idx == 0:
                cell.set_text_props(color="white", fontweight="bold")
            cell.set_edgecolor("#bdc3c7")

        ax.set_title(
            "Model Comparison",
            fontsize=FONT_SIZES["title"],
            fontweight="bold",
            pad=20,
        )

        fig.tight_layout()
        _save_and_close(fig, save_path)


# ------------------------------------------------------------------
# CLI smoke test
# ------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    out_dir = PROJECT_ROOT / "outputs" / "figures" / "_smoke_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    vis = Visualiser()
    rng = np.random.default_rng(42)

    # 1. Class distribution
    y_demo = np.concatenate([np.zeros(9500), np.ones(500)])
    vis.plot_class_distribution(y_demo, out_dir / "class_distribution.png")

    # 6. ROC curves (synthetic)
    demo_results = {
        "Model A": (y_demo, rng.random(len(y_demo))),
        "Model B": (y_demo, rng.random(len(y_demo))),
    }
    vis.plot_roc_curves(demo_results, out_dir / "roc_curves.png")

    # 7. PR curves
    vis.plot_pr_curves(demo_results, out_dir / "pr_curves.png")

    # 8. Ablation curve
    stages = ["Full", "-card_id", "-address", "-match", "-count", "-vesta"]
    metrics = [
        {"mean": 0.95, "lower": 0.94, "upper": 0.96},
        {"mean": 0.93, "lower": 0.92, "upper": 0.94},
        {"mean": 0.91, "lower": 0.89, "upper": 0.93},
        {"mean": 0.88, "lower": 0.86, "upper": 0.90},
        {"mean": 0.85, "lower": 0.83, "upper": 0.87},
        {"mean": 0.80, "lower": 0.77, "upper": 0.83},
    ]
    vis.plot_ablation_curve(stages, metrics, out_dir / "ablation_curve.png")

    # 9. Ablation waterfall
    vis.plot_ablation_waterfall(
        stages[1:],
        [-0.02, -0.02, -0.03, -0.03, -0.05],
        out_dir / "ablation_waterfall.png",
    )

    # 10. Feature importance
    importances = pd.Series(rng.random(15), index=[f"feat_{i}" for i in range(15)])
    vis.plot_feature_importance(importances, top_n=10, save_path=out_dir / "feature_importance.png")

    # 13. Lead-time analysis
    vis.plot_lead_time_analysis(
        [1, 3, 7, 14, 30],
        [0.92, 0.89, 0.85, 0.81, 0.75],
        out_dir / "lead_time.png",
    )

    # 14. Model comparison table
    comp_df = pd.DataFrame({
        "AUC-ROC": [0.95, 0.93, 0.88],
        "AP": [0.72, 0.68, 0.55],
        "F1": [0.65, 0.61, 0.50],
    }, index=["XGBoost", "LightGBM", "Logistic Reg."])
    vis.plot_model_comparison_table(comp_df, out_dir / "model_comparison.png")

    logger.info("Smoke test complete. Figures saved to %s", out_dir)
