"""
Behavioural Drift Decomposition Module
Decomposes aggregate behavioural change into component dimensions and
measures each dimension's predictive power for pre-fraud indication.

Drift Dimensions:
    1. Temporal Drift — KL-divergence of transaction time patterns
    2. Device/Channel Drift — Jaccard distance of device sets
    3. Amount Pattern Drift — Wasserstein distance of counting patterns
    4. Email/Identity Drift — frequency of identity feature changes
    5. Velocity Drift — rate of change in transaction frequency
"""

import logging
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance, entropy
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path=None):
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class BehaviouralDriftAnalyser:
    """
    Decomposes aggregate behavioural change into component dimensions
    and measures each dimension's predictive power for pre-fraud indication.
    """

    def __init__(self, config=None):
        self.config = config or load_config()
        self.drift_scores = None
        self.dimension_auc = {}

    # ------------------------------------------------------------------
    # Temporal Drift
    # ------------------------------------------------------------------
    def compute_temporal_drift(self, df, time_col="TransactionDT",
                               baseline_window=30*86400, recent_window=7*86400):
        """
        KL-divergence of transaction time distribution (hour-of-day pattern)
        between a rolling baseline window and recent window.

        Each transaction gets a drift score based on how unusual its local
        temporal pattern is compared to the broader baseline.
        """
        if time_col not in df.columns:
            logger.warning("Column %s not found — skipping temporal drift.", time_col)
            return pd.Series(0.0, index=df.index, name="temporal_drift")

        # Derive hour of day
        hour = (df[time_col] / 3600) % 24

        # Global baseline distribution (24 bins)
        baseline_hist, _ = np.histogram(hour, bins=24, range=(0, 24), density=True)
        baseline_hist = baseline_hist + 1e-10  # Smoothing

        # For each row, compute local distribution from surrounding transactions
        drift_scores = np.zeros(len(df))
        sorted_df = df.sort_values(time_col).reset_index(drop=True)
        sorted_hour = (sorted_df[time_col] / 3600) % 24

        window_size = min(500, len(df) // 10)

        for i in range(len(sorted_df)):
            # Local window around this transaction
            start = max(0, i - window_size)
            end = min(len(sorted_df), i + window_size)
            local_hours = sorted_hour.iloc[start:end]

            local_hist, _ = np.histogram(local_hours, bins=24, range=(0, 24), density=True)
            local_hist = local_hist + 1e-10

            # Jensen-Shannon divergence (symmetric, bounded)
            drift_scores[i] = jensenshannon(baseline_hist, local_hist)

        # Map back to original index
        result = pd.Series(drift_scores, index=sorted_df.index, name="temporal_drift")
        result = result.reindex(df.index, fill_value=0.0)

        logger.info("Temporal drift — mean: %.4f, std: %.4f", result.mean(), result.std())
        return result

    # ------------------------------------------------------------------
    # Device/Channel Drift
    # ------------------------------------------------------------------
    def compute_device_drift(self, df, device_cols=None):
        """
        Measure device/channel diversity per transaction context.
        Uses the number of unique device attributes as a proxy for
        device switching behaviour.
        """
        if device_cols is None:
            device_cols = ["DeviceType", "DeviceInfo", "id_30", "id_31", "id_33"]

        available = [c for c in device_cols if c in df.columns]
        if not available:
            logger.warning("No device columns found — skipping device drift.")
            return pd.Series(0.0, index=df.index, name="device_drift")

        # Number of non-null device attributes (diversity indicator)
        device_present = df[available].notna().sum(axis=1)

        # Uniqueness score: how many distinct values across device features
        device_nunique = df[available].nunique(axis=1)

        # Combined score (normalised)
        raw_score = device_nunique / max(len(available), 1)

        # Invert: having *fewer* device features filled is more suspicious
        drift = 1.0 - raw_score
        drift.name = "device_drift"

        logger.info("Device drift — mean: %.4f, std: %.4f", drift.mean(), drift.std())
        return drift

    # ------------------------------------------------------------------
    # Amount Pattern Drift
    # ------------------------------------------------------------------
    def compute_amount_drift(self, df, count_cols=None):
        """
        Wasserstein distance between baseline and local counting-feature
        distributions as a proxy for amount pattern changes.
        """
        if count_cols is None:
            count_cols = [f"C{i}" for i in range(1, 15)]

        available = [c for c in count_cols if c in df.columns]
        if not available:
            logger.warning("No counting columns found — skipping amount drift.")
            return pd.Series(0.0, index=df.index, name="amount_drift")

        # Compute row-level summary statistics
        c_values = df[available].fillna(0)
        row_mean = c_values.mean(axis=1)
        row_std = c_values.std(axis=1).fillna(0)

        # Global baseline statistics
        global_mean = row_mean.mean()
        global_std = row_mean.std()

        # Z-score distance from global baseline
        drift = ((row_mean - global_mean) / max(global_std, 1e-10)).abs()

        # Add variability component
        global_row_std = row_std.mean()
        std_drift = ((row_std - global_row_std) / max(row_std.std(), 1e-10)).abs()

        combined = (drift + std_drift) / 2.0
        combined.name = "amount_drift"

        logger.info("Amount drift — mean: %.4f, std: %.4f", combined.mean(), combined.std())
        return combined

    # ------------------------------------------------------------------
    # Email/Identity Drift
    # ------------------------------------------------------------------
    def compute_email_drift(self, df, email_cols=None, id_cols=None):
        """
        Binary indicator of unusual email/identity patterns.
        Measures rarity of email domains and identity feature combinations.
        """
        if email_cols is None:
            email_cols = ["P_emaildomain", "R_emaildomain"]
        if id_cols is None:
            id_cols = [f"id_{i}" for i in range(1, 12)]

        available_email = [c for c in email_cols if c in df.columns]
        available_id = [c for c in id_cols if c in df.columns]

        scores = pd.Series(0.0, index=df.index)

        # Email domain rarity
        for col in available_email:
            freq = df[col].value_counts(normalize=True)
            # Rare domains get higher drift scores
            col_score = df[col].map(freq).fillna(1.0)
            scores += (1.0 - col_score)  # Invert: rare = high drift

        # Identity feature missingness (missing identity = suspicious)
        if available_id:
            id_missing = df[available_id].isnull().mean(axis=1)
            scores += id_missing

        # Normalise
        if scores.max() > 0:
            scores = scores / scores.max()

        scores.name = "email_drift"
        logger.info("Email drift — mean: %.4f, std: %.4f", scores.mean(), scores.std())
        return scores

    # ------------------------------------------------------------------
    # Velocity Drift
    # ------------------------------------------------------------------
    def compute_velocity_drift(self, df, time_col="TransactionDT", d_cols=None):
        """
        Rate of change in transaction frequency.
        Uses D-series timedelta features as proxies for transaction velocity.
        """
        if d_cols is None:
            d_cols = [f"D{i}" for i in range(1, 16)]

        available = [c for c in d_cols if c in df.columns]

        if not available and time_col not in df.columns:
            logger.warning("No velocity-relevant columns found.")
            return pd.Series(0.0, index=df.index, name="velocity_drift")

        scores = pd.Series(0.0, index=df.index)

        if available:
            d_values = df[available].fillna(df[available].median())
            # Small timedeltas = high frequency = potentially suspicious
            d_mean = d_values.mean(axis=1)
            global_median = d_mean.median()
            # Low values (short time gaps) get high drift scores
            scores = (global_median - d_mean).clip(lower=0)
            if scores.max() > 0:
                scores = scores / scores.max()

        scores.name = "velocity_drift"
        logger.info("Velocity drift — mean: %.4f, std: %.4f", scores.mean(), scores.std())
        return scores

    # ------------------------------------------------------------------
    # Combined drift computation
    # ------------------------------------------------------------------
    def compute_all_drift_scores(self, df):
        """Compute all five drift dimensions and return a combined DataFrame."""
        temporal = self.compute_temporal_drift(df)
        device = self.compute_device_drift(df)
        amount = self.compute_amount_drift(df)
        email = self.compute_email_drift(df)
        velocity = self.compute_velocity_drift(df)

        drift_df = pd.DataFrame({
            "temporal_drift": temporal,
            "device_drift": device,
            "amount_drift": amount,
            "email_drift": email,
            "velocity_drift": velocity,
        }, index=df.index)

        # Normalise all dimensions to [0, 1]
        scaler = MinMaxScaler()
        drift_df[drift_df.columns] = scaler.fit_transform(drift_df.fillna(0))

        # Composite drift score (equal weighting initially)
        drift_df["composite_drift"] = drift_df.mean(axis=1)

        self.drift_scores = drift_df
        logger.info("Computed all drift scores. Shape: %s", drift_df.shape)
        return drift_df

    # ------------------------------------------------------------------
    # Predictive power analysis per dimension
    # ------------------------------------------------------------------
    def evaluate_drift_dimensions(self, drift_df, y_true):
        """
        Measure AUC-ROC of each drift dimension independently
        against the fraud label.
        """
        results = {}
        for col in drift_df.columns:
            try:
                auc = roc_auc_score(y_true, drift_df[col])
                results[col] = auc
                logger.info("  %s AUC-ROC: %.4f", col, auc)
            except ValueError:
                results[col] = 0.5
                logger.warning("  %s: could not compute AUC (constant or missing)", col)

        self.dimension_auc = results
        return results

    # ------------------------------------------------------------------
    # Lead time analysis
    # ------------------------------------------------------------------
    def lead_time_analysis(self, df, y_true, time_col="TransactionDT",
                           lookback_days=(1, 3, 7, 14, 30)):
        """
        For each confirmed fraud, look back N days and compute drift scores
        to determine if drift signals precede fraud.

        Returns a DataFrame with AUC-ROC at each lead time.
        """
        if time_col not in df.columns:
            logger.warning("No time column — cannot perform lead time analysis.")
            return pd.DataFrame()

        fraud_mask = y_true == 1
        fraud_times = df.loc[fraud_mask, time_col]

        lead_results = []

        for lookback in lookback_days:
            lookback_seconds = lookback * 86400

            # For each fraud transaction, find transactions in the lookback window
            # that belong to the pre-fraud period
            pre_fraud_mask = pd.Series(False, index=df.index)

            for _, fraud_time in fraud_times.items():
                window_start = fraud_time - lookback_seconds
                window_mask = (
                    (df[time_col] >= window_start) &
                    (df[time_col] < fraud_time) &
                    (~fraud_mask)  # Only non-fraud transactions in the window
                )
                pre_fraud_mask = pre_fraud_mask | window_mask

            # Label pre-fraud window transactions as positive
            n_pre_fraud = pre_fraud_mask.sum()
            n_normal = (~fraud_mask & ~pre_fraud_mask).sum()

            if n_pre_fraud == 0 or n_normal == 0:
                logger.info("  Lookback %d days: insufficient data", lookback)
                continue

            # Create binary label: 1=pre-fraud window, 0=normal
            y_lead = pre_fraud_mask.astype(int)
            # Only evaluate on non-fraud transactions
            eval_mask = ~fraud_mask
            y_lead_eval = y_lead[eval_mask]

            drift_df = self.drift_scores
            if drift_df is None:
                logger.warning("Drift scores not computed.")
                break

            drift_eval = drift_df.loc[eval_mask]

            row = {"lookback_days": lookback, "n_pre_fraud": int(n_pre_fraud), "n_normal": int(n_normal)}
            for col in drift_eval.columns:
                try:
                    auc = roc_auc_score(y_lead_eval, drift_eval[col])
                    row[f"auc_{col}"] = auc
                except ValueError:
                    row[f"auc_{col}"] = 0.5

            lead_results.append(row)
            logger.info(
                "  Lookback %d days: n_pre_fraud=%d, composite AUC=%.4f",
                lookback, n_pre_fraud, row.get("auc_composite_drift", 0.5),
            )

        lead_df = pd.DataFrame(lead_results)
        return lead_df

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_results(self, drift_df=None, lead_df=None):
        """Save drift scores and analysis results."""
        out_dir = PROJECT_ROOT / "results" / "tables"
        out_dir.mkdir(parents=True, exist_ok=True)

        if drift_df is not None:
            drift_df.to_csv(out_dir / "drift_scores.csv", index=False)
            logger.info("Saved drift scores.")

        if lead_df is not None:
            lead_df.to_csv(out_dir / "lead_time_analysis.csv", index=False)
            logger.info("Saved lead time analysis.")

        if self.dimension_auc:
            auc_df = pd.DataFrame([self.dimension_auc])
            auc_df.to_csv(out_dir / "drift_dimension_auc.csv", index=False)
            logger.info("Saved drift dimension AUC scores.")

    def generate_plots(self, drift_df, y_true, lead_df=None):
        """Generate drift analysis visualisations."""
        from src.visualisation import Visualiser

        vis = Visualiser()
        fig_dir = PROJECT_ROOT / "results" / "figures" / "drift"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Drift score distributions
        vis.plot_drift_scores(
            drift_df.assign(isFraud=y_true.values),
            fig_dir / "drift_distributions.png",
        )

        # Lead time analysis
        if lead_df is not None and not lead_df.empty:
            auc_cols = [c for c in lead_df.columns if c.startswith("auc_")]
            if "auc_composite_drift" in auc_cols:
                vis.plot_lead_time_analysis(
                    lead_df["lookback_days"].tolist(),
                    lead_df["auc_composite_drift"].tolist(),
                    fig_dir / "lead_time_auc.png",
                )

        logger.info("Drift plots saved to %s", fig_dir)


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    from src.data_loader import DataLoader

    loader = DataLoader()
    train_df, val_df, test_df = loader.load_processed()

    # Use combined train+val for drift analysis
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    y = df["isFraud"]

    analyser = BehaviouralDriftAnalyser()
    drift_df = analyser.compute_all_drift_scores(df)

    logger.info("\nDrift dimension predictive power:")
    dim_auc = analyser.evaluate_drift_dimensions(drift_df, y)

    logger.info("\nLead time analysis:")
    lead_df = analyser.lead_time_analysis(df, y)

    analyser.save_results(drift_df, lead_df)
    analyser.generate_plots(drift_df, y, lead_df)
    logger.info("Drift analysis complete.")
