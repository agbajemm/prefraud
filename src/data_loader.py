"""
Data Loading and Preprocessing Module
Handles IEEE-CIS and Credit Card Fraud datasets with chronological splitting.
"""

import os
import logging
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path=None):
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class DataLoader:
    """
    Loads IEEE-CIS Fraud Detection data, merges transaction and identity
    tables, handles missing values, encodes categoricals, and performs
    chronological train/validation/test splits.
    """

    def __init__(self, config=None):
        self.config = config or load_config()
        self.label_encoders = {}
        self.numerical_medians = {}
        self.categorical_modes = {}
        self.feature_names = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_ieee_data(self):
        """Load and merge IEEE-CIS transaction + identity CSVs."""
        txn_path = PROJECT_ROOT / self.config["data"]["ieee_transaction_path"]
        id_path = PROJECT_ROOT / self.config["data"]["ieee_identity_path"]

        logger.info("Loading transaction data from %s", txn_path)
        txn = pd.read_csv(txn_path)
        logger.info("Transaction shape: %s", txn.shape)

        logger.info("Loading identity data from %s", id_path)
        identity = pd.read_csv(id_path)
        logger.info("Identity shape: %s", identity.shape)

        logger.info("Merging on TransactionID (left join)")
        df = txn.merge(identity, on="TransactionID", how="left")
        logger.info("Merged shape: %s", df.shape)

        self._log_dataset_stats(df)
        return df

    def load_credit_card_data(self):
        """Load the secondary Credit Card Fraud dataset (MLG-ULB)."""
        cc_path = PROJECT_ROOT / self.config["data"]["credit_card_path"]
        logger.info("Loading credit card data from %s", cc_path)
        df = pd.read_csv(cc_path)
        logger.info("Credit card shape: %s", df.shape)
        self._log_dataset_stats(df, target_col="Class")
        return df

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def preprocess(self, df, fit=True):
        """
        Full preprocessing pipeline:
        1. Identify numerical / categorical columns
        2. Handle missing values
        3. Label-encode categoricals
        """
        exclude_cols = {"TransactionID", "isFraud"}
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        cat_cols = df[feature_cols].select_dtypes(include=["object", "category", "string"]).columns.tolist()
        num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

        # --- Missing value imputation ---
        if fit:
            self.numerical_medians = df[num_cols].median().to_dict()
            self.categorical_modes = {
                c: df[c].mode().iloc[0] if not df[c].mode().empty else "missing"
                for c in cat_cols
            }

        for c in num_cols:
            if c in self.numerical_medians:
                df[c] = df[c].fillna(self.numerical_medians[c])
        for c in cat_cols:
            df[c] = df[c].fillna(self.categorical_modes.get(c, "missing"))

        # --- Label encode categoricals ---
        for c in cat_cols:
            df[c] = df[c].astype(str)
            if fit:
                le = LabelEncoder()
                df[c] = le.fit_transform(df[c])
                self.label_encoders[c] = le
            else:
                le = self.label_encoders.get(c)
                if le is not None:
                    # Handle unseen labels gracefully
                    known = set(le.classes_)
                    df[c] = df[c].apply(lambda x: x if x in known else "unknown")
                    le_classes = np.append(le.classes_, "unknown")
                    le.classes_ = le_classes
                    df[c] = le.transform(df[c])
                else:
                    le = LabelEncoder()
                    df[c] = le.fit_transform(df[c])
                    self.label_encoders[c] = le

        self.feature_names = [c for c in df.columns if c not in exclude_cols]
        return df

    # ------------------------------------------------------------------
    # Chronological Splitting
    # ------------------------------------------------------------------
    def chronological_split(self, df, time_col="TransactionDT"):
        """
        Split data chronologically (NOT randomly) to simulate real-world
        deployment where future data is unseen.
        - First 70% by time → training
        - Next 15% by time → validation
        - Final 15% by time → testing
        """
        df = df.sort_values(time_col).reset_index(drop=True)
        n = len(df)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        logger.info(
            "Chronological split: train=%d, val=%d, test=%d",
            len(train_df), len(val_df), len(test_df),
        )
        for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            fraud_rate = split_df["isFraud"].mean() * 100 if "isFraud" in split_df.columns else 0
            logger.info("  %s fraud rate: %.3f%%", name, fraud_rate)

        return train_df, val_df, test_df

    def get_xy(self, df, target_col="isFraud"):
        """Extract feature matrix X and target vector y."""
        exclude = {"TransactionID", target_col}
        feature_cols = [c for c in df.columns if c not in exclude]
        X = df[feature_cols]
        y = df[target_col] if target_col in df.columns else None
        return X, y

    # ------------------------------------------------------------------
    # Save / Load processed data
    # ------------------------------------------------------------------
    def save_processed(self, train_df, val_df, test_df):
        """Persist processed splits to disk."""
        out_dir = PROJECT_ROOT / self.config["data"]["processed_path"]
        os.makedirs(out_dir, exist_ok=True)
        train_df.to_parquet(out_dir / "train.parquet", index=False)
        val_df.to_parquet(out_dir / "val.parquet", index=False)
        test_df.to_parquet(out_dir / "test.parquet", index=False)
        logger.info("Saved processed splits to %s", out_dir)

    def load_processed(self):
        """Load previously saved processed splits."""
        proc_dir = PROJECT_ROOT / self.config["data"]["processed_path"]
        train_df = pd.read_parquet(proc_dir / "train.parquet")
        val_df = pd.read_parquet(proc_dir / "val.parquet")
        test_df = pd.read_parquet(proc_dir / "test.parquet")
        logger.info(
            "Loaded processed splits: train=%d, val=%d, test=%d",
            len(train_df), len(val_df), len(test_df),
        )
        return train_df, val_df, test_df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _log_dataset_stats(self, df, target_col="isFraud"):
        logger.info("Shape: %s", df.shape)
        logger.info("Columns: %d", df.shape[1])
        if target_col in df.columns:
            fraud_count = df[target_col].sum()
            fraud_rate = df[target_col].mean() * 100
            logger.info(
                "Fraud: %d / %d (%.3f%%)", fraud_count, len(df), fraud_rate
            )
        missing_pct = df.isnull().mean() * 100
        high_missing = missing_pct[missing_pct > 50]
        if len(high_missing) > 0:
            logger.info("Features with >50%% missing: %d", len(high_missing))
            for feat, pct in high_missing.items():
                logger.info("  %s: %.1f%%", feat, pct)


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load_ieee_data()
    df = loader.preprocess(df, fit=True)
    train_df, val_df, test_df = loader.chronological_split(df)
    loader.save_processed(train_df, val_df, test_df)
    logger.info("Data loading and preprocessing complete.")
