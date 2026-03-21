"""
Feature Engineering Module
Creates derived features, classifies feature importance, and manages
the feature taxonomy (direct vs indirect indicators).
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class FeatureEngineer:
    """
    Creates derived temporal, velocity, and interaction features
    from the IEEE-CIS dataset and manages the feature taxonomy.
    """

    # Feature groups for ablation study
    DIRECT_INDICATOR_GROUPS = {
        "transaction_amount": ["TransactionAmt"],
        "card_identifiers": ["card1", "card2", "card3", "card4", "card5", "card6"],
        "address_distance": ["addr1", "addr2", "dist1", "dist2"],
        "match_features": [f"M{i}" for i in range(1, 10)],
        "counting_features": [f"C{i}" for i in range(1, 15)],
    }

    BEHAVIOURAL_GROUPS = {
        "temporal": ["TransactionDT"] + [f"D{i}" for i in range(1, 16)],
        "device": ["DeviceType", "DeviceInfo", "id_30", "id_31", "id_33"],
        "email": ["P_emaildomain", "R_emaildomain"],
        "vesta": [f"V{i}" for i in range(1, 340)],
        "identity": [f"id_{i}" for i in range(1, 39) if i not in (30, 31, 33)],
    }

    def __init__(self):
        self.feature_classification = None

    # ------------------------------------------------------------------
    # Derived feature creation
    # ------------------------------------------------------------------
    def create_temporal_features(self, df):
        """Derive hour-of-day, day-of-week, and time-since-last from TransactionDT."""
        if "TransactionDT" not in df.columns:
            return df

        # TransactionDT is seconds from a reference point
        df["hour_of_day"] = (df["TransactionDT"] / 3600) % 24
        df["day_of_week"] = (df["TransactionDT"] / 86400) % 7
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_night"] = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 5)).astype(int)

        logger.info("Created temporal derived features: hour_of_day, day_of_week, is_weekend, is_night")
        return df

    def create_velocity_features(self, df):
        """
        Compute transaction frequency / velocity features.
        Uses D-series timedelta columns as proxies.
        """
        d_cols = [f"D{i}" for i in range(1, 16) if f"D{i}" in df.columns]
        if len(d_cols) >= 2:
            df["D_mean"] = df[d_cols].mean(axis=1)
            df["D_std"] = df[d_cols].std(axis=1)
            df["D_range"] = df[d_cols].max(axis=1) - df[d_cols].min(axis=1)
            logger.info("Created velocity features: D_mean, D_std, D_range")

        return df

    def create_count_features(self, df):
        """Aggregate counting features into summary statistics."""
        c_cols = [f"C{i}" for i in range(1, 15) if f"C{i}" in df.columns]
        if len(c_cols) >= 2:
            df["C_sum"] = df[c_cols].sum(axis=1)
            df["C_mean"] = df[c_cols].mean(axis=1)
            df["C_max"] = df[c_cols].max(axis=1)
            logger.info("Created count aggregate features: C_sum, C_mean, C_max")
        return df

    def create_vesta_summary_features(self, df):
        """Summarise Vesta (V) columns into aggregate statistics."""
        v_cols = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]
        if len(v_cols) >= 10:
            df["V_mean"] = df[v_cols].mean(axis=1)
            df["V_std"] = df[v_cols].std(axis=1)
            df["V_null_count"] = df[v_cols].isnull().sum(axis=1)
            logger.info(
                "Created Vesta summary features from %d V-columns", len(v_cols)
            )
        return df

    def create_all_derived_features(self, df):
        """Run the full derived-feature pipeline."""
        df = self.create_temporal_features(df)
        df = self.create_velocity_features(df)
        df = self.create_count_features(df)
        df = self.create_vesta_summary_features(df)
        return df

    # ------------------------------------------------------------------
    # Feature importance ranking
    # ------------------------------------------------------------------
    def compute_mutual_information(self, X, y):
        """Mutual information between each feature and the target."""
        # Use only numeric and finite values
        X_clean = X.select_dtypes(include=[np.number]).copy()
        X_clean = X_clean.fillna(0)
        mi = mutual_info_classif(X_clean, y, random_state=42, n_neighbors=5)
        return pd.Series(mi, index=X_clean.columns).sort_values(ascending=False)

    def compute_permutation_importance(self, model, X, y, n_repeats=5):
        """Permutation importance on a fitted model."""
        result = permutation_importance(
            model, X, y,
            n_repeats=n_repeats,
            random_state=42,
            scoring="roc_auc",
        )
        return pd.Series(
            result.importances_mean, index=X.columns
        ).sort_values(ascending=False)

    def classify_features(self, feature_importances, feature_names):
        """
        Classify each feature as DIRECT, INDIRECT, or AMBIGUOUS
        based on which group it belongs to.
        """
        direct_features = set()
        for group_feats in self.DIRECT_INDICATOR_GROUPS.values():
            direct_features.update(group_feats)

        indirect_features = set()
        for group_feats in self.BEHAVIOURAL_GROUPS.values():
            indirect_features.update(group_feats)

        records = []
        for feat in feature_names:
            if feat in direct_features:
                category = "DIRECT"
            elif feat in indirect_features:
                category = "INDIRECT"
            else:
                category = "AMBIGUOUS"

            imp_score = feature_importances.get(feat, 0.0)
            records.append({
                "feature": feat,
                "category": category,
                "importance_score": imp_score,
            })

        self.feature_classification = pd.DataFrame(records).sort_values(
            "importance_score", ascending=False
        )
        return self.feature_classification

    def save_feature_classification(self, df_classification=None):
        """Save feature classification CSV."""
        if df_classification is None:
            df_classification = self.feature_classification
        out_path = PROJECT_ROOT / "data" / "features" / "feature_classification.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_classification.to_csv(out_path, index=False)
        logger.info("Saved feature classification to %s", out_path)

    # ------------------------------------------------------------------
    # Feature set helpers for ablation
    # ------------------------------------------------------------------
    def get_direct_features(self, available_columns):
        """Return list of direct indicator features present in the data."""
        direct = set()
        for group_feats in self.DIRECT_INDICATOR_GROUPS.values():
            direct.update(group_feats)
        return [c for c in available_columns if c in direct]

    def get_indirect_features(self, available_columns):
        """Return list of indirect behavioural features present in the data."""
        indirect = set()
        for group_feats in self.BEHAVIOURAL_GROUPS.values():
            indirect.update(group_feats)
        return [c for c in available_columns if c in indirect]

    def get_feature_group(self, group_name, available_columns):
        """Return features belonging to a named group that are present in data."""
        all_groups = {**self.DIRECT_INDICATOR_GROUPS, **self.BEHAVIOURAL_GROUPS}
        group_feats = all_groups.get(group_name, [])
        return [c for c in available_columns if c in group_feats]
