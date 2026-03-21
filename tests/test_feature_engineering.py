"""
Tests for the Feature Engineering module.
"""

import numpy as np
import pandas as pd
import pytest
from src.feature_engineering import FeatureEngineer


@pytest.fixture
def sample_df():
    """Create a small synthetic dataset mimicking IEEE-CIS structure."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "TransactionID": range(n),
        "TransactionDT": np.random.randint(86400, 86400 * 180, size=n),
        "TransactionAmt": np.random.lognormal(3, 1, size=n),
        "isFraud": np.random.binomial(1, 0.035, size=n),
        "card1": np.random.randint(1000, 9999, size=n),
        "card2": np.random.choice([100, 200, 300, np.nan], size=n),
        "addr1": np.random.randint(100, 500, size=n),
        "addr2": np.random.choice([87, 60, 96], size=n),
        "dist1": np.random.exponential(50, size=n),
        "dist2": np.random.exponential(50, size=n),
        "P_emaildomain": np.random.choice(["gmail.com", "yahoo.com", "hotmail.com", None], size=n),
        "R_emaildomain": np.random.choice(["gmail.com", "yahoo.com", None], size=n),
        "DeviceType": np.random.choice(["desktop", "mobile", None], size=n),
        "DeviceInfo": np.random.choice(["Windows", "iOS", "Android", None], size=n),
        "id_30": np.random.choice(["Windows 10", "iOS 12", None], size=n),
        "id_31": np.random.choice(["chrome", "safari", "firefox", None], size=n),
        "id_33": np.random.choice(["1920x1080", "1366x768", None], size=n),
        **{f"C{i}": np.random.randint(0, 10, size=n) for i in range(1, 15)},
        **{f"M{i}": np.random.choice(["T", "F", None], size=n) for i in range(1, 10)},
        **{f"D{i}": np.random.exponential(100, size=n) for i in range(1, 16)},
        **{f"V{i}": np.random.randn(n) for i in range(1, 20)},
    })


@pytest.fixture
def engineer():
    return FeatureEngineer()


class TestTemporalFeatures:
    def test_creates_hour_of_day(self, engineer, sample_df):
        result = engineer.create_temporal_features(sample_df.copy())
        assert "hour_of_day" in result.columns
        assert result["hour_of_day"].min() >= 0
        assert result["hour_of_day"].max() < 24

    def test_creates_day_of_week(self, engineer, sample_df):
        result = engineer.create_temporal_features(sample_df.copy())
        assert "day_of_week" in result.columns
        assert result["day_of_week"].min() >= 0
        assert result["day_of_week"].max() < 7

    def test_creates_is_weekend(self, engineer, sample_df):
        result = engineer.create_temporal_features(sample_df.copy())
        assert "is_weekend" in result.columns
        assert set(result["is_weekend"].unique()).issubset({0, 1})

    def test_creates_is_night(self, engineer, sample_df):
        result = engineer.create_temporal_features(sample_df.copy())
        assert "is_night" in result.columns
        assert set(result["is_night"].unique()).issubset({0, 1})

    def test_handles_missing_time_column(self, engineer):
        df = pd.DataFrame({"A": [1, 2, 3]})
        result = engineer.create_temporal_features(df)
        assert "hour_of_day" not in result.columns


class TestVelocityFeatures:
    def test_creates_d_summary(self, engineer, sample_df):
        result = engineer.create_velocity_features(sample_df.copy())
        assert "D_mean" in result.columns
        assert "D_std" in result.columns
        assert "D_range" in result.columns

    def test_d_range_non_negative(self, engineer, sample_df):
        result = engineer.create_velocity_features(sample_df.copy())
        assert (result["D_range"] >= 0).all()


class TestCountFeatures:
    def test_creates_c_aggregates(self, engineer, sample_df):
        result = engineer.create_count_features(sample_df.copy())
        assert "C_sum" in result.columns
        assert "C_mean" in result.columns
        assert "C_max" in result.columns


class TestVestaSummary:
    def test_creates_vesta_summary(self, engineer, sample_df):
        result = engineer.create_vesta_summary_features(sample_df.copy())
        assert "V_mean" in result.columns
        assert "V_std" in result.columns
        assert "V_null_count" in result.columns


class TestFeatureClassification:
    def test_classifies_direct(self, engineer):
        importances = {"TransactionAmt": 0.5, "card1": 0.3, "D1": 0.1}
        result = engineer.classify_features(importances, list(importances.keys()))
        direct = result[result["category"] == "DIRECT"]
        assert "TransactionAmt" in direct["feature"].values
        assert "card1" in direct["feature"].values

    def test_classifies_indirect(self, engineer):
        importances = {"D1": 0.1, "DeviceType": 0.2}
        result = engineer.classify_features(importances, list(importances.keys()))
        indirect = result[result["category"] == "INDIRECT"]
        assert "D1" in indirect["feature"].values
        assert "DeviceType" in indirect["feature"].values

    def test_classifies_ambiguous(self, engineer):
        importances = {"unknown_feature": 0.05}
        result = engineer.classify_features(importances, list(importances.keys()))
        ambig = result[result["category"] == "AMBIGUOUS"]
        assert "unknown_feature" in ambig["feature"].values


class TestFeatureGroupHelpers:
    def test_get_direct_features(self, engineer, sample_df):
        direct = engineer.get_direct_features(sample_df.columns.tolist())
        assert "TransactionAmt" in direct
        assert "card1" in direct
        assert "D1" not in direct

    def test_get_indirect_features(self, engineer, sample_df):
        indirect = engineer.get_indirect_features(sample_df.columns.tolist())
        assert "D1" in indirect
        assert "DeviceType" in indirect
        assert "TransactionAmt" not in indirect

    def test_get_feature_group(self, engineer, sample_df):
        temporal = engineer.get_feature_group("temporal", sample_df.columns.tolist())
        assert "TransactionDT" in temporal
        assert "D1" in temporal
        assert "card1" not in temporal


class TestAllDerivedFeatures:
    def test_pipeline_runs(self, engineer, sample_df):
        result = engineer.create_all_derived_features(sample_df.copy())
        # Should have more columns than original
        assert result.shape[1] > sample_df.shape[1]
        # Original columns preserved
        for col in sample_df.columns:
            assert col in result.columns
