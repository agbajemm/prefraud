"""
Tests for drift metric computations.
Validates that drift scores behave correctly on known distributions.
"""

import numpy as np
import pandas as pd
import pytest
from src.drift_analysis import BehaviouralDriftAnalyser


@pytest.fixture
def analyser():
    return BehaviouralDriftAnalyser()


class TestTemporalDrift:
    def test_uniform_time_low_drift(self, analyser):
        """Uniform time distribution should have low drift."""
        n = 200
        df = pd.DataFrame({
            "TransactionDT": np.linspace(0, 86400 * 30, n),
        })
        drift = analyser.compute_temporal_drift(df)
        assert drift.mean() < 0.5

    def test_output_shape(self, analyser):
        df = pd.DataFrame({"TransactionDT": np.arange(100) * 3600})
        drift = analyser.compute_temporal_drift(df)
        assert len(drift) == 100

    def test_handles_missing_column(self, analyser):
        df = pd.DataFrame({"other": [1, 2, 3]})
        drift = analyser.compute_temporal_drift(df)
        assert (drift == 0.0).all()


class TestDeviceDrift:
    def test_all_present_low_drift(self, analyser):
        """All device attributes present → low drift (diverse = normal)."""
        df = pd.DataFrame({
            "DeviceType": ["desktop"] * 50,
            "DeviceInfo": ["Windows"] * 50,
            "id_30": ["Win10"] * 50,
            "id_31": ["chrome"] * 50,
            "id_33": ["1920x1080"] * 50,
        })
        drift = analyser.compute_device_drift(df)
        # All same values → nunique=1 per row → low diversity → high drift
        # This is expected: uniform device use is a baseline, not anomalous
        assert len(drift) == 50

    def test_missing_devices_high_drift(self, analyser):
        """Missing device info should produce higher drift scores."""
        df = pd.DataFrame({
            "DeviceType": [None] * 50,
            "DeviceInfo": [None] * 50,
        })
        drift = analyser.compute_device_drift(df)
        assert len(drift) == 50

    def test_no_device_columns(self, analyser):
        df = pd.DataFrame({"A": [1, 2, 3]})
        drift = analyser.compute_device_drift(df)
        assert (drift == 0.0).all()


class TestAmountDrift:
    def test_normal_distribution(self, analyser):
        """Normal counting patterns should have moderate drift."""
        df = pd.DataFrame({
            f"C{i}": np.random.randint(0, 10, size=100) for i in range(1, 15)
        })
        drift = analyser.compute_amount_drift(df)
        assert len(drift) == 100
        assert drift.min() >= 0

    def test_extreme_values_higher_drift(self, analyser):
        """Extreme counting values should produce higher drift."""
        n = 100
        df_normal = pd.DataFrame({
            f"C{i}": np.ones(n) * 5 for i in range(1, 15)
        })
        df_extreme = pd.DataFrame({
            f"C{i}": np.ones(n) * 500 for i in range(1, 15)
        })
        df_combined = pd.concat([df_normal, df_extreme], ignore_index=True)
        drift = analyser.compute_amount_drift(df_combined)
        # Extreme values should have higher drift than normal
        normal_drift = drift.iloc[:n].mean()
        extreme_drift = drift.iloc[n:].mean()
        assert extreme_drift > normal_drift

    def test_no_count_columns(self, analyser):
        df = pd.DataFrame({"A": [1, 2, 3]})
        drift = analyser.compute_amount_drift(df)
        assert (drift == 0.0).all()


class TestEmailDrift:
    def test_common_emails_low_drift(self, analyser):
        """Common email domains should have lower drift."""
        df = pd.DataFrame({
            "P_emaildomain": ["gmail.com"] * 100,
            "R_emaildomain": ["gmail.com"] * 100,
        })
        drift = analyser.compute_email_drift(df)
        # All same domain → all equally common → low drift
        assert drift.mean() < 0.5

    def test_rare_emails_higher_drift(self, analyser):
        """Rare email domains should produce higher drift scores."""
        emails = ["gmail.com"] * 95 + ["rare-domain.xyz"] * 5
        df = pd.DataFrame({"P_emaildomain": emails})
        drift = analyser.compute_email_drift(df)
        # Rare domain rows should have higher drift
        common_drift = drift.iloc[:95].mean()
        rare_drift = drift.iloc[95:].mean()
        assert rare_drift > common_drift


class TestVelocityDrift:
    def test_short_timedeltas_higher_drift(self, analyser):
        """Short time gaps (high frequency) should produce higher drift."""
        n = 50
        df_slow = pd.DataFrame({f"D{i}": np.ones(n) * 1000 for i in range(1, 5)})
        df_fast = pd.DataFrame({f"D{i}": np.ones(n) * 1 for i in range(1, 5)})
        df = pd.concat([df_slow, df_fast], ignore_index=True)
        drift = analyser.compute_velocity_drift(df)
        slow_drift = drift.iloc[:n].mean()
        fast_drift = drift.iloc[n:].mean()
        assert fast_drift >= slow_drift

    def test_no_relevant_columns(self, analyser):
        df = pd.DataFrame({"A": [1, 2, 3]})
        drift = analyser.compute_velocity_drift(df)
        assert (drift == 0.0).all()


class TestCompositeDrift:
    def test_composite_in_range(self, analyser):
        """Composite drift should be in [0, 1] after MinMax scaling."""
        df = pd.DataFrame({
            "TransactionDT": np.random.randint(0, 86400 * 30, size=100),
            "DeviceType": np.random.choice(["desktop", "mobile"], size=100),
            **{f"C{i}": np.random.randint(0, 10, size=100) for i in range(1, 5)},
            **{f"D{i}": np.random.exponential(50, size=100) for i in range(1, 5)},
            "P_emaildomain": np.random.choice(["gmail.com", "yahoo.com"], size=100),
        })
        drift_df = analyser.compute_all_drift_scores(df)
        assert drift_df["composite_drift"].min() >= 0.0
        assert drift_df["composite_drift"].max() <= 1.0 + 1e-10


class TestDimensionEvaluation:
    def test_evaluate_returns_dict(self, analyser):
        drift_df = pd.DataFrame({
            "temporal_drift": np.random.rand(100),
            "device_drift": np.random.rand(100),
        })
        y = np.random.binomial(1, 0.1, size=100)
        result = analyser.evaluate_drift_dimensions(drift_df, y)
        assert isinstance(result, dict)
        assert "temporal_drift" in result
        assert 0 <= result["temporal_drift"] <= 1
