"""
Tests for Baseline Model and Pre-Fraud Model modules.
Uses small synthetic data to validate model training and evaluation pipelines.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


@pytest.fixture
def synthetic_data():
    """Generate a small binary classification dataset."""
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        weights=[0.95, 0.05],
        random_state=42,
    )
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df["isFraud"] = y
    df["TransactionID"] = range(len(df))

    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=y, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["isFraud"], random_state=42)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


class TestBaselineModelBasics:
    """Test that the baseline model module imports and its key classes exist."""

    def test_import(self):
        from src.baseline_model import BaselineModel
        assert BaselineModel is not None

    def test_instantiation(self):
        from src.baseline_model import BaselineModel
        model = BaselineModel()
        assert model is not None


class TestFeatureAblationBasics:
    """Test the feature ablation engine structure."""

    def test_import(self):
        from src.feature_ablation import FeatureAblationEngine
        assert FeatureAblationEngine is not None

    def test_ablation_stages_defined(self):
        from src.feature_ablation import FeatureAblationEngine
        engine = FeatureAblationEngine()
        assert len(engine.ABLATION_STAGES) == 8

    def test_ablation_stage_names(self):
        from src.feature_ablation import FeatureAblationEngine
        engine = FeatureAblationEngine()
        names = [s["name"] for s in engine.ABLATION_STAGES]
        assert "Stage 0: Full Model" in names
        assert "Stage 7: Behavioural Only" in names

    def test_tipping_point_none_when_empty(self):
        from src.feature_ablation import FeatureAblationEngine
        engine = FeatureAblationEngine()
        assert engine.get_tipping_point() is None


class TestPrecisionAtRecall:
    """Test the precision_at_recall utility."""

    def test_perfect_predictions(self):
        from src.feature_ablation import precision_at_recall
        y_true = np.array([0, 0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
        p = precision_at_recall(y_true, y_prob, target_recall=0.80)
        assert p > 0.5

    def test_random_predictions(self):
        from src.feature_ablation import precision_at_recall
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.1, size=1000)
        y_prob = np.random.rand(1000)
        p = precision_at_recall(y_true, y_prob, target_recall=0.80)
        assert 0.0 <= p <= 1.0


class TestDriftAnalysis:
    """Test the drift analysis module."""

    def test_import(self):
        from src.drift_analysis import BehaviouralDriftAnalyser
        assert BehaviouralDriftAnalyser is not None

    def test_temporal_drift(self):
        from src.drift_analysis import BehaviouralDriftAnalyser
        analyser = BehaviouralDriftAnalyser()
        df = pd.DataFrame({
            "TransactionDT": np.random.randint(0, 86400 * 30, size=100),
        })
        result = analyser.compute_temporal_drift(df)
        assert len(result) == 100
        assert result.name == "temporal_drift"

    def test_device_drift(self):
        from src.drift_analysis import BehaviouralDriftAnalyser
        analyser = BehaviouralDriftAnalyser()
        df = pd.DataFrame({
            "DeviceType": ["desktop", "mobile", None, "desktop"] * 25,
            "DeviceInfo": ["Windows", "iOS", None, "Android"] * 25,
        })
        result = analyser.compute_device_drift(df)
        assert len(result) == 100
        assert result.name == "device_drift"

    def test_velocity_drift(self):
        from src.drift_analysis import BehaviouralDriftAnalyser
        analyser = BehaviouralDriftAnalyser()
        df = pd.DataFrame({
            f"D{i}": np.random.exponential(50, size=100) for i in range(1, 6)
        })
        result = analyser.compute_velocity_drift(df)
        assert len(result) == 100
        assert result.name == "velocity_drift"

    def test_compute_all_drift_scores(self):
        from src.drift_analysis import BehaviouralDriftAnalyser
        analyser = BehaviouralDriftAnalyser()
        df = pd.DataFrame({
            "TransactionDT": np.random.randint(0, 86400 * 30, size=50),
            "DeviceType": np.random.choice(["desktop", "mobile"], size=50),
            **{f"C{i}": np.random.randint(0, 10, size=50) for i in range(1, 5)},
            **{f"D{i}": np.random.exponential(50, size=50) for i in range(1, 5)},
            "P_emaildomain": np.random.choice(["gmail.com", "yahoo.com"], size=50),
        })
        drift_df = analyser.compute_all_drift_scores(df)
        assert "temporal_drift" in drift_df.columns
        assert "composite_drift" in drift_df.columns
        assert drift_df.shape[0] == 50
