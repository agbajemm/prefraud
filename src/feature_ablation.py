"""
Feature Ablation Engine
Implements the supervisor's core methodology: progressive removal of direct
fraud indicators to identify the pre-fraud boundary.

Stages:
    0: Full model (baseline) — all features
    1: Remove TransactionAmt (strongest single predictor)
    2: Remove card identifiers (card1-card6)
    3: Remove address/distance features (addr1, addr2, dist1, dist2)
    4: Remove direct matching features (M1-M9)
    5: Remove counting features (C1-C14)
    6: Remove top 50% Vesta features by SHAP importance
    7: Retain ONLY behavioural/temporal/device features
"""

import logging
import json
import pickle
import numpy as np
import pandas as pd
import yaml
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    confusion_matrix,
)
from scipy import stats
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path=None):
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def precision_at_recall(y_true, y_prob, target_recall=0.80):
    """Calculate precision at a specified recall level."""
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    valid = recalls >= target_recall
    if valid.any():
        return precisions[valid].max()
    return 0.0


class FeatureAblationEngine:
    """
    Implements progressive removal of direct fraud indicators to determine
    the boundary between fraud detection and pre-fraud prediction.
    """

    # Ablation stages: each maps stage_name -> list of feature patterns to remove
    ABLATION_STAGES = [
        {
            "name": "Stage 0: Full Model",
            "remove": [],
            "description": "All features retained (baseline)",
        },
        {
            "name": "Stage 1: Remove TransactionAmt",
            "remove": ["TransactionAmt"],
            "description": "Strongest single fraud predictor removed",
        },
        {
            "name": "Stage 2: Remove Card Identifiers",
            "remove": ["card1", "card2", "card3", "card4", "card5", "card6"],
            "description": "Card number hashes and attributes removed",
        },
        {
            "name": "Stage 3: Remove Address/Distance",
            "remove": ["addr1", "addr2", "dist1", "dist2"],
            "description": "Geographic identifiers removed",
        },
        {
            "name": "Stage 4: Remove Match Features",
            "remove": [f"M{i}" for i in range(1, 10)],
            "description": "Boolean match features removed",
        },
        {
            "name": "Stage 5: Remove Counting Features",
            "remove": [f"C{i}" for i in range(1, 15)],
            "description": "Counting features removed",
        },
        {
            "name": "Stage 6: Remove Top Vesta Features",
            "remove": "vesta_top50",  # Resolved dynamically
            "description": "Top 50% Vesta features by SHAP removed",
        },
        {
            "name": "Stage 7: Behavioural Only",
            "remove": "all_non_behavioural",  # Resolved dynamically
            "description": "Only behavioural/temporal/device features retained",
        },
    ]

    # Behavioural features to always retain in Stage 7
    BEHAVIOURAL_FEATURES = (
        ["TransactionDT", "DeviceType", "DeviceInfo",
         "id_30", "id_31", "id_33",
         "P_emaildomain", "R_emaildomain"]
        + [f"D{i}" for i in range(1, 16)]
        + [f"id_{i}" for i in range(1, 39)]
    )

    def __init__(self, config=None):
        self.config = config or load_config()
        self.results = []
        self.vesta_importances = None

    def set_vesta_importances(self, importances):
        """
        Set Vesta feature importances (e.g. from SHAP) so Stage 6
        can resolve which Vesta features to remove.
        """
        v_cols = {k: v for k, v in importances.items() if k.startswith("V") and k[1:].isdigit()}
        self.vesta_importances = pd.Series(v_cols).sort_values(ascending=False)

    def _resolve_features_to_remove(self, stage, available_features):
        """Resolve dynamic removal lists for stages 6 and 7."""
        remove_spec = stage["remove"]

        if isinstance(remove_spec, list):
            return [f for f in remove_spec if f in available_features]

        if remove_spec == "vesta_top50":
            if self.vesta_importances is not None and len(self.vesta_importances) > 0:
                top_n = len(self.vesta_importances) // 2
                top_vesta = self.vesta_importances.index[:top_n].tolist()
                return [f for f in top_vesta if f in available_features]
            # Fallback: remove all V columns
            return [f for f in available_features if f.startswith("V") and f[1:].isdigit()]

        if remove_spec == "all_non_behavioural":
            behavioural_set = set(self.BEHAVIOURAL_FEATURES)
            return [f for f in available_features if f not in behavioural_set]

        return []

    def run_ablation(self, X_train, y_train, X_val, y_val, X_test, y_test, n_folds=5):
        """
        Execute the full progressive ablation study.

        Returns a list of stage result dicts with metrics and metadata.
        """
        self.results = []
        cumulative_removed = set()
        all_features = list(X_train.columns)

        for stage_idx, stage in enumerate(self.ABLATION_STAGES):
            logger.info("=" * 60)
            logger.info("ABLATION %s", stage["name"])
            logger.info("  %s", stage["description"])

            # Resolve which features to remove at this stage
            features_to_remove = self._resolve_features_to_remove(stage, all_features)
            cumulative_removed.update(features_to_remove)

            # Remaining features
            remaining = [f for f in all_features if f not in cumulative_removed]
            logger.info(
                "  Removing %d features | Remaining: %d",
                len(features_to_remove), len(remaining),
            )

            if len(remaining) == 0:
                logger.warning("  No features remaining — stopping ablation.")
                break

            # Subset data
            X_tr = X_train[remaining]
            X_v = X_val[remaining]
            X_te = X_test[remaining]

            # Train & evaluate with cross-validation
            stage_result = self._train_and_evaluate(
                X_tr, y_train, X_v, y_val, X_te, y_test,
                stage_name=stage["name"],
                n_folds=n_folds,
            )
            stage_result["stage_index"] = stage_idx
            stage_result["stage_name"] = stage["name"]
            stage_result["description"] = stage["description"]
            stage_result["features_removed_this_stage"] = features_to_remove
            stage_result["cumulative_removed_count"] = len(cumulative_removed)
            stage_result["remaining_features"] = remaining
            stage_result["remaining_count"] = len(remaining)

            self.results.append(stage_result)

            # Check tipping point
            if stage_result["test_auc_roc"] < 0.80 and stage_idx > 0:
                logger.info(
                    "  *** TIPPING POINT: AUC-ROC dropped below 0.80 at %s ***",
                    stage["name"],
                )

        self._compute_deltas()
        return self.results

    def _train_and_evaluate(self, X_train, y_train, X_val, y_val, X_test, y_test,
                            stage_name, n_folds=5):
        """Train XGBoost with CV on the given feature subset and evaluate."""
        # Calculate scale_pos_weight
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        spw = n_neg / max(n_pos, 1)

        params = self.config["model"]["baseline"]["params"].copy()
        params["scale_pos_weight"] = spw
        if params.get("eval_metric") == "auc":
            params["eval_metric"] = "auc"

        # Cross-validation
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_auc_roc = []
        cv_auc_pr = []

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train)):
            Xf_tr = X_train.iloc[tr_idx]
            yf_tr = y_train.iloc[tr_idx]
            Xf_va = X_train.iloc[va_idx]
            yf_va = y_train.iloc[va_idx]

            model = xgb.XGBClassifier(
                n_estimators=params.get("n_estimators", 500),
                max_depth=params.get("max_depth", 8),
                learning_rate=params.get("learning_rate", 0.05),
                subsample=params.get("subsample", 0.8),
                colsample_bytree=params.get("colsample_bytree", 0.8),
                scale_pos_weight=spw,
                eval_metric="auc",
                early_stopping_rounds=params.get("early_stopping_rounds", 50),
                random_state=42,
                n_jobs=-1,
                tree_method="hist",
                verbosity=0,
            )
            model.fit(
                Xf_tr, yf_tr,
                eval_set=[(Xf_va, yf_va)],
                verbose=False,
            )
            y_prob = model.predict_proba(Xf_va)[:, 1]
            cv_auc_roc.append(roc_auc_score(yf_va, y_prob))
            cv_auc_pr.append(average_precision_score(yf_va, y_prob))

        # Train final model on full training data with validation early-stopping
        final_model = xgb.XGBClassifier(
            n_estimators=params.get("n_estimators", 500),
            max_depth=params.get("max_depth", 8),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            scale_pos_weight=spw,
            eval_metric="auc",
            early_stopping_rounds=params.get("early_stopping_rounds", 50),
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
        )
        final_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Evaluate on held-out test set
        y_test_prob = final_model.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_prob >= 0.5).astype(int)

        test_auc_roc = roc_auc_score(y_test, y_test_prob)
        test_auc_pr = average_precision_score(y_test, y_test_prob)
        test_f1 = f1_score(y_test, y_test_pred)
        test_p_at_r80 = precision_at_recall(y_test, y_test_prob, 0.80)
        cm = confusion_matrix(y_test, y_test_pred)

        logger.info(
            "  CV AUC-ROC: %.4f ± %.4f | Test AUC-ROC: %.4f | AUC-PR: %.4f | F1: %.4f",
            np.mean(cv_auc_roc), np.std(cv_auc_roc),
            test_auc_roc, test_auc_pr, test_f1,
        )

        return {
            "cv_auc_roc": cv_auc_roc,
            "cv_auc_roc_mean": float(np.mean(cv_auc_roc)),
            "cv_auc_roc_std": float(np.std(cv_auc_roc)),
            "cv_auc_pr": cv_auc_pr,
            "cv_auc_pr_mean": float(np.mean(cv_auc_pr)),
            "test_auc_roc": float(test_auc_roc),
            "test_auc_pr": float(test_auc_pr),
            "test_f1": float(test_f1),
            "test_precision_at_recall_80": float(test_p_at_r80),
            "confusion_matrix": cm.tolist(),
            "y_test_true": y_test.values.tolist(),
            "y_test_prob": y_test_prob.tolist(),
            "model": final_model,
        }

    def _compute_deltas(self):
        """Compute performance delta from previous stage."""
        for i in range(1, len(self.results)):
            prev = self.results[i - 1]
            curr = self.results[i]
            curr["delta_auc_roc"] = curr["test_auc_roc"] - prev["test_auc_roc"]
            curr["delta_auc_pr"] = curr["test_auc_pr"] - prev["test_auc_pr"]
            curr["delta_f1"] = curr["test_f1"] - prev["test_f1"]

            # Paired t-test on CV folds
            if len(prev["cv_auc_roc"]) == len(curr["cv_auc_roc"]):
                t_stat, p_val = stats.ttest_rel(prev["cv_auc_roc"], curr["cv_auc_roc"])
                curr["ttest_statistic"] = float(t_stat)
                curr["ttest_pvalue"] = float(p_val)
                curr["significant_drop"] = p_val < 0.05
            else:
                curr["ttest_pvalue"] = None
                curr["significant_drop"] = None

        # First stage has no delta
        if self.results:
            self.results[0]["delta_auc_roc"] = 0.0
            self.results[0]["delta_auc_pr"] = 0.0
            self.results[0]["delta_f1"] = 0.0

    def get_tipping_point(self, threshold=0.80):
        """Find the stage where AUC-ROC drops below the threshold."""
        for r in self.results:
            if r["test_auc_roc"] < threshold:
                return r
        return None

    def get_pre_fraud_boundary_features(self):
        """
        Return the feature set at the pre-fraud boundary — the last stage
        where meaningful prediction is still possible but direct detection
        has become unreliable.
        """
        tipping = self.get_tipping_point(0.80)
        if tipping is not None:
            return tipping.get("remaining_features", [])
        # If no tipping point, return Stage 7 features
        if self.results:
            return self.results[-1].get("remaining_features", [])
        return []

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_results(self):
        """Save ablation results to disk."""
        out_dir = PROJECT_ROOT / "results" / "tables"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Summary table
        summary_rows = []
        for r in self.results:
            summary_rows.append({
                "stage": r["stage_name"],
                "description": r["description"],
                "features_remaining": r["remaining_count"],
                "test_auc_roc": r["test_auc_roc"],
                "test_auc_pr": r["test_auc_pr"],
                "test_f1": r["test_f1"],
                "test_p_at_r80": r["test_precision_at_recall_80"],
                "cv_auc_roc_mean": r["cv_auc_roc_mean"],
                "cv_auc_roc_std": r["cv_auc_roc_std"],
                "delta_auc_roc": r.get("delta_auc_roc", 0),
                "ttest_pvalue": r.get("ttest_pvalue", None),
                "significant": r.get("significant_drop", None),
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(out_dir / "ablation_summary.csv", index=False)
        logger.info("Saved ablation summary to %s", out_dir / "ablation_summary.csv")

        # Full results (excluding non-serialisable objects)
        serialisable = []
        for r in self.results:
            s = {k: v for k, v in r.items()
                 if k not in ("model", "y_test_true", "y_test_prob")}
            serialisable.append(s)
        with open(out_dir / "ablation_results.json", "w") as f:
            json.dump(serialisable, f, indent=2, default=str)

        # Save models
        model_dir = PROJECT_ROOT / "results" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        for r in self.results:
            stage_idx = r["stage_index"]
            model = r.get("model")
            if model is not None:
                with open(model_dir / f"ablation_stage_{stage_idx}.pkl", "wb") as f:
                    pickle.dump(model, f)

        logger.info("Saved ablation models and results.")

    def generate_plots(self):
        """Generate ablation visualisation plots."""
        from src.visualisation import Visualiser

        vis = Visualiser()
        fig_dir = PROJECT_ROOT / "results" / "figures" / "ablation"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # Ablation curve
        stages = [r["stage_name"] for r in self.results]
        metrics = []
        for r in self.results:
            m = r["cv_auc_roc_mean"]
            s = r["cv_auc_roc_std"]
            metrics.append({"mean": m, "lower": m - 1.96 * s, "upper": m + 1.96 * s})
        vis.plot_ablation_curve(stages, metrics, fig_dir / "ablation_curve.png")

        # Waterfall chart
        stage_names = [r["stage_name"] for r in self.results[1:]]
        deltas = [r.get("delta_auc_roc", 0) for r in self.results[1:]]
        vis.plot_ablation_waterfall(stage_names, deltas, fig_dir / "ablation_waterfall.png")

        # Overlaid ROC curves
        roc_data = {}
        for r in self.results:
            roc_data[r["stage_name"]] = (
                np.array(r["y_test_true"]),
                np.array(r["y_test_prob"]),
            )
        vis.plot_roc_curves(roc_data, fig_dir / "ablation_roc_overlay.png")
        vis.plot_pr_curves(roc_data, fig_dir / "ablation_pr_overlay.png")

        logger.info("Ablation plots saved to %s", fig_dir)


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    from src.data_loader import DataLoader

    loader = DataLoader()
    train_df, val_df, test_df = loader.load_processed()

    exclude = {"TransactionID", "isFraud"}
    feature_cols = [c for c in train_df.columns if c not in exclude]

    X_train = train_df[feature_cols]
    y_train = train_df["isFraud"]
    X_val = val_df[feature_cols]
    y_val = val_df["isFraud"]
    X_test = test_df[feature_cols]
    y_test = test_df["isFraud"]

    # Optionally load SHAP importances for Vesta resolution
    shap_path = PROJECT_ROOT / "results" / "tables" / "shap_importances.csv"
    engine = FeatureAblationEngine()
    if shap_path.exists():
        shap_df = pd.read_csv(shap_path)
        importances = dict(zip(shap_df["feature"], shap_df["importance"]))
        engine.set_vesta_importances(importances)

    results = engine.run_ablation(X_train, y_train, X_val, y_val, X_test, y_test)
    engine.save_results()
    engine.generate_plots()

    tipping = engine.get_tipping_point()
    if tipping:
        logger.info("Tipping point: %s (AUC-ROC=%.4f)", tipping["stage_name"], tipping["test_auc_roc"])

    boundary_features = engine.get_pre_fraud_boundary_features()
    logger.info("Pre-fraud boundary: %d features remaining", len(boundary_features))
