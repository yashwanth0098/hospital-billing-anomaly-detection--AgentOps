import json
import os

import joblib
import pandas as pd
from loguru import logger

from ..data_preprocessing.preprocessor import DataPreprocessingPipeline

_ARTIFACT_DIR       = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "artifacts")
)
KMEANS_PATH         = os.path.join(_ARTIFACT_DIR, "kmeans.pkl")
CLUSTER_MODELS_PATH = os.path.join(_ARTIFACT_DIR, "cluster_models.pkl")
GLOBAL_IF_PATH      = os.path.join(_ARTIFACT_DIR, "global_isolation_forest.pkl")
CLUSTER_META_PATH   = os.path.join(_ARTIFACT_DIR, "cluster_meta.json")
THRESHOLDS_PATH     = os.path.join(_ARTIFACT_DIR, "thresholds.json")


class InferencePredictor:
    """
    Scores a new batch of raw patient records using the saved 3-track pipeline.

    Loads all trained artifacts at construction time:
      kmeans.pkl               — cluster assignment
      cluster_models.pkl       — per-cluster IsolationForest models
      global_isolation_forest  — global IsolationForest (Track B + fallback)
      cluster_meta.json        — optimal k, fallback cluster list
      thresholds.json          — 99th-percentile thresholds for A and B

    predict() encodes the raw DataFrame with the saved transformer state,
    runs all three tracks, and returns one row per patient with scores,
    business-rule flags, and a tier label (0–3).
    """

    def __init__(self, include_optional: bool = True):
        self.include_optional    = include_optional
        self._kmeans             = joblib.load(KMEANS_PATH)
        self._cluster_models     = joblib.load(CLUSTER_MODELS_PATH)
        self._global_model       = joblib.load(GLOBAL_IF_PATH)

        with open(CLUSTER_META_PATH, encoding="utf-8") as f:
            self._cluster_meta   = json.load(f)
        with open(THRESHOLDS_PATH, encoding="utf-8") as f:
            self._thresholds     = json.load(f)

        self._fallback_clusters  = set(self._cluster_meta.get("fallback_clusters", []))

        logger.info(
            "InferencePredictor loaded | k={} | fallback_clusters={} | "
            "threshold_A={:.4f} | threshold_B={:.4f}",
            self._cluster_meta["optimal_k"],
            sorted(self._fallback_clusters),
            self._thresholds["threshold_A"],
            self._thresholds["threshold_B"],
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score a raw DataFrame of patient records.

        Args:
            df: Raw billing records — same schema as the training dataset.

        Returns:
            DataFrame with encoded features + score_A + score_B + flag_C +
            per-rule columns + tier (0-3) per patient.
        """
        pipeline     = DataPreprocessingPipeline(include_optional=self.include_optional)
        X            = pipeline.run_inference(df)
        feature_cols = list(X.columns)
        logger.info(
            "InferencePredictor.predict | rows={} features={}",
            len(X), len(feature_cols),
        )

        # ── Track B: global score (available for every row) ──────────────────
        X["score_B"] = -self._global_model.decision_function(X[feature_cols])

        # ── Track A: assign cluster then score ───────────────────────────────
        X["cluster"] = self._kmeans.predict(X[feature_cols])
        X["score_A"] = X["score_B"].copy()   # default — overridden below

        for cluster_id in X["cluster"].unique():
            if cluster_id in self._fallback_clusters:
                # small cluster at training time — global score already set
                logger.debug(
                    "Cluster {} → keeping score_B (was a fallback cluster at training)",
                    cluster_id,
                )
                continue

            if cluster_id not in self._cluster_models:
                # cluster not seen at training — safe default is global score
                logger.warning(
                    "Cluster {} not found in trained models → using score_B",
                    cluster_id,
                )
                continue

            mask      = X["cluster"] == cluster_id
            X_cluster = X.loc[mask, feature_cols]
            X.loc[mask, "score_A"] = (
                -self._cluster_models[cluster_id].decision_function(X_cluster)
            )

        # ── Track C: business rules ───────────────────────────────────────────
        X = self._apply_business_rules(X)

        # ── Threshold flags and tier assignment ───────────────────────────────
        threshold_A  = self._thresholds["threshold_A"]
        threshold_B  = self._thresholds["threshold_B"]
        X["flag_A"]  = (X["score_A"] >= threshold_A).astype(int)
        X["flag_B"]  = (X["score_B"] >= threshold_B).astype(int)
        X["n_signals"] = X["flag_A"] + X["flag_B"] + X["flag_C"]
        X["tier"]    = X["n_signals"].clip(upper=3)

        logger.info(
            "Prediction complete | Tier3={} Tier2={} Tier1={} Tier0={}",
            int((X["tier"] == 3).sum()),
            int((X["tier"] == 2).sum()),
            int((X["tier"] == 1).sum()),
            int((X["tier"] == 0).sum()),
        )
        return X

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _apply_business_rules(self, X: pd.DataFrame) -> pd.DataFrame:
        X["charge_cluster_mean"] = (
            X.groupby("cluster")["charge_amount_USD"].transform("mean")
        )
        X["charge_cluster_std"] = (
            X.groupby("cluster")["charge_amount_USD"].transform("std")
        )
        X["charge_z_score"] = (
            (X["charge_amount_USD"] - X["charge_cluster_mean"])
            / X["charge_cluster_std"].replace(0, 1)
        )

        rule_1        = X["charge_z_score"] > 3
        rule_2        = (X["claim_status_paid"] == 1) & (X["payment_amount_USD"] == 0)
        payment_ratio = X["payment_amount_USD"] / (X["charge_amount_USD"] + 1e-6)
        rule_3        = (X["claim_status_paid"] == 1) & (payment_ratio < 0.5)
        rule_4        = (X["claim_status_denied"] == 1) & (X["payment_amount_USD"] > 0)

        X["rule_1"] = rule_1.astype(int)
        X["rule_2"] = rule_2.astype(int)
        X["rule_3"] = rule_3.astype(int)
        X["rule_4"] = rule_4.astype(int)
        X["flag_C"] = (rule_1 | rule_2 | rule_3 | rule_4).astype(int)

        X = X.drop(columns=["charge_cluster_mean", "charge_cluster_std"])

        logger.info(
            "Business rules | R1={} R2={} R3={} R4={} total={}",
            int(rule_1.sum()), int(rule_2.sum()),
            int(rule_3.sum()), int(rule_4.sum()),
            int(X["flag_C"].sum()),
        )
        return X
