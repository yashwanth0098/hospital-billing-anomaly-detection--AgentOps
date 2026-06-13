import json
import os

import joblib
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score

from ..data_preprocessing.preprocessor import DataPreprocessingPipeline
_ARTIFACT_DIR       = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "artifacts")
)
KMEANS_PATH         = os.path.join(_ARTIFACT_DIR, "kmeans.pkl")
CLUSTER_MODELS_PATH = os.path.join(_ARTIFACT_DIR, "cluster_models.pkl")
GLOBAL_IF_PATH      = os.path.join(_ARTIFACT_DIR, "global_isolation_forest.pkl")
PREDICTIONS_PATH    = os.path.join(_ARTIFACT_DIR, "predictions.pkl")
CLUSTER_META_PATH   = os.path.join(_ARTIFACT_DIR, "cluster_meta.json")


MIN_CLUSTER_SIZE = 100   # clusters below this use global model score instead


class ModelTrainer:
    """
    3-track anomaly detection trainer.

    Track B  —  Single global IsolationForest (fitted first)          →  score_B
    Track A  —  K-Means clustering + IsolationForest per cluster      →  score_A
                clusters with fewer than MIN_CLUSTER_SIZE records
                are statistically unreliable — they fall back to
                score_B so small-cluster patients are still scored
                correctly rather than by a fragile local model.
    Track C  —  4 billing business rules                              →  flag_C

    All three are saved in predictions.pkl for the Evaluator to combine into tiers.
    """

    def __init__(
        self,
        contamination: float    = 0.01,
        n_estimators: int       = 100,
        random_state: int       = 42,
        min_cluster_size: int   = MIN_CLUSTER_SIZE,
    ):
        self.contamination     = contamination
        self.n_estimators      = n_estimators
        self.random_state      = random_state
        self.min_cluster_size  = min_cluster_size
        self._kmeans           = None
        self._cluster_models   = {}
        self._global_model     = None
        self._optimal_k        = None
        self._fallback_clusters: list[int] = []   # clusters that used global fallback

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        X = DataPreprocessingPipeline.load_transformed_data()
        feature_cols = list(X.columns)
        logger.info("ModelTrainer.run | rows={} features={}", len(X), len(feature_cols))

        # ── Track B: global IsolationForest (must run FIRST) ─────────────────
        # Fitted before Track A so small clusters can fall back to this score.
        logger.info("Track B — Global IsolationForest")
        self._global_model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self._global_model.fit(X[feature_cols])
        X["score_B"] = -self._global_model.decision_function(X[feature_cols])
        logger.info("Global IF fitted | rows={}", len(X))

        # ── Track A: clustered IsolationForest ───────────────────────────────
        logger.info("Track A — K-Means + Clustered IsolationForest")
        self._optimal_k = self._find_optimal_k(X, feature_cols)
        self._kmeans    = KMeans(
            n_clusters=self._optimal_k, random_state=self.random_state, n_init=10
        )
        X["cluster"] = self._kmeans.fit_predict(X[feature_cols])
        logger.info("K-Means fitted | k={}", self._optimal_k)

        # Initialise score_A from score_B — small clusters will keep this value
        X["score_A"] = X["score_B"].copy()

        for cluster_id in sorted(X["cluster"].unique()):
            mask       = X["cluster"] == cluster_id
            n_in_cluster = int(mask.sum())

            if n_in_cluster < self.min_cluster_size:
                # Too few records to fit a reliable IsolationForest.
                # score_A already holds score_B for these patients — no model fitted.
                self._fallback_clusters.append(int(cluster_id))
                logger.warning(
                    "Cluster {} | rows={} < min_cluster_size={} "
                    "→ falling back to global model score (score_B)",
                    cluster_id, n_in_cluster, self.min_cluster_size,
                )
                continue

            X_cluster = X.loc[mask, feature_cols]
            model     = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=self.random_state,
            )
            model.fit(X_cluster)
            X.loc[mask, "score_A"]           = -model.decision_function(X_cluster)
            self._cluster_models[cluster_id] = model
            logger.info(
                "Clustered IF | cluster={} rows={} flagged={}",
                cluster_id, n_in_cluster,
                int((model.predict(X_cluster) == -1).sum()),
            )

        if self._fallback_clusters:
            logger.info(
                "Fallback clusters (used global score) : {}",
                self._fallback_clusters,
            )

        # ── Track C: business rules ───────────────────────────────────────────
        logger.info("Track C — Business rules")
        X = self._apply_business_rules(X)

        # ── Save all artifacts ────────────────────────────────────────────────
        save_cols = (
            feature_cols
            + ["cluster", "score_A", "score_B", "flag_C",
               "charge_z_score", "rule_1", "rule_2", "rule_3", "rule_4"]
        )
        result = X[save_cols].copy()
        self._save_artifacts(result)

        logger.success("ModelTrainer complete | predictions_shape={}", result.shape)
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _find_optimal_k(self, X: pd.DataFrame, feature_cols: list) -> int:
        k_range    = range(2, 11)
        silhouettes = []
        for k in k_range:
            km     = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(X[feature_cols])
            sil    = silhouette_score(X[feature_cols], labels)
            silhouettes.append(sil)
            logger.debug("k={} silhouette={:.4f}", k, sil)
        optimal_k = list(k_range)[silhouettes.index(max(silhouettes))]
        logger.info(
            "Optimal k={} selected | silhouette={:.4f}",
            optimal_k, max(silhouettes),
        )
        return optimal_k

    def _apply_business_rules(self, X: pd.DataFrame) -> pd.DataFrame:
        # Rule 1: charge z-score > 3 within cluster
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
        rule_1 = X["charge_z_score"] > 3

        # Rule 2: Paid claim with zero payment collected
        rule_2 = (X["claim_status_paid"] == 1) & (X["payment_amount_USD"] == 0)

        # Rule 3: Paid claim but collected < 50% of charge
        payment_ratio = X["payment_amount_USD"] / (X["charge_amount_USD"] + 1e-6)
        rule_3 = (X["claim_status_paid"] == 1) & (payment_ratio < 0.5)

        # Rule 4: Denied claim but payment > 0 (workflow violation)
        rule_4 = (X["claim_status_denied"] == 1) & (X["payment_amount_USD"] > 0)

        X["rule_1"] = rule_1.astype(int)
        X["rule_2"] = rule_2.astype(int)
        X["rule_3"] = rule_3.astype(int)
        X["rule_4"] = rule_4.astype(int)
        X["flag_C"] = (rule_1 | rule_2 | rule_3 | rule_4).astype(int)

        logger.info(
            "Business rules | R1={} R2={} R3={} R4={} total={}",
            int(rule_1.sum()), int(rule_2.sum()),
            int(rule_3.sum()), int(rule_4.sum()),
            int(X["flag_C"].sum()),
        )
        # Drop intermediate columns
        X = X.drop(columns=["charge_cluster_mean", "charge_cluster_std"])
        return X

    def _save_artifacts(self, result: pd.DataFrame) -> None:
        os.makedirs(_ARTIFACT_DIR, exist_ok=True)

        joblib.dump(self._kmeans, KMEANS_PATH)
        logger.info("K-Means saved | path={}", KMEANS_PATH)

        joblib.dump(self._cluster_models, CLUSTER_MODELS_PATH)
        logger.info("Cluster models saved | n_models={} | path={}", len(self._cluster_models), CLUSTER_MODELS_PATH)

        joblib.dump(self._global_model, GLOBAL_IF_PATH)
        logger.info("Global IF saved | path={}", GLOBAL_IF_PATH)

        cluster_meta = {
            "optimal_k"        : self._optimal_k,
            "min_cluster_size" : self.min_cluster_size,
            "cluster_sizes"    : {
                str(k): int(v)
                for k, v in result["cluster"].value_counts().sort_index().items()
            },
            "fallback_clusters": self._fallback_clusters,
        }
        with open(CLUSTER_META_PATH, "w") as f:
            json.dump(cluster_meta, f, indent=2)
        logger.info("Cluster meta saved | path={}", CLUSTER_META_PATH)

        result.to_pickle(PREDICTIONS_PATH)
        logger.info("Predictions saved | shape={} | path={}", result.shape, PREDICTIONS_PATH)
