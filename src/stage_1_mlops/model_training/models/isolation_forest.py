import os

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from loguru import logger


class IsolationForestModel:
    """
    Thin wrapper around sklearn IsolationForest.

    predict()           → -1 (anomaly) / 1 (normal)
    anomaly_scores()    → continuous float; lower = more anomalous
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.01,
        random_state: int = 42,
    ):
        self.n_estimators  = n_estimators
        self.contamination = contamination
        self.random_state  = random_state
        self._model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )

    # ── Core ──────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame) -> "IsolationForestModel":
        logger.info(
            "IsolationForestModel.fit | rows={} features={} contamination={} n_estimators={}",
            len(X), X.shape[1], self.contamination, self.n_estimators,
        )
        self._model.fit(X)
        logger.success("IsolationForest fitted successfully")
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Returns -1 for anomaly, 1 for normal."""
        preds = self._model.predict(X)
        return pd.Series(preds, index=X.index, name="anomaly")

    def anomaly_scores(self, X: pd.DataFrame) -> pd.Series:
        """
        Raw decision function scores. Lower (more negative) = more anomalous.
        Negated so higher = more anomalous, consistent with typical score convention.
        """
        scores = -self._model.decision_function(X)
        return pd.Series(scores, index=X.index, name="anomaly_score")

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self._model, path)
        logger.info("Model saved | path={}", path)

    @classmethod
    def load(cls, path: str) -> "IsolationForestModel":
        obj = cls()
        obj._model = joblib.load(path)
        logger.info("Model loaded | path={}", path)
        return obj
