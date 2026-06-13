import json
import os

import pandas as pd
from loguru import logger

_ARTIFACT_DIR    = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "artifacts")
)
PREDICTIONS_PATH = os.path.join(_ARTIFACT_DIR, "predictions.pkl")
THRESHOLDS_PATH  = os.path.join(_ARTIFACT_DIR, "thresholds.json")


class ThresholdAnalyzer:
    """
    Computes 99th percentile thresholds for score_A (clustered IF)
    and score_B (global IF).

    Converts continuous anomaly scores to binary flags for tier assignment.
    99th percentile adapts automatically as daily data accumulates —
    always flags top 1% regardless of dataset size.
    """

    def __init__(self, percentile: int = 99):
        self.percentile = percentile

    def run(self) -> dict:
        df = pd.read_pickle(PREDICTIONS_PATH)
        logger.info("ThresholdAnalyzer.run | rows={} | percentile={}",
                    len(df), self.percentile)

        threshold_A = float(df["score_A"].quantile(self.percentile / 100))
        threshold_B = float(df["score_B"].quantile(self.percentile / 100))

        logger.info(
            "score_A | min={:.4f}  mean={:.4f}  max={:.4f}  p{}={:.4f}",
            df["score_A"].min(), df["score_A"].mean(),
            df["score_A"].max(), self.percentile, threshold_A,
        )
        logger.info(
            "score_B | min={:.4f}  mean={:.4f}  max={:.4f}  p{}={:.4f}",
            df["score_B"].min(), df["score_B"].mean(),
            df["score_B"].max(), self.percentile, threshold_B,
        )

        n_A = int((df["score_A"] >= threshold_A).sum())
        n_B = int((df["score_B"] >= threshold_B).sum())
        logger.info(
            "Records above threshold | Track A={} ({:.2f}%)  Track B={} ({:.2f}%)",
            n_A, n_A / len(df) * 100,
            n_B, n_B / len(df) * 100,
        )

        thresholds = {
            "threshold_A": threshold_A,
            "threshold_B": threshold_B,
            "percentile":  self.percentile,
        }

        os.makedirs(os.path.dirname(THRESHOLDS_PATH), exist_ok=True)
        with open(THRESHOLDS_PATH, "w") as f:
            json.dump(thresholds, f, indent=2)

        logger.success("Thresholds saved | path={}", THRESHOLDS_PATH)
        return thresholds
