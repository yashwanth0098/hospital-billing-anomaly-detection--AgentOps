import json
import os

import pandas as pd
from loguru import logger

_ARTIFACT_DIR    = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "artifacts")
)
TIER_LABELS_PATH = os.path.join(_ARTIFACT_DIR, "tier_labels.pkl")
THRESHOLDS_PATH  = os.path.join(_ARTIFACT_DIR, "thresholds.json")


class ScoreDistribution:
    """
    Logs anomaly score statistics broken down by tier for Track A and Track B.

    Validates that scores meaningfully separate anomalies from normal records:
      large gap between Tier 3 mean and Tier 0 mean  →  model discriminates well
      small gap                                       →  model cannot distinguish
    """

    def run(self) -> None:
        df = pd.read_pickle(TIER_LABELS_PATH)
        with open(THRESHOLDS_PATH) as f:
            thresholds = json.load(f)
        logger.info("ScoreDistribution.run | rows={}", len(df))

        for track_label, score_col, threshold_key in [
            ("A — Clustered IF", "score_A", "threshold_A"),
            ("B — Global IF",    "score_B", "threshold_B"),
        ]:
            threshold = thresholds[threshold_key]
            logger.info("─" * 60)
            logger.info("Track {} | threshold (p{})={:.4f}",
                        track_label, thresholds["percentile"], threshold)

            for tier in [3, 2, 1, 0]:
                subset = df[df["tier"] == tier][score_col]
                if len(subset) == 0:
                    continue
                above = int((subset >= threshold).sum())
                logger.info(
                    "  Tier {} | n={:>5} | min={:.4f}  mean={:.4f}  max={:.4f}"
                    "  | above_threshold={}",
                    tier, len(subset),
                    subset.min(), subset.mean(), subset.max(),
                    above,
                )

            # Separation gap: Tier 3 mean vs Tier 0 mean
            t3_mean = df[df["tier"] == 3][score_col].mean()
            t0_mean = df[df["tier"] == 0][score_col].mean()
            gap     = t3_mean - t0_mean
            logger.info(
                "  Score gap (Tier3 mean − Tier0 mean) = {:.4f}  "
                "→  {}",
                gap,
                "GOOD separation" if gap > 0.05 else "LOW separation — review model",
            )

        logger.info("─" * 60)
