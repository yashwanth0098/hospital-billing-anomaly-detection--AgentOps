import os

import pandas as pd
from loguru import logger

from ..data_drift import DataDriftDetector
from .predictor import InferencePredictor

_ARTIFACT_DIR         = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "artifacts")
)
REFERENCE_DIST_PATH   = os.path.join(_ARTIFACT_DIR, "reference_distribution.json")
INFERENCE_OUTPUT_PATH = os.path.join(_ARTIFACT_DIR, "inference_results.pkl")
DRIFT_REPORT_PATH     = os.path.join(_ARTIFACT_DIR, "drift_report.csv")


class AnomalyScorer:
    """
    End-to-end inference orchestrator for daily batch scoring.

    Workflow per batch:
      1. Drift check  — compare incoming data against training reference (JSD + PSI).
                        Logs a warning if drift is detected but does NOT block scoring.
                        Human reviewer decides whether a retrain is needed.
      2. Score        — runs the 3-track pipeline (Track A + B + C) via InferencePredictor.
      3. Save         — persists results to artifacts/inference_results.pkl and
                        appends the drift run to artifacts/drift_report.csv.

    Returns a DataFrame with tier label (0–3) and all intermediate scores per patient.
    """

    def __init__(self, include_optional: bool = True):
        self._drift_detector = DataDriftDetector(reference_path=REFERENCE_DIST_PATH)
        self._predictor      = InferencePredictor(include_optional=include_optional)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame, save_results: bool = True) -> pd.DataFrame:
        """
        Score a raw batch of patient records end-to-end.

        Args:
            df:           Raw billing records — same schema as the training dataset.
            save_results: Write scored output to artifacts/inference_results.pkl.

        Returns:
            DataFrame with tier label and all intermediate scores per patient.
        """
        logger.info("AnomalyScorer.run | rows={}", len(df))

        # ── Step 1: drift check ───────────────────────────────────────────────
        drift_report = self._drift_detector.detect(df)
        drift_report.save_csv(DRIFT_REPORT_PATH)

        if drift_report.overall_drift:
            drifted_count = len(drift_report.drifted_columns)
            logger.warning(
                "Data drift detected | {} column(s) drifted: {} — "
                "scores are still produced but consider retraining.",
                drifted_count,
                drift_report.drifted_columns,
            )
        else:
            logger.info(
                "No drift detected — batch is consistent with training distribution."
            )

        # ── Step 2: 3-track scoring ───────────────────────────────────────────
        result = self._predictor.predict(df)

        # ── Step 3: persist results ───────────────────────────────────────────
        if save_results:
            os.makedirs(_ARTIFACT_DIR, exist_ok=True)
            result.to_pickle(INFERENCE_OUTPUT_PATH)
            logger.info(
                "Inference results saved | shape={} | path={}",
                result.shape, INFERENCE_OUTPUT_PATH,
            )

        logger.success(
            "AnomalyScorer complete | Tier3={} Tier2={} Tier1={} Tier0={}",
            int((result["tier"] == 3).sum()),
            int((result["tier"] == 2).sum()),
            int((result["tier"] == 1).sum()),
            int((result["tier"] == 0).sum()),
        )
        return result
