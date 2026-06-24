"""
Joins the raw billing batch with InferencePredictor scored output,
then splits into tier-specific DataFrames for daily reporting.

Why join on index:
    InferencePredictor.predict() encodes the raw DataFrame (drops text columns
    like department, diagnosis, patient_id) but preserves the pandas index.
    Re-joining on index restores identity fields needed for actionable reports.
"""
import pandas as pd
from loguru import logger

# Raw columns to pull back for report readability
_RAW_IDENTITY_COLS = [
    "patient_id", "appointment_id", "department", "diagnosis",
    "visit_type", "claim_status", "payer_type", "is_emergency",
    "visit_date", "payer_name", "physician_id", "visit_reason",
]

# Scored columns produced by InferencePredictor.predict()
_SCORE_COLS = [
    "score_A", "score_B", "cluster", "charge_z_score",
    "rule_1", "rule_2", "rule_3", "rule_4", "flag_C",
    "flag_A", "flag_B", "n_signals", "tier",
    "age", "charge_amount_USD", "payment_amount_USD",
]


class FlaggedExtractor:
    """
    Merges raw identity fields with scored fields, then splits by tier.

    Returns:
        {
          'all':    full enriched DataFrame,
          'tier_3': Investigate Immediately,
          'tier_2': Scheduled Review,
          'tier_1': Monitor,
          'tier_0': Normal,
        }
    """

    def extract(
        self,
        raw_df: pd.DataFrame,
        scored_df: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        id_cols    = [c for c in _RAW_IDENTITY_COLS if c in raw_df.columns]
        score_cols = [c for c in _SCORE_COLS if c in scored_df.columns]

        enriched = scored_df[score_cols].join(raw_df[id_cols], how="left")
        enriched.index.name = "record_index"

        logger.info(
            "FlaggedExtractor.extract | rows={} | identity_cols={} | score_cols={}",
            len(enriched), id_cols, score_cols,
        )

        tiers = {
            "all":    enriched,
            "tier_3": enriched[enriched["tier"] == 3].copy(),
            "tier_2": enriched[enriched["tier"] == 2].copy(),
            "tier_1": enriched[enriched["tier"] == 1].copy(),
            "tier_0": enriched[enriched["tier"] == 0].copy(),
        }

        logger.info(
            "Tier split | T3={} T2={} T1={} T0={}",
            len(tiers["tier_3"]), len(tiers["tier_2"]),
            len(tiers["tier_1"]), len(tiers["tier_0"]),
        )
        return tiers
