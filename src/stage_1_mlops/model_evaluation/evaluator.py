import json
import os

import pandas as pd
from loguru import logger

_ARTIFACT_DIR          = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "artifacts")
)
PREDICTIONS_PATH       = os.path.join(_ARTIFACT_DIR, "predictions.pkl")
THRESHOLDS_PATH        = os.path.join(_ARTIFACT_DIR, "thresholds.json")
TIER_LABELS_PATH       = os.path.join(_ARTIFACT_DIR, "tier_labels.pkl")
EVALUATION_REPORT_PATH = os.path.join(_ARTIFACT_DIR, "evaluation_report.json")


class Evaluator:
    """
    Combines outputs of all three tracks and assigns a tier per patient.

    Tier 3  —  all 3 tracks fire   →  investigate immediately
    Tier 2  —  any 2 tracks fire   →  scheduled review
    Tier 1  —  only 1 track fires  →  monitor
    Tier 0  —  none fire           →  normal
    """

    def run(self) -> pd.DataFrame:
        df         = pd.read_pickle(PREDICTIONS_PATH)
        thresholds = self._load_thresholds()
        logger.info("Evaluator.run | rows={}", len(df))

        # ── Step 1: Convert scores to binary flags ────────────────────────────
        df["track_A_flag"] = (df["score_A"] >= thresholds["threshold_A"]).astype(int)
        df["track_B_flag"] = (df["score_B"] >= thresholds["threshold_B"]).astype(int)
        df["track_C_flag"] = df["flag_C"]

        logger.info(
            "Binary flags | Track A={} Track B={} Track C={}",
            int(df["track_A_flag"].sum()),
            int(df["track_B_flag"].sum()),
            int(df["track_C_flag"].sum()),
        )

        # ── Step 2: Assign tiers ──────────────────────────────────────────────
        df["n_signals"] = (
            df["track_A_flag"] + df["track_B_flag"] + df["track_C_flag"]
        )
        df["tier"] = df["n_signals"].clip(upper=3)

        # ── Step 3: Build evaluation report ──────────────────────────────────
        report = self._build_report(df, thresholds)

        # ── Step 4: Save artifacts ────────────────────────────────────────────
        os.makedirs(_ARTIFACT_DIR, exist_ok=True)
        df.to_pickle(TIER_LABELS_PATH)
        with open(EVALUATION_REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)

        logger.success(
            "Evaluation complete | tier_labels → {} | report → {}",
            TIER_LABELS_PATH, EVALUATION_REPORT_PATH,
        )
        self._print_summary(report)
        return df

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_thresholds(self) -> dict:
        if not os.path.exists(THRESHOLDS_PATH):
            raise FileNotFoundError(
                f"Thresholds not found at '{THRESHOLDS_PATH}'. "
                "Run ThresholdAnalyzer first."
            )
        with open(THRESHOLDS_PATH) as f:
            return json.load(f)

    def _build_report(self, df: pd.DataFrame, thresholds: dict) -> dict:
        total = len(df)

        # Tier distribution
        tier_counts = df["tier"].value_counts().sort_index()

        # Track agreement rates
        a_b = int((df["track_A_flag"] == df["track_B_flag"]).sum())
        a_c = int((df["track_A_flag"] == df["track_C_flag"]).sum())
        b_c = int((df["track_B_flag"] == df["track_C_flag"]).sum())

        # Business rule contribution within Tier 3
        tier3 = df[df["tier"] == 3]
        rule_contribution = {}
        if len(tier3) > 0:
            rule_contribution = {
                "rule_1_charge_zscore" : int(tier3["rule_1"].sum()),
                "rule_2_paid_zero"     : int(tier3["rule_2"].sum()),
                "rule_3_paid_low"      : int(tier3["rule_3"].sum()),
                "rule_4_denied_paid"   : int(tier3["rule_4"].sum()),
            }

        # Anomaly profile: avg feature values for Tier 3 vs Tier 0
        profile_cols = [
            "charge_amount_USD", "payment_amount_USD",
            "age", "is_emergency", "score_A", "score_B",
        ]
        tier3_profile = (
            tier3[profile_cols].mean().round(4).to_dict()
            if len(tier3) > 0 else {}
        )
        tier0_profile = df[df["tier"] == 0][profile_cols].mean().round(4).to_dict()

        # Per-cluster breakdown
        cluster_breakdown = {}
        for cluster_id, grp in df.groupby("cluster"):
            cluster_breakdown[str(int(cluster_id))] = {
                "total"        : len(grp),
                "tier_3"       : int((grp["tier"] == 3).sum()),
                "tier_2"       : int((grp["tier"] == 2).sum()),
                "tier_1"       : int((grp["tier"] == 1).sum()),
                "tier_0"       : int((grp["tier"] == 0).sum()),
                "anomaly_rate_pct": round(
                    (grp["tier"] >= 2).sum() / len(grp) * 100, 2
                ),
            }

        def _tier_entry(tier_val: int) -> dict:
            count = int(tier_counts.get(tier_val, 0))
            return {"count": count, "pct": round(count / total * 100, 2)}

        return {
            "total_records"    : total,
            "thresholds"       : thresholds,
            "tier_distribution": {
                "tier_3_investigate": _tier_entry(3),
                "tier_2_review"     : _tier_entry(2),
                "tier_1_monitor"    : _tier_entry(1),
                "tier_0_normal"     : _tier_entry(0),
            },
            "track_agreement"  : {
                "A_vs_B_pct": round(a_b / total * 100, 2),
                "A_vs_C_pct": round(a_c / total * 100, 2),
                "B_vs_C_pct": round(b_c / total * 100, 2),
            },
            "rule_contribution_in_tier3": rule_contribution,
            "anomaly_profile"  : {
                "tier_3_avg": tier3_profile,
                "tier_0_avg": tier0_profile,
            },
            "cluster_breakdown": cluster_breakdown,
        }

    def _print_summary(self, report: dict) -> None:
        td    = report["tier_distribution"]
        ta    = report["track_agreement"]
        total = report["total_records"]
        logger.info("─" * 60)
        logger.info("EVALUATION SUMMARY | total_records={}", total)
        logger.info("Tier 3 — investigate  : {:>5}  ({:.2f}%)",
                    td["tier_3_investigate"]["count"], td["tier_3_investigate"]["pct"])
        logger.info("Tier 2 — review       : {:>5}  ({:.2f}%)",
                    td["tier_2_review"]["count"], td["tier_2_review"]["pct"])
        logger.info("Tier 1 — monitor      : {:>5}  ({:.2f}%)",
                    td["tier_1_monitor"]["count"], td["tier_1_monitor"]["pct"])
        logger.info("Tier 0 — normal       : {:>5}  ({:.2f}%)",
                    td["tier_0_normal"]["count"], td["tier_0_normal"]["pct"])
        logger.info("Track agreement A↔B   : {:.2f}%", ta["A_vs_B_pct"])
        logger.info("Track agreement A↔C   : {:.2f}%", ta["A_vs_C_pct"])
        logger.info("Track agreement B↔C   : {:.2f}%", ta["B_vs_C_pct"])
        logger.info("─" * 60)
