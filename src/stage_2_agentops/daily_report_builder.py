"""
Builds the structured, LLM-ready daily report dict from tier-split DataFrames.

Output shape:
    {
      run_id, run_date, batch_info,
      anomaly_summary,       <- tier counts + anomaly rates
      track_breakdown,       <- per-track fire counts
      rule_breakdown,        <- per-business-rule hit counts
      tier_3_cases,          <- one dict per Tier 3 record (full detail)
      tier_2_summary,        <- aggregate stats for Tier 2
      tier_1_summary,        <- aggregate stats for Tier 1
      llm_context,           <- plain-text paragraph ready for LLM consumption
    }
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from loguru import logger

_RULE_LABELS: dict[str, str] = {
    "rule_1": "charge_zscore_gt3",
    "rule_2": "paid_claim_zero_payment",
    "rule_3": "paid_claim_low_payment_ratio",
    "rule_4": "denied_claim_with_payment",
}

_TIER3_IDENTITY = [
    "patient_id", "appointment_id", "department", "diagnosis",
    "visit_type", "visit_reason", "claim_status", "payer_type",
    "payer_name", "is_emergency", "visit_date", "physician_id",
]


class DailyReportBuilder:
    """Assembles the full structured report dict for one daily batch."""

    def build(
        self,
        tiers: dict[str, pd.DataFrame],
        run_id: str,
        source_name: str,
        run_date: str | None = None,
    ) -> dict[str, Any]:
        run_date = run_date or datetime.now().strftime("%Y-%m-%d")
        all_df   = tiers["all"]
        total    = len(all_df)

        report: dict[str, Any] = {
            "run_id":          run_id,
            "run_date":        run_date,
            "batch_info":      {"total_records": total, "source": source_name},
            "anomaly_summary": self._anomaly_summary(tiers, total),
            "track_breakdown": self._track_breakdown(all_df),
            "rule_breakdown":  self._rule_breakdown(all_df),
            "tier_3_cases":    self._tier3_cases(tiers["tier_3"]),
            "tier_2_summary":  self._tier2_summary(tiers["tier_2"]),
            "tier_1_summary":  self._tier1_summary(tiers["tier_1"]),
            "llm_context":     self._llm_context(tiers, run_id, run_date, source_name),
        }

        logger.info(
            "DailyReportBuilder.build | run_id={} | T3={} T2={} T1={} total={}",
            run_id,
            len(tiers["tier_3"]), len(tiers["tier_2"]),
            len(tiers["tier_1"]), total,
        )
        return report

    # ── Sections ─────────────────────────────────────────────────────────────

    def _anomaly_summary(self, tiers: dict, total: int) -> dict:
        t3 = len(tiers["tier_3"])
        t2 = len(tiers["tier_2"])
        t1 = len(tiers["tier_1"])
        t0 = len(tiers["tier_0"])
        safe = max(total, 1)
        return {
            "tier_3_investigate_immediately": t3,
            "tier_2_scheduled_review":        t2,
            "tier_1_monitor":                 t1,
            "tier_0_normal":                  t0,
            "overall_anomaly_rate_pct":       round((t1 + t2 + t3) / safe * 100, 2),
            "high_priority_rate_pct":         round((t2 + t3) / safe * 100, 2),
        }

    def _track_breakdown(self, df: pd.DataFrame) -> dict:
        if df.empty:
            return {}
        return {
            "track_A_fired":    int(df["flag_A"].sum()) if "flag_A" in df else 0,
            "track_B_fired":    int(df["flag_B"].sum()) if "flag_B" in df else 0,
            "track_C_fired":    int(df["flag_C"].sum()) if "flag_C" in df else 0,
            "all_three_fired":  int((df["n_signals"] == 3).sum()),
            "two_tracks_fired": int((df["n_signals"] == 2).sum()),
            "one_track_fired":  int((df["n_signals"] == 1).sum()),
        }

    def _rule_breakdown(self, df: pd.DataFrame) -> dict:
        return {
            label: int(df[col].sum())
            for col, label in _RULE_LABELS.items()
            if col in df
        }

    def _tier3_cases(self, t3: pd.DataFrame) -> list[dict]:
        cases = []
        for idx, row in t3.iterrows():
            rules_triggered = [
                _RULE_LABELS[col]
                for col in _RULE_LABELS
                if row.get(col, 0) == 1
            ]
            case: dict[str, Any] = {
                "record_index":       int(idx),
                "priority":           "INVESTIGATE_IMMEDIATELY",
                "score_A":            round(float(row.get("score_A", 0)), 4),
                "score_B":            round(float(row.get("score_B", 0)), 4),
                "charge_z_score":     round(float(row.get("charge_z_score", 0)), 4),
                "cluster":            int(row.get("cluster", -1)),
                "charge_amount_USD":  round(float(row.get("charge_amount_USD", 0)), 2),
                "payment_amount_USD": round(float(row.get("payment_amount_USD", 0)), 2),
                "age":                int(row.get("age", 0)),
                "flags": {
                    "track_A": int(row.get("flag_A", 0)),
                    "track_B": int(row.get("flag_B", 0)),
                    "track_C": int(row.get("flag_C", 0)),
                },
                "rules_triggered": rules_triggered,
            }
            # attach raw identity fields when available
            for col in _TIER3_IDENTITY:
                val = row.get(col)
                if val is not None and pd.notna(val):
                    case[col] = str(val) if col == "visit_date" else val
            cases.append(case)
        return cases

    def _tier2_summary(self, t2: pd.DataFrame) -> dict:
        if t2.empty:
            return {"count": 0}

        summary: dict[str, Any] = {
            "count": len(t2),
            "avg_charge_USD":  _safe_mean(t2, "charge_amount_USD"),
            "avg_payment_USD": _safe_mean(t2, "payment_amount_USD"),
            "avg_score_A":     _safe_mean(t2, "score_A", decimals=4),
            "avg_score_B":     _safe_mean(t2, "score_B", decimals=4),
        }

        if "department" in t2:
            summary["top_departments"] = (
                t2["department"].value_counts().head(3).index.tolist()
            )
        if "cluster" in t2:
            summary["cluster_distribution"] = (
                t2["cluster"].value_counts()
                .sort_index()
                .apply(int)
                .to_dict()
            )

        top_rules = sorted(
            [
                (_RULE_LABELS[col], int(t2[col].sum()))
                for col in _RULE_LABELS
                if col in t2 and t2[col].sum() > 0
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        summary["top_rules"] = [r[0] for r in top_rules]
        return summary

    def _tier1_summary(self, t1: pd.DataFrame) -> dict:
        if t1.empty:
            return {"count": 0}
        summary: dict[str, Any] = {
            "count":       len(t1),
            "avg_score_A": _safe_mean(t1, "score_A", decimals=4),
            "avg_score_B": _safe_mean(t1, "score_B", decimals=4),
        }
        if "department" in t1:
            summary["top_departments"] = (
                t1["department"].value_counts().head(3).index.tolist()
            )
        return summary

    # ── LLM context ───────────────────────────────────────────────────────────

    def _llm_context(
        self,
        tiers: dict[str, pd.DataFrame],
        run_id: str,
        run_date: str,
        source_name: str,
    ) -> str:
        all_df = tiers["all"]
        t3, t2, t1, t0 = (
            tiers["tier_3"], tiers["tier_2"],
            tiers["tier_1"], tiers["tier_0"],
        )
        total = len(all_df)
        safe  = max(total, 1)

        lines = [
            f"DAILY ANOMALY REPORT | {run_date} | Run: {run_id} | Source: {source_name}",
            f"Total records processed: {total}",
            "",
        ]

        if len(t3):
            pct = round(len(t3) / safe * 100, 1)
            lines.append(
                f"TIER 3 - IMMEDIATE ACTION ({len(t3)} cases, {pct}%): "
                "All three detection tracks fired simultaneously."
            )
            rule_hits = {
                _RULE_LABELS[col]: int(t3[col].sum())
                for col in _RULE_LABELS
                if col in t3 and t3[col].sum() > 0
            }
            if rule_hits:
                top = sorted(rule_hits.items(), key=lambda x: x[1], reverse=True)
                lines.append(
                    f"  Dominant rules: "
                    + ", ".join(f"{r}={c}" for r, c in top) + "."
                )
            if "department" in t3:
                top_depts = t3["department"].value_counts().head(3)
                lines.append(
                    f"  Top departments: "
                    + ", ".join(f"{d}({c})" for d, c in top_depts.items()) + "."
                )
            if "charge_amount_USD" in t3:
                lines.append(
                    f"  Avg charge: ${t3['charge_amount_USD'].mean():.2f}  |  "
                    f"Avg payment: ${t3['payment_amount_USD'].mean():.2f}."
                )
            lines.append("")

        if len(t2):
            pct = round(len(t2) / safe * 100, 1)
            lines.append(f"TIER 2 - SCHEDULED REVIEW ({len(t2)} cases, {pct}%):")
            a = int(t2["flag_A"].sum()) if "flag_A" in t2 else 0
            b = int(t2["flag_B"].sum()) if "flag_B" in t2 else 0
            c = int(t2["flag_C"].sum()) if "flag_C" in t2 else 0
            lines.append(f"  Track signals — A:{a}  B:{b}  C:{c}.")
            if "department" in t2:
                top_depts = t2["department"].value_counts().head(3)
                lines.append(
                    f"  Top departments: "
                    + ", ".join(f"{d}({c})" for d, c in top_depts.items()) + "."
                )
            lines.append("")

        if len(t1):
            pct = round(len(t1) / safe * 100, 1)
            lines.append(
                f"TIER 1 - MONITOR ({len(t1)} cases, {pct}%): "
                "Single-signal anomalies, no immediate action required."
            )
            lines.append("")

        lines.append(
            f"Tier 0 (normal): {len(t0)} records ({round(len(t0)/safe*100,1)}%)."
        )
        return "\n".join(lines)


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_mean(df: pd.DataFrame, col: str, decimals: int = 2) -> float | None:
    if col not in df or df.empty:
        return None
    return round(float(df[col].mean()), decimals)
