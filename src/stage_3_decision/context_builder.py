"""
Assembles a comprehensive situation brief for the decision agent LLM.

Reads the outputs of BOTH Stage 1 (evaluation_report.json) and Stage 2
(daily agentops report dict) and composes them into one structured briefing
document — exactly what a real compliance officer would read before their
daily review meeting.

The brief covers:
  - What the pipeline detected today (tier counts, anomaly rates)
  - How the three detection tracks performed (ML + business rules)
  - Every Tier 3 critical case with full billing + signal detail
  - Tier 2 aggregate patterns (department, payer, rule trends)
  - Model health context from Stage 1 training evaluation
  - Any data drift signals
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

_ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
_EVAL_REPORT_PATH = _ARTIFACTS_DIR / "evaluation_report.json"


class ContextBuilder:
    """
    Constructs the full situation brief passed to DecisionAgent.

    Args:
        eval_report_path: Path to Stage 1 evaluation_report.json.
                          If None, model health context is omitted gracefully.
    """

    def __init__(self, eval_report_path: str | Path | None = None) -> None:
        self._eval_path = Path(eval_report_path) if eval_report_path else _EVAL_REPORT_PATH

    def build(self, stage2_report: dict[str, Any]) -> str:
        """
        Build the full situation brief string from Stage 1 + Stage 2 outputs.

        Args:
            stage2_report: The complete Stage 2 report dict (from AgentOpsRunner).

        Returns:
            A structured plain-text situation brief ready for LLM consumption.
        """
        eval_report = self._load_eval_report()

        sections: list[str] = [
            self._header(stage2_report),
            self._anomaly_overview(stage2_report),
            self._detection_evidence(stage2_report),
            self._critical_cases(stage2_report),
            self._tier2_patterns(stage2_report),
            self._model_health(eval_report),
        ]

        brief = "\n\n".join(s for s in sections if s.strip())
        logger.info(
            "ContextBuilder.build | run_id={} | brief_chars={}",
            stage2_report.get("run_id", "?"), len(brief),
        )
        return brief

    # ── sections ──────────────────────────────────────────────────────────────

    def _header(self, r: dict[str, Any]) -> str:
        bi = r.get("batch_info", {})
        return (
            f"=== DAILY COMPLIANCE SITUATION BRIEF ===\n"
            f"Date        : {r.get('run_date', 'N/A')}\n"
            f"Run ID      : {r.get('run_id', 'N/A')}\n"
            f"Data source : {bi.get('source', 'N/A')}\n"
            f"Total billing records processed : {bi.get('total_records', 0):,}"
        )

    def _anomaly_overview(self, r: dict[str, Any]) -> str:
        s = r.get("anomaly_summary", {})
        if not s:
            return ""
        lines = [
            "--- ANOMALY OVERVIEW ---",
            f"Tier 3  CRITICAL  (all 3 tracks fired — investigate immediately) : {s.get('tier_3_investigate_immediately', 0)}",
            f"Tier 2  HIGH      (2 of 3 tracks fired — scheduled review)       : {s.get('tier_2_scheduled_review', 0)}",
            f"Tier 1  LOW       (1 of 3 tracks fired — monitor)                : {s.get('tier_1_monitor', 0)}",
            f"Tier 0  NORMAL    (no tracks fired)                              : {s.get('tier_0_normal', 0)}",
            f"Overall anomaly rate    : {s.get('overall_anomaly_rate_pct', 0):.1f}%",
            f"High-priority rate (T2+T3) : {s.get('high_priority_rate_pct', 0):.1f}%",
        ]
        return "\n".join(lines)

    def _detection_evidence(self, r: dict[str, Any]) -> str:
        tb = r.get("track_breakdown", {})
        rb = r.get("rule_breakdown", {})
        if not tb and not rb:
            return ""

        lines = ["--- DETECTION EVIDENCE (HOW CASES WERE FLAGGED) ---"]

        if tb:
            lines += [
                "ML Track A  (cluster isolation forest) fired on : "
                f"{tb.get('track_A_fired', 0)} records",
                "ML Track B  (global  isolation forest) fired on : "
                f"{tb.get('track_B_fired', 0)} records",
                "Rules Track C (business rules)         fired on : "
                f"{tb.get('track_C_fired', 0)} records",
                f"All 3 tracks agreed (Tier 3)  : {tb.get('all_three_fired', 0)} records",
                f"Any 2 tracks agreed (Tier 2)  : {tb.get('two_tracks_fired', 0)} records",
            ]

        if rb:
            lines.append("Business rule breakdown:")
            for rule, count in rb.items():
                lines.append(f"  {rule}: {count} records")

        return "\n".join(lines)

    def _critical_cases(self, r: dict[str, Any]) -> str:
        cases = r.get("tier_3_cases", [])
        if not cases:
            return "--- TIER 3 CRITICAL CASES ---\nNo Tier 3 cases in this run."

        lines = [f"--- TIER 3 CRITICAL CASES ({len(cases)} records — require immediate decision) ---"]
        for i, c in enumerate(cases, 1):
            charge  = float(c.get("charge_amount_USD", 0))
            payment = float(c.get("payment_amount_USD", 0))
            ratio   = f"{payment/charge:.1%}" if charge > 0 else "N/A"
            rules   = ", ".join(c.get("rules_triggered", [])) or "none"

            lines.append(
                f"\nCASE {i} | record #{c.get('record_index', '?')}"
            )
            lines.append(
                f"  Patient     : {c.get('patient_id', 'N/A')}  |  "
                f"Age: {c.get('age', 'N/A')}  |  "
                f"Department: {c.get('department', 'N/A')}"
            )
            lines.append(
                f"  Diagnosis   : {c.get('diagnosis', 'N/A')}  |  "
                f"Visit: {c.get('visit_type', 'N/A')}  |  "
                f"Emergency: {c.get('is_emergency', 'N/A')}"
            )
            lines.append(
                f"  Claim status: {c.get('claim_status', 'N/A')}  |  "
                f"Payer: {c.get('payer_name', 'N/A')} ({c.get('payer_type', 'N/A')})"
            )
            lines.append(
                f"  Charge: ${charge:,.2f}  |  "
                f"Payment received: ${payment:,.2f}  |  "
                f"Payment ratio: {ratio}"
            )
            lines.append(
                f"  ML score A: {c.get('score_A', '?')}  |  "
                f"ML score B: {c.get('score_B', '?')}  |  "
                f"Charge z-score: {c.get('charge_z_score', '?')} (vs cluster peers)"
            )
            lines.append(f"  Business rules triggered: {rules}")
            lines.append(f"  Cluster assignment: {c.get('cluster', 'N/A')}")

        return "\n".join(lines)

    def _tier2_patterns(self, r: dict[str, Any]) -> str:
        t2 = r.get("tier_2_summary", {})
        count = int(t2.get("count", 0))
        if count == 0:
            return ""

        avg_charge  = float(t2.get("avg_charge_USD") or 0)
        avg_payment = float(t2.get("avg_payment_USD") or 0)
        ratio       = f"{avg_payment/avg_charge:.1%}" if avg_charge > 0 else "N/A"

        lines = [
            f"--- TIER 2 PATTERNS ({count} records — scheduled review) ---",
            f"Average charge           : ${avg_charge:,.2f}",
            f"Average payment received : ${avg_payment:,.2f}",
            f"Average payment ratio    : {ratio}",
            f"Avg ML score A           : {t2.get('avg_score_A', 'N/A')}",
            f"Avg ML score B           : {t2.get('avg_score_B', 'N/A')}",
        ]
        if t2.get("top_departments"):
            lines.append(
                f"Top departments in Tier 2: {', '.join(str(d) for d in t2['top_departments'])}"
            )
        if t2.get("top_rules"):
            lines.append(
                f"Most triggered rules     : {', '.join(t2['top_rules'])}"
            )
        if t2.get("cluster_distribution"):
            dist = ", ".join(
                f"cluster {k}: {v}" for k, v in t2["cluster_distribution"].items()
            )
            lines.append(f"Cluster distribution     : {dist}")

        return "\n".join(lines)

    def _model_health(self, eval_report: dict[str, Any] | None) -> str:
        if not eval_report:
            return ""
        td = eval_report.get("tier_distribution", {})
        if not td:
            return ""

        lines = [
            "--- MODEL HEALTH CONTEXT (from Stage 1 training evaluation) ---",
        ]
        for tier, info in td.items():
            count = info.get("count", 0) if isinstance(info, dict) else info
            lines.append(f"  Training {tier}: {count} records")

        notes = eval_report.get("notes") or eval_report.get("evaluation_notes")
        if notes:
            lines.append(f"Evaluation notes: {notes}")

        return "\n".join(lines)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _load_eval_report(self) -> dict[str, Any] | None:
        if not self._eval_path.exists():
            logger.debug(
                "ContextBuilder | eval report not found at {} — skipping model health section",
                self._eval_path,
            )
            return None
        try:
            with open(self._eval_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("ContextBuilder | failed to load eval report | error={}", exc)
            return None
