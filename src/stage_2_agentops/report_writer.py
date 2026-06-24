"""
Persists the daily AgentOps report as:
  - <run_id>.json          — machine-readable, LLM-ready structured report
  - <run_id>_summary.md   — human-readable markdown for audit / review

Both land in artifacts/agentops_reports/ by default.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "agentops_reports"


class ReportWriter:
    """Writes the structured report dict to JSON + markdown."""

    def __init__(self, output_dir: str | Path | None = None):
        self.output_dir = Path(output_dir) if output_dir else _DEFAULT_OUTPUT_DIR

    def write(self, report: dict[str, Any], run_id: str) -> dict[str, str]:
        """
        Write JSON and markdown reports.

        Returns:
            {'json_path': str, 'md_path': str}
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        json_path = self.output_dir / f"{run_id}.json"
        md_path   = self.output_dir / f"{run_id}_summary.md"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        logger.success("Report JSON written | path={}", json_path)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(_to_markdown(report))
        logger.success("Report markdown written | path={}", md_path)

        return {"json_path": str(json_path), "md_path": str(md_path)}


# ── markdown renderer ─────────────────────────────────────────────────────────

def _to_markdown(report: dict[str, Any]) -> str:
    bi = report["batch_info"]
    s  = report["anomaly_summary"]

    lines: list[str] = [
        f"# Daily Anomaly Report — {report['run_date']}",
        f"**Run ID:** {report['run_id']}  ",
        f"**Source:** {bi['source']}  ",
        f"**Records processed:** {bi['total_records']}",
        "",
        "## Priority Summary",
        "| Tier | Label | Count | Rate |",
        "|------|-------|------:|-----:|",
        f"| 3 | Investigate Immediately | {s['tier_3_investigate_immediately']} | — |",
        f"| 2 | Scheduled Review | {s['tier_2_scheduled_review']} | — |",
        f"| 1 | Monitor | {s['tier_1_monitor']} | — |",
        f"| 0 | Normal | {s['tier_0_normal']} | — |",
        f"| — | **Overall anomaly rate** | — | **{s['overall_anomaly_rate_pct']}%** |",
        f"| — | **High-priority rate (T2+T3)** | — | **{s['high_priority_rate_pct']}%** |",
        "",
    ]

    # Track breakdown
    tb = report.get("track_breakdown", {})
    if tb:
        lines += [
            "## Track Breakdown",
            f"- Track A (cluster IsolationForest): {tb.get('track_A_fired', 0)} records",
            f"- Track B (global IsolationForest): {tb.get('track_B_fired', 0)} records",
            f"- Track C (business rules): {tb.get('track_C_fired', 0)} records",
            f"- All 3 tracks fired: {tb.get('all_three_fired', 0)} records",
            f"- Exactly 2 tracks fired: {tb.get('two_tracks_fired', 0)} records",
            f"- Exactly 1 track fired: {tb.get('one_track_fired', 0)} records",
            "",
        ]

    # Rule breakdown
    rb = report.get("rule_breakdown", {})
    if rb:
        lines += ["## Business Rule Breakdown"]
        for label, count in rb.items():
            lines.append(f"- `{label}`: {count} records")
        lines.append("")

    # Tier 3 cases — full detail
    t3_cases = report.get("tier_3_cases", [])
    if t3_cases:
        lines.append(f"## Tier 3 — Investigate Immediately ({len(t3_cases)} cases)")
        lines.append("")
        for i, case in enumerate(t3_cases, 1):
            lines.append(f"### Case {i} (record #{case.get('record_index', '?')})")
            for field in [
                "patient_id", "appointment_id", "department", "diagnosis",
                "visit_type", "visit_reason", "claim_status", "payer_type",
                "payer_name", "is_emergency", "physician_id",
            ]:
                if field in case:
                    lines.append(f"- **{field}:** {case[field]}")
            lines.append(f"- **charge_amount_USD:** ${case.get('charge_amount_USD', 0):,.2f}")
            lines.append(f"- **payment_amount_USD:** ${case.get('payment_amount_USD', 0):,.2f}")
            lines.append(f"- **score_A:** {case.get('score_A')}  |  **score_B:** {case.get('score_B')}  |  **charge_z_score:** {case.get('charge_z_score')}")
            lines.append(f"- **cluster:** {case.get('cluster')}")
            rules = case.get("rules_triggered", [])
            lines.append(f"- **rules_triggered:** {', '.join(rules) if rules else 'none'}")
            lines.append("")

    # Tier 2 summary
    t2s = report.get("tier_2_summary", {})
    if t2s.get("count", 0):
        lines += [
            f"## Tier 2 — Scheduled Review ({t2s['count']} cases)",
            f"- Avg charge: ${t2s.get('avg_charge_USD') or 0:,.2f}",
            f"- Avg payment: ${t2s.get('avg_payment_USD') or 0:,.2f}",
            f"- Avg score A: {t2s.get('avg_score_A')}",
            f"- Avg score B: {t2s.get('avg_score_B')}",
        ]
        if "top_departments" in t2s:
            lines.append(f"- Top departments: {', '.join(str(d) for d in t2s['top_departments'])}")
        if "top_rules" in t2s:
            lines.append(f"- Top rules: {', '.join(t2s['top_rules'])}")
        if "cluster_distribution" in t2s:
            dist = ", ".join(f"cluster {k}: {v}" for k, v in t2s["cluster_distribution"].items())
            lines.append(f"- Cluster distribution: {dist}")
        lines.append("")

    # Tier 1 summary
    t1s = report.get("tier_1_summary", {})
    if t1s.get("count", 0):
        lines += [
            f"## Tier 1 — Monitor ({t1s['count']} cases)",
            f"- Avg score A: {t1s.get('avg_score_A')}",
            f"- Avg score B: {t1s.get('avg_score_B')}",
        ]
        if "top_departments" in t1s:
            lines.append(f"- Top departments: {', '.join(str(d) for d in t1s['top_departments'])}")
        lines.append("")

    # LLM context block — what the downstream LLM will receive
    lines += [
        "## LLM Context",
        "```",
        report.get("llm_context", ""),
        "```",
    ]

    return "\n".join(lines)
