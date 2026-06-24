"""
Persists the Stage 3 stakeholder decision brief as:
  - <run_id>_decisions.json   — machine-readable, API/dashboard-ready
  - <run_id>_decisions.md     — human-readable compliance decision brief

Both land in artifacts/decision_reports/ by default.

The markdown format is intentionally structured as a real compliance memo —
immediate actions first, then department directives, then management summary —
because that is the order a real stakeholder reads in a live situation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

_DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[2] / "artifacts" / "decision_reports"
)

_RISK_BADGE = {
    "CRITICAL": "🔴 CRITICAL",
    "HIGH":     "🟠 HIGH",
    "MEDIUM":   "🟡 MEDIUM",
    "LOW":      "🟢 LOW",
}

_HEALTH_BADGE = {
    "HEALTHY":             "✅ HEALTHY",
    "NEEDS_REVIEW":        "⚠️  NEEDS REVIEW",
    "RETRAIN_RECOMMENDED": "🔁 RETRAIN RECOMMENDED",
}


class DecisionWriter:
    """Writes the Stage 3 decision report dict to JSON + markdown."""

    def __init__(self, output_dir: str | Path | None = None) -> None:
        self.output_dir = Path(output_dir) if output_dir else _DEFAULT_OUTPUT_DIR

    def write(self, report: dict[str, Any], run_id: str) -> dict[str, str]:
        """
        Write JSON and markdown decision reports.

        Returns:
            {'json_path': str, 'md_path': str}
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        json_path = self.output_dir / f"{run_id}_decisions.json"
        md_path   = self.output_dir / f"{run_id}_decisions.md"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        logger.success("Decision JSON written | path={}", json_path)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(_to_markdown(report))
        logger.success("Decision markdown written | path={}", md_path)

        return {"json_path": str(json_path), "md_path": str(md_path)}


# ── markdown renderer ─────────────────────────────────────────────────────────

def _to_markdown(report: dict[str, Any]) -> str:
    decision = report.get("decision_brief", {})
    ds       = report.get("decision_summary", {})
    risk     = decision.get("risk_level", "HIGH")
    badge    = _RISK_BADGE.get(risk, risk)

    lines: list[str] = [
        f"# Compliance Decision Brief — {report.get('run_date', '?')}",
        f"**Run ID:** {report.get('run_id', '?')}  |  "
        f"**Risk Level:** {badge}  |  "
        f"**Model:** {report.get('model', '?')}",
        f"**Source:** `{report.get('source_report', '?')}`",
        "",
    ]

    # ── Situation Assessment ──────────────────────────────────────────────────
    assessment = decision.get("situation_assessment", "")
    if assessment:
        lines += [
            "## Situation Assessment",
            assessment,
            "",
        ]

    # ── Decision Summary table ────────────────────────────────────────────────
    lines += [
        "## Decision Summary",
        "| Metric | Value |",
        "|--------|------:|",
        f"| Tier 3 critical cases | {ds.get('tier3_count', 0)} |",
        f"| Tier 2 cases          | {ds.get('tier2_count', 0)} |",
        f"| Immediate actions     | {ds.get('immediate_action_count', 0)} |",
        f"| Departments directed  | {ds.get('department_directive_count', 0)} |",
        f"| Payer directives      | {ds.get('payer_directive_count', 0)} |",
        "",
    ]

    # ── Immediate Actions ─────────────────────────────────────────────────────
    actions = decision.get("immediate_actions", [])
    if actions:
        lines.append(f"## Immediate Actions ({len(actions)} — act today)")
        lines.append("")
        for a in actions:
            lines.append(
                f"### Action {a.get('priority', '?')} — "
                f"Case: `{a.get('case_ref', 'N/A')}`"
            )
            lines.append(f"**Action:** {a.get('action', '—')}")
            lines.append(f"**Owner:** {a.get('owner', '—')}")
            lines.append(f"**Reason:** {a.get('reason', '—')}")
            lines.append("")

    # ── Department Directives ─────────────────────────────────────────────────
    dept_dirs = decision.get("department_directives", [])
    if dept_dirs:
        lines.append(f"## Department Directives ({len(dept_dirs)})")
        lines.append("")
        for d in dept_dirs:
            lines.append(f"### {d.get('department', '?')}")
            lines.append(f"**Finding:** {d.get('finding', '—')}")
            lines.append(f"**Directive:** {d.get('directive', '—')}")
            lines.append(f"**Timeline:** {d.get('timeline', '—')}")
            lines.append("")

    # ── Payer Directives ──────────────────────────────────────────────────────
    payer_dirs = decision.get("payer_directives", [])
    if payer_dirs:
        lines.append(f"## Payer Directives ({len(payer_dirs)})")
        lines.append("")
        for p in payer_dirs:
            lines.append(f"### {p.get('payer', '?')}")
            lines.append(f"**Finding:** {p.get('finding', '—')}")
            lines.append(f"**Directive:** {p.get('directive', '—')}")
            lines.append("")

    # ── Pipeline Feedback ─────────────────────────────────────────────────────
    pf = decision.get("pipeline_feedback", {})
    if pf:
        health = pf.get("model_health", "HEALTHY")
        lines += [
            "## Pipeline Feedback",
            f"**Model health:** {_HEALTH_BADGE.get(health, health)}",
            "",
        ]
        for obs in pf.get("observations", []):
            lines.append(f"- {obs}")
        for rec in pf.get("threshold_recommendations", []):
            lines.append(f"- Threshold: {rec}")
        for flag in pf.get("data_quality_flags", []):
            lines.append(f"- Data quality: {flag}")
        lines.append("")

    # ── Executive Summary ─────────────────────────────────────────────────────
    summary = decision.get("executive_summary", "")
    if summary:
        lines += [
            "## Executive Summary",
            summary,
            "",
        ]

    return "\n".join(lines)
