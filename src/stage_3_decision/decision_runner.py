"""
Stage 3 — Decision Making orchestrator.

Takes the outputs of BOTH Stage 1 (evaluation report) and Stage 2 (daily
agentops report) and runs them through an intelligent stakeholder LLM that
acts as a Senior Compliance Officer making a real-time daily review decision.

Pipeline per run:
    Stage 1 evaluation_report.json  ─┐
                                      ├─► ContextBuilder ──► situation_brief (str)
    Stage 2 agentops report (dict)   ─┘
                                                 │
                                                 ▼
                                         DecisionAgent
                                    (llama3.2 via Ollama)
                                    reads full brief once,
                                    reasons holistically,
                                    outputs complete decision
                                                 │
                                                 ▼
                                        DecisionWriter
                               artifacts/decision_reports/
                               <run_id>_decisions.json
                               <run_id>_decisions.md

Usage:
    from src.stage_3_decision import DecisionRunner

    runner = DecisionRunner()

    # From a Stage 2 JSON file on disk:
    paths = runner.run("artifacts/agentops_reports/run_20260624_143022.json")

    # Directly from a Stage 2 report dict (e.g. chained from AgentOpsRunner):
    paths = runner.run(stage2_report_dict)

    # paths = {
    #   'json_path': 'artifacts/decision_reports/run_20260624_143022_decisions.json',
    #   'md_path':   'artifacts/decision_reports/run_20260624_143022_decisions.md',
    # }
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from .context_builder import ContextBuilder
from .decision_agent import DecisionAgent
from .decision_writer import DecisionWriter
from .llm_client import OllamaClient


class DecisionRunner:
    """
    Entry point for Stage 3 Decision Making.

    Args:
        model:            Ollama model tag (default: llama3.2).
        host:             Ollama server host (default: http://localhost:11434).
        output_dir:       Directory for decision reports.
        eval_report_path: Override path to Stage 1 evaluation_report.json.
    """

    def __init__(
        self,
        model:            str           = "llama3.2",
        host:             str           = "http://localhost:11434",
        output_dir:       str | Path | None = None,
        eval_report_path: str | Path | None = None,
    ) -> None:
        llm = OllamaClient(model=model, host=host)
        self._context_builder = ContextBuilder(eval_report_path=eval_report_path)
        self._agent           = DecisionAgent(llm=llm)
        self._writer          = DecisionWriter(output_dir=output_dir)
        self._model           = model

    def run(
        self,
        stage2_report: str | Path | dict[str, Any],
    ) -> dict[str, str]:
        """
        Run the full Stage 3 decision pipeline on a Stage 2 report.

        Args:
            stage2_report: Path to a Stage 2 JSON file OR a pre-loaded dict.

        Returns:
            {'json_path': str, 'md_path': str}
        """
        report, source_path = self._load(stage2_report)
        run_id   = report.get("run_id", "run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        run_date = report.get("run_date", datetime.now().strftime("%Y-%m-%d"))

        t3_count = len(report.get("tier_3_cases", []))
        t2_count = int(report.get("tier_2_summary", {}).get("count", 0))

        logger.info(
            "DecisionRunner.run | run_id={} | model={} | tier3={} tier2={}",
            run_id, self._model, t3_count, t2_count,
        )

        # Step 1 — build the full situational brief from Stage 1 + Stage 2
        situation_brief = self._context_builder.build(report)

        # Step 2 — the stakeholder LLM reads the brief and makes decisions
        decision_brief = self._agent.decide(situation_brief)

        # Step 3 — assemble the final decision report
        decision_report = {
            "run_id":          run_id,
            "run_date":        run_date,
            "stage":           "stage_3_decision",
            "model":           self._model,
            "source_report":   source_path,
            "decision_summary": {
                "tier3_count":               t3_count,
                "tier2_count":               t2_count,
                "risk_level":                decision_brief.get("risk_level", "HIGH"),
                "immediate_action_count":    len(decision_brief.get("immediate_actions", [])),
                "department_directive_count":len(decision_brief.get("department_directives", [])),
                "payer_directive_count":     len(decision_brief.get("payer_directives", [])),
            },
            "decision_brief":  decision_brief,
            "situation_brief": situation_brief,
        }

        # Step 4 — persist JSON + markdown
        paths = self._writer.write(decision_report, run_id)

        logger.success(
            "Stage 3 complete | run_id={} | risk={} | actions={} | "
            "dept_directives={} | json={}",
            run_id,
            decision_brief.get("risk_level"),
            len(decision_brief.get("immediate_actions", [])),
            len(decision_brief.get("department_directives", [])),
            paths["json_path"],
        )
        return paths

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load(
        stage2_report: str | Path | dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        if isinstance(stage2_report, dict):
            return stage2_report, "inline_dict"
        path = Path(stage2_report)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        logger.info("Stage 2 report loaded | path={} | run_id={}", path, data.get("run_id"))
        return data, str(path)
