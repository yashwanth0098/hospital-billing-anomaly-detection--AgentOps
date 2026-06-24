"""
Daily AgentOps runner — Stage 2 orchestration entry point.

Pipeline per run:
    1. Load raw billing batch (CSV path or DataFrame)
    2. Run Stage 1 InferencePredictor  → scored_df (encoded features + tier)
    3. FlaggedExtractor                → join raw identity cols back, split by tier
    4. DailyReportBuilder              → structured LLM-ready report dict
    5. ReportWriter                    → JSON + markdown to artifacts/agentops_reports/

Usage:
    from src.stage_2_agentops import AgentOpsRunner

    runner = AgentOpsRunner()
    paths  = runner.run("data/daily_batch_20260624.csv")
    # paths = {'json_path': '...run_20260624_143022.json',
    #          'md_path':   '...run_20260624_143022_summary.md'}
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from ..stage_1_mlops.data_ingestion.schema import MODE_IMPUTE_COLUMNS
from ..stage_1_mlops.inference.predictor import InferencePredictor
from .flagged_extractor import FlaggedExtractor
from .daily_report_builder import DailyReportBuilder
from .report_writer import ReportWriter


class AgentOpsRunner:
    """
    Entry point for the daily AgentOps information pipeline.

    Accepts a raw billing batch, runs scoring through the trained Stage 1 models,
    extracts and structures the flagged outcomes into a LLM-ready daily report.
    """

    def __init__(self, output_dir: str | Path | None = None):
        self._predictor = InferencePredictor()
        self._extractor = FlaggedExtractor()
        self._builder   = DailyReportBuilder()
        self._writer    = ReportWriter(output_dir=output_dir) if output_dir else ReportWriter()

    def run(
        self,
        batch: str | Path | pd.DataFrame,
        source_name: str | None = None,
    ) -> dict[str, str]:
        """
        Run the full daily pipeline on a new billing batch.

        Args:
            batch:       CSV file path or a raw billing DataFrame.
            source_name: Label for the batch (defaults to filename or 'inline_dataframe').

        Returns:
            {'json_path': str, 'md_path': str}
        """
        run_id = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")

        raw_df, source_name = self._load(batch, source_name)
        logger.info(
            "AgentOpsRunner.run | run_id={} | source={} | rows={}",
            run_id, source_name, len(raw_df),
        )

        # Step 1 — score through Stage 1 trained models
        # impute_df has NaN diagnosis/visit_reason/payer_name filled (mirrors
        # DataIngestionPipeline cleaning) so frequency maps don't see unknowns.
        # raw_df is kept unchanged for the identity join so reports show original values.
        scored_df = self._predictor.predict(_impute(raw_df))

        # Step 2 — enrich: join raw identity cols back, split by tier
        tiers = self._extractor.extract(raw_df, scored_df)

        # Step 3 — build structured report
        report = self._builder.build(tiers, run_id, source_name)

        # Step 4 — persist
        paths = self._writer.write(report, run_id)

        s = report["anomaly_summary"]
        logger.success(
            "AgentOps complete | run_id={} | T3={} T2={} T1={} T0={} | json={}",
            run_id,
            s["tier_3_investigate_immediately"],
            s["tier_2_scheduled_review"],
            s["tier_1_monitor"],
            s["tier_0_normal"],
            paths["json_path"],
        )
        return paths

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load(
        batch: str | Path | pd.DataFrame,
        source_name: str | None,
    ) -> tuple[pd.DataFrame, str]:
        if isinstance(batch, pd.DataFrame):
            return batch, source_name or "inline_dataframe"
        path = Path(batch)
        df = pd.read_csv(path)
        logger.info(
            "Batch loaded | path={} | rows={} cols={}",
            path, len(df), len(df.columns),
        )
        return df, source_name or path.name


# ── module-level helper ───────────────────────────────────────────────────────

def _impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply mode imputation to the columns that DataIngestionPipeline cleans
    before building frequency maps (diagnosis, visit_reason, payer_name).
    Prevents NaN values in those columns from appearing as unseen categories
    during inference, which would produce spurious WARNING log lines.
    Returns a copy — the caller's raw_df is not modified.
    """
    out = df.copy()
    for col in MODE_IMPUTE_COLUMNS:
        if col in out.columns:
            null_count = int(out[col].isna().sum())
            if null_count:
                mode_val = out[col].mode()
                if not mode_val.empty:
                    out[col] = out[col].fillna(mode_val[0])
                    logger.debug(
                        "Imputed {} NaN(s) in '{}' with mode='{}'",
                        null_count, col, mode_val[0],
                    )
    return out
