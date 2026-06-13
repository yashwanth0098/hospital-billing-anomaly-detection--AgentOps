"""Dataclasses for the drift detection report."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd


@dataclass
class ColumnDriftResult:
    """Drift test result for a single column."""

    column: str
    col_type: str     # "numeric" | "categorical"
    jsd_score: float  # Jensen-Shannon Distance [0, 1]  — primary trigger
    psi_score: float  # Population Stability Index      — industry standard
    drifted: bool
    severity: str     # "none" | "moderate" | "significant"


@dataclass
class DriftReport:
    """Aggregated drift report across all monitored columns."""

    total_columns_checked: int
    drifted_columns: list[str]
    overall_drift: bool
    reference_rows: int
    batch_rows: int
    run_timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    results: dict[str, ColumnDriftResult] = field(default_factory=dict)

    # ── human-readable summary ────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a human-readable multi-line summary."""
        drift_flag = "YES -- action required" if self.overall_drift else "NO -- data is stable"
        lines = [
            f"Columns checked : {self.total_columns_checked}",
            f"Overall drift   : {drift_flag}",
            f"Drifted columns : {self.drifted_columns or ['none']}",
            "",
            f"  {'Column':<26} {'Type':<13} {'JSD':>7} {'PSI':>7} {'Severity':<14} Status",
            "  " + "-" * 72,
        ]
        for col, r in self.results.items():
            status = "DRIFT" if r.drifted else "ok"
            lines.append(
                f"  {col:<26} {r.col_type:<13} {r.jsd_score:>7.4f} "
                f"{r.psi_score:>7.4f} {r.severity:<14} {status}"
            )
        return "\n".join(lines)

    # ── CSV export ────────────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the per-column results into a flat DataFrame.

        Each row represents one column's drift result for this run.
        The run_timestamp, reference_rows, and batch_rows are repeated
        on every row so the CSV can be filtered or grouped over time.
        """
        rows = []
        for col, r in self.results.items():
            rows.append({
                "run_timestamp":   self.run_timestamp,
                "reference_rows":  self.reference_rows,
                "batch_rows":      self.batch_rows,
                "column":          r.column,
                "col_type":        r.col_type,
                "jsd_score":       r.jsd_score,
                "psi_score":       r.psi_score,
                "severity":        r.severity,
                "drifted":         r.drifted,
                "overall_drift":   self.overall_drift,
            })
        return pd.DataFrame(rows)

    def save_csv(self, path: str | Path) -> Path:
        """
        Append this run's results to a CSV file.

        - If the file does not exist it is created with a header row.
        - If it already exists the new rows are appended without rewriting
          the header, so the file grows into a time-series history of
          every drift-detection run.

        Returns the resolved path for logging.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        df = self.to_dataframe()
        file_exists = path.exists() and path.stat().st_size > 0

        df.to_csv(
            path,
            mode="a",
            header=not file_exists,
            index=False,
            encoding="utf-8",
        )
        return path
