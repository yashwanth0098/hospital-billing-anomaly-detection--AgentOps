"""
Data drift detector — uses JSD (primary) + PSI (secondary).

Why JSD + PSI instead of KS + Chi-squared
------------------------------------------
- JSD works on BOTH numeric and categorical with the same formula; KS is
  numeric-only, chi-squared is categorical-only.
- JSD output is bounded [0, 1]: 0 = identical, 1 = completely different.
  KS and chi-squared give p-values that become over-sensitive at large n.
- JSD is symmetric: JSD(A, B) == JSD(B, A). PSI is not symmetric, but it
  is the recognized standard in healthcare billing / insurance compliance.
- Chi-squared breaks when any category has < 5 expected counts — unreliable
  for rare diagnosis codes in this dataset. JSD handles zero-count bins.

Thresholds
----------
JSD  (primary — triggers "drifted" flag):
  < 0.05   -> none
  0.05-0.10 -> moderate
  >= 0.10  -> significant

PSI  (secondary — reported for compliance / auditors):
  < 0.10   -> none
  0.10-0.20 -> moderate
  >= 0.20  -> significant

Two-step usage
--------------
  detector = DataDriftDetector()
  detector.fit(training_df)        # call once; saves reference to disk
  report = detector.detect(new_df) # call on every new production batch
  print(report.summary())
"""
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from .reference_store import ReferenceStore
from .report import ColumnDriftResult, DriftReport


class DataDriftDetector:
    """Detects statistical distribution shift between a reference and a new batch."""

    NUMERIC_COLS = ["age", "charge_amount_USD", "payment_amount_USD"]
    CATEGORICAL_COLS = [
        "gender", "visit_type", "department",
        "payer_type", "claim_status", "is_emergency",
    ]

    # JSD thresholds (primary trigger)
    JSD_MODERATE    = 0.05
    JSD_SIGNIFICANT = 0.10

    # PSI thresholds (secondary / compliance reporting)
    PSI_MODERATE    = 0.10
    PSI_SIGNIFICANT = 0.20

    BINS = 10
    _EPS = 1e-10  # prevents log(0); small enough not to affect results

    def __init__(
        self,
        reference_path: str = "artifacts/reference_distribution.json",
        report_csv_path: str = "artifacts/drift_report.csv",
    ) -> None:
        self._path = Path(reference_path)
        self._report_csv = Path(report_csv_path)
        self._store = ReferenceStore()

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> None:
        """Persist reference distribution from the baseline / training dataset."""
        logger.info(
            "Drift detector: fitting reference distribution | rows={} "
            "| numeric_cols={} | categorical_cols={}",
            len(df), self.NUMERIC_COLS, self.CATEGORICAL_COLS,
        )
        self._store.save(
            df,
            numeric_cols=self.NUMERIC_COLS,
            categorical_cols=self.CATEGORICAL_COLS,
            path=self._path,
        )
        logger.success(
            "Reference distribution saved | path={} | {} numeric + {} categorical columns",
            self._path,
            len(self.NUMERIC_COLS),
            len(self.CATEGORICAL_COLS),
        )

    def detect(self, df: pd.DataFrame) -> DriftReport:
        """Compare df against the saved reference, return a DriftReport and save CSV."""
        logger.info(
            "Drift detection started | batch_rows={} | reference={}",
            len(df), self._path,
        )
        ref = self._store.load(self._path)
        results: dict[str, ColumnDriftResult] = {}

        for col in self.NUMERIC_COLS:
            if col in df.columns and col in ref["numeric"]:
                results[col] = self._check_numeric(col, ref["numeric"][col], df[col])

        for col in self.CATEGORICAL_COLS:
            if col in df.columns and col in ref["categorical"]:
                results[col] = self._check_categorical(
                    col, ref["categorical"][col], df[col]
                )

        drifted = [c for c, r in results.items() if r.drifted]

        if drifted:
            logger.warning(
                "Drift detection complete | {} / {} column(s) drifted: {}",
                len(drifted), len(results), drifted,
            )
        else:
            logger.success(
                "Drift detection complete | no drift detected | {} columns checked",
                len(results),
            )

        report = DriftReport(
            total_columns_checked=len(results),
            drifted_columns=drifted,
            overall_drift=bool(drifted),
            reference_rows=int(ref.get("row_count", 0)),
            batch_rows=len(df),
            results=results,
        )

        # persist every run to CSV (appends — builds a time-series history)
        saved_path = report.save_csv(self._report_csv)
        logger.info(
            "Drift report appended to CSV | path={} | rows_added={}",
            saved_path, len(results),
        )

        return report

    # ── numeric ───────────────────────────────────────────────────────────────

    def _check_numeric(
        self, col: str, ref_data: dict, cur_series: pd.Series
    ) -> ColumnDriftResult:
        ref_sample = np.array(ref_data["sample"], dtype=float)
        cur_sample = cur_series.dropna().astype(float).values

        jsd = self._jsd_numeric(ref_data["bin_edges"], ref_sample, cur_sample)
        psi = self._psi_numeric(ref_data["bin_edges"], ref_sample, cur_sample)
        drifted  = jsd >= self.JSD_MODERATE
        severity = self._jsd_severity(jsd)

        _log = logger.warning if drifted else logger.debug
        _log(
            "Column check | col={} type=numeric | JSD={:.4f} PSI={:.4f} | severity={} | {}",
            col, jsd, psi, severity, "DRIFT" if drifted else "ok",
        )

        return ColumnDriftResult(
            column=col,
            col_type="numeric",
            jsd_score=round(float(jsd), 6),
            psi_score=round(float(psi), 6),
            drifted=drifted,
            severity=severity,
        )

    # ── categorical ───────────────────────────────────────────────────────────

    def _check_categorical(
        self, col: str, ref_counts: dict[str, int], cur_series: pd.Series
    ) -> ColumnDriftResult:
        cur_clean = cur_series.dropna()
        all_cats = sorted(set(ref_counts) | set(cur_clean.unique()))

        ref_freq = np.array([ref_counts.get(c, 0) for c in all_cats], dtype=float)
        cur_freq = np.array(
            [cur_clean.value_counts().get(c, 0) for c in all_cats], dtype=float
        )

        jsd = self._jsd_categorical(ref_freq, cur_freq)
        psi = self._psi_categorical(ref_freq, cur_freq)
        drifted  = jsd >= self.JSD_MODERATE
        severity = self._jsd_severity(jsd)

        _log = logger.warning if drifted else logger.debug
        _log(
            "Column check | col={} type=categorical | JSD={:.4f} PSI={:.4f} | severity={} | {}",
            col, jsd, psi, severity, "DRIFT" if drifted else "ok",
        )

        return ColumnDriftResult(
            column=col,
            col_type="categorical",
            jsd_score=round(float(jsd), 6),
            psi_score=round(float(psi), 6),
            drifted=drifted,
            severity=severity,
        )

    # ── JSD implementations ───────────────────────────────────────────────────

    def _jsd_numeric(
        self,
        bin_edges: list[float],
        ref_sample: np.ndarray,
        cur_sample: np.ndarray,
    ) -> float:
        """
        Jensen-Shannon Distance for a numeric column.

        Formula: JSD(P, Q) = sqrt( [KL(P||M) + KL(Q||M)] / 2 )
        where M = (P + Q) / 2  and  KL(P||Q) = sum( P * log(P/Q) )

        Bins the two samples using pre-computed percentile edges so that
        each bin contains roughly equal reference counts.
        """
        edges = np.array(bin_edges, dtype=float)
        edges[0]  = min(edges[0],  cur_sample.min()) - self._EPS
        edges[-1] = max(edges[-1], cur_sample.max()) + self._EPS

        p_counts, _ = np.histogram(ref_sample, bins=edges)
        q_counts, _ = np.histogram(cur_sample, bins=edges)

        p = (p_counts + self._EPS) / (p_counts.sum() + self._EPS * len(p_counts))
        q = (q_counts + self._EPS) / (q_counts.sum() + self._EPS * len(q_counts))

        return float(self._jsd_from_pq(p, q))

    def _jsd_categorical(
        self, ref_freq: np.ndarray, cur_freq: np.ndarray
    ) -> float:
        """Jensen-Shannon Distance for a categorical column using raw counts."""
        p = (ref_freq + self._EPS) / (ref_freq.sum() + self._EPS * len(ref_freq))
        q = (cur_freq + self._EPS) / (cur_freq.sum() + self._EPS * len(cur_freq))
        return float(self._jsd_from_pq(p, q))

    @staticmethod
    def _jsd_from_pq(p: np.ndarray, q: np.ndarray) -> float:
        """
        Core JSD calculation given two already-normalised probability arrays.

        JSD divergence = ( KL(P||M) + KL(Q||M) ) / 2
        JSD distance   = sqrt( JSD divergence )       <- what we return

        The distance form is preferred: it satisfies the triangle inequality
        and maps cleanly to [0, 1] when using log base 2.
        """
        m = (p + q) / 2.0
        kl_pm = np.sum(p * np.log2(p / m))
        kl_qm = np.sum(q * np.log2(q / m))
        jsd_divergence = (kl_pm + kl_qm) / 2.0
        jsd_distance = np.sqrt(max(0.0, jsd_divergence))  # clip float noise
        return float(np.clip(jsd_distance, 0.0, 1.0))

    # ── PSI implementations ───────────────────────────────────────────────────

    def _psi_numeric(
        self,
        bin_edges: list[float],
        ref_sample: np.ndarray,
        cur_sample: np.ndarray,
    ) -> float:
        """PSI = sum( (cur% - ref%) * ln(cur% / ref%) ) for a numeric column."""
        edges = np.array(bin_edges, dtype=float)
        edges[0]  = min(edges[0],  cur_sample.min()) - self._EPS
        edges[-1] = max(edges[-1], cur_sample.max()) + self._EPS

        ref_c, _ = np.histogram(ref_sample, bins=edges)
        cur_c, _ = np.histogram(cur_sample, bins=edges)

        ref_pct = (ref_c + self._EPS) / len(ref_sample)
        cur_pct = (cur_c + self._EPS) / len(cur_sample)

        return float(max(0.0, np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))))

    def _psi_categorical(
        self, ref_freq: np.ndarray, cur_freq: np.ndarray
    ) -> float:
        """PSI for a categorical column using raw count arrays."""
        ref_pct = (ref_freq + self._EPS) / (ref_freq.sum() + self._EPS * len(ref_freq))
        cur_pct_raw = cur_freq / max(cur_freq.sum(), 1)
        cur_pct = (cur_pct_raw + self._EPS) / (cur_pct_raw + self._EPS).sum()
        return float(max(0.0, np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))))

    # ── severity maps ─────────────────────────────────────────────────────────

    def _jsd_severity(self, jsd: float) -> str:
        """Map a JSD score to a severity label."""
        if jsd >= self.JSD_SIGNIFICANT:
            return "significant"
        if jsd >= self.JSD_MODERATE:
            return "moderate"
        return "none"
