import re

import pandas as pd
from loguru import logger

from .schema import (
    EXPECTED_COLUMNS,
    ALLOWED_GENDER, ALLOWED_VISIT_TYPE, ALLOWED_IS_EMERGENCY,
    ALLOWED_PAYER_TYPE, ALLOWED_CLAIM_STATUS, ALLOWED_DEPARTMENTS,
    NUMERIC_COLUMNS, AGE_MIN, AGE_MAX, CHARGE_MIN, PAYMENT_MIN,
    APPOINTMENT_ID_PATTERN,
)

_TOTAL_BUSINESS_CHECKS = 12


class DataValidator:
    """
    Validates the hospital billing dataframe against schema and business rules.

    All checks run before raising, so the caller receives a complete list of
    every violation found — not just the first one.
    """

    def validate(self, df: pd.DataFrame) -> None:
        logger.info(
            "Validation started | rows={} cols={}",
            len(df), len(df.columns),
        )

        # Critical structural checks — cannot proceed if these fail.
        logger.debug("Check [1/2 structural]: dataframe is not empty")
        self._check_not_empty(df)

        logger.debug("Check [2/2 structural]: all {} expected columns present", len(EXPECTED_COLUMNS))
        self._check_columns(df)

        logger.debug("Running {} business-rule checks ...", _TOTAL_BUSINESS_CHECKS)

        # Business-rule checks — collect all violations then raise together.
        issues: list[str] = []
        issues += self._check_numeric_types(df)
        issues += self._check_age_range(df)
        issues += self._check_categorical(df, "gender",       ALLOWED_GENDER)
        issues += self._check_categorical(df, "visit_type",   ALLOWED_VISIT_TYPE)
        issues += self._check_categorical(df, "is_emergency", ALLOWED_IS_EMERGENCY)
        issues += self._check_categorical(df, "payer_type",   ALLOWED_PAYER_TYPE)
        issues += self._check_categorical(df, "claim_status", ALLOWED_CLAIM_STATUS)
        issues += self._check_categorical(df, "department",   ALLOWED_DEPARTMENTS)
        issues += self._check_charge_positive(df)
        issues += self._check_payment_non_negative(df)
        issues += self._check_payment_not_exceeds_charge(df)
        issues += self._check_appointment_id_format(df)
        issues += self._check_inpatient_dates(df)
        issues += self._check_discharge_after_admission(df)

        if issues:
            for msg in issues:
                logger.warning("Validation issue: {}", msg)
            logger.error(
                "Validation FAILED | {}/{} business checks raised issues",
                len(issues), _TOTAL_BUSINESS_CHECKS,
            )
            bullet_list = "\n".join(f"  • {msg}" for msg in issues)
            raise ValueError(
                f"Data validation failed — {len(issues)} issue(s) found:\n{bullet_list}"
            )

        logger.success(
            "Validation passed | {} structural + {} business-rule checks — 0 issues",
            2, _TOTAL_BUSINESS_CHECKS,
        )

    # ── Structural checks ─────────────────────────────────────────────────────

    def _check_not_empty(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("Loaded dataframe is empty.")

    def _check_columns(self, df: pd.DataFrame) -> None:
        missing = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing expected columns: {sorted(missing)}")

    # ── Type / range checks ───────────────────────────────────────────────────

    def _check_numeric_types(self, df: pd.DataFrame) -> list[str]:
        issues = []
        for col in NUMERIC_COLUMNS:
            if col not in df.columns:
                continue
            non_numeric = pd.to_numeric(df[col], errors="coerce").isna() & df[col].notna()
            count = int(non_numeric.sum())
            if count:
                issues.append(f"'{col}': {count} non-numeric value(s) found.")
        return issues

    def _check_age_range(self, df: pd.DataFrame) -> list[str]:
        if "age" not in df.columns:
            return []
        ages = pd.to_numeric(df["age"], errors="coerce")
        out_of_range = df[(ages < AGE_MIN) | (ages > AGE_MAX)]
        if out_of_range.empty:
            return []
        return [
            f"'age': {len(out_of_range)} row(s) outside valid range "
            f"[{AGE_MIN}, {AGE_MAX}]. "
            f"Row indices: {out_of_range.index.tolist()[:10]}"
        ]

    # ── Categorical checks ────────────────────────────────────────────────────

    def _check_categorical(
        self, df: pd.DataFrame, col: str, allowed: set[str]
    ) -> list[str]:
        if col not in df.columns:
            return []
        non_null = df[col].dropna()
        bad_mask   = ~non_null.isin(allowed)
        bad_values = non_null[bad_mask].unique().tolist()
        if not bad_values:
            return []
        return [
            f"'{col}': unexpected value(s) {bad_values} — "
            f"allowed: {sorted(allowed)}"
        ]

    # ── Amount checks ─────────────────────────────────────────────────────────

    def _check_charge_positive(self, df: pd.DataFrame) -> list[str]:
        if "charge_amount_USD" not in df.columns:
            return []
        charges = pd.to_numeric(df["charge_amount_USD"], errors="coerce")
        bad = df[charges <= CHARGE_MIN]
        if bad.empty:
            return []
        return [
            f"'charge_amount_USD': {len(bad)} row(s) with value <= 0. "
            f"Row indices: {bad.index.tolist()[:10]}"
        ]

    def _check_payment_non_negative(self, df: pd.DataFrame) -> list[str]:
        if "payment_amount_USD" not in df.columns:
            return []
        payments = pd.to_numeric(df["payment_amount_USD"], errors="coerce")
        bad = df[payments < PAYMENT_MIN]
        if bad.empty:
            return []
        return [
            f"'payment_amount_USD': {len(bad)} row(s) with negative value. "
            f"Row indices: {bad.index.tolist()[:10]}"
        ]

    def _check_payment_not_exceeds_charge(self, df: pd.DataFrame) -> list[str]:
        if not {"charge_amount_USD", "payment_amount_USD"}.issubset(df.columns):
            return []
        charges  = pd.to_numeric(df["charge_amount_USD"],  errors="coerce")
        payments = pd.to_numeric(df["payment_amount_USD"], errors="coerce")
        bad = df[payments > charges]
        if bad.empty:
            return []
        return [
            f"'payment_amount_USD' > 'charge_amount_USD': {len(bad)} row(s). "
            f"Row indices: {bad.index.tolist()[:10]}"
        ]

    # ── Format check ──────────────────────────────────────────────────────────

    def _check_appointment_id_format(self, df: pd.DataFrame) -> list[str]:
        if "appointment_id" not in df.columns:
            return []
        pattern = re.compile(APPOINTMENT_ID_PATTERN)
        non_null = df["appointment_id"].dropna().astype(str)
        bad = non_null[~non_null.str.match(pattern)]
        if bad.empty:
            return []
        return [
            f"'appointment_id': {len(bad)} row(s) don't match format 'A#####'. "
            f"Row indices: {bad.index.tolist()[:10]}"
        ]

    # ── Cross-column logical checks ───────────────────────────────────────────

    def _check_inpatient_dates(self, df: pd.DataFrame) -> list[str]:
        """Inpatient visits must have both admission_date and discharge_date."""
        required = {"visit_type", "admission_date", "discharge_date"}
        if not required.issubset(df.columns):
            return []
        inpatient         = df[df["visit_type"] == "Inpatient"]
        missing_admission = inpatient["admission_date"].isna() | (inpatient["admission_date"] == "")
        missing_discharge = inpatient["discharge_date"].isna() | (inpatient["discharge_date"] == "")
        issues = []
        if missing_admission.any():
            idx = inpatient[missing_admission].index.tolist()[:10]
            issues.append(
                f"'admission_date': {missing_admission.sum()} Inpatient row(s) missing admission date. "
                f"Row indices: {idx}"
            )
        if missing_discharge.any():
            idx = inpatient[missing_discharge].index.tolist()[:10]
            issues.append(
                f"'discharge_date': {missing_discharge.sum()} Inpatient row(s) missing discharge date. "
                f"Row indices: {idx}"
            )
        return issues

    def _check_discharge_after_admission(self, df: pd.DataFrame) -> list[str]:
        """discharge_date must be >= admission_date when both are present."""
        required = {"admission_date", "discharge_date"}
        if not required.issubset(df.columns):
            return []
        admit     = pd.to_datetime(df["admission_date"], format="%d-%m-%Y", errors="coerce")
        discharge = pd.to_datetime(df["discharge_date"], format="%d-%m-%Y", errors="coerce")
        both_present = admit.notna() & discharge.notna()
        bad = df[both_present & (discharge < admit)]
        if bad.empty:
            return []
        return [
            f"'discharge_date' < 'admission_date': {len(bad)} row(s) where patient "
            f"was discharged before admitted. Row indices: {bad.index.tolist()[:10]}"
        ]
