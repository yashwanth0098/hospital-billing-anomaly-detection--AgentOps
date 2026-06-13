import json
import os

import pandas as pd
from loguru import logger


# ── Column configuration ──────────────────────────────────────────────────────

CORE_COLUMNS = [
    "department", "diagnosis", "visit_reason",
    "age", "is_emergency", "payment_amount_USD", "charge_amount_USD",
]

OPTIONAL_COLUMNS = ["claim_status", "visit_type", "gender"]

NUMERIC_PASSTHROUGH = ["age", "payment_amount_USD", "charge_amount_USD"]

FREQUENCY_COLUMNS = ["department", "diagnosis", "visit_reason"]

BINARY_MAPPINGS: dict[str, dict[str, int]] = {
    "is_emergency": {"Yes": 1, "No": 0},
    "visit_type":   {"Inpatient": 1, "Outpatient": 0},
    "gender":       {"Male": 1, "Female": 0},
}

# Fixed categories defined from schema — not learned from data — so inference
# columns are always deterministic regardless of which statuses appear in a batch.
OHE_COLUMNS: dict[str, list[str]] = {
    "claim_status": sorted({"Submitted", "Approved", "Denied", "Paid"}),
}


class DataTransformer:
    """
    Encodes selected billing columns for anomaly detection.

    Encoding per column type:
      Numeric    — age, charge_amount_USD, payment_amount_USD → pass-through (no scaling)
      Binary     — is_emergency, visit_type, gender           → 0 / 1
      Frequency  — department, diagnosis, visit_reason        → category proportion
      One-Hot    — claim_status                               → 4 binary columns

    fit()  learns frequency maps from training data and must be called once before
    transform(). Use save() / load() to persist state across training and inference runs.
    """

    def __init__(self, include_optional: bool = True):
        self.include_optional = include_optional
        self._freq_maps: dict[str, dict] = {}
        self._fitted = False

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "DataTransformer":
        logger.info("DataTransformer.fit | rows={}", len(df))

        for col in FREQUENCY_COLUMNS:
            self._freq_maps[col] = df[col].value_counts(normalize=True).to_dict()
            logger.debug(
                "Frequency map learned | col={} | n_categories={}",
                col, len(self._freq_maps[col]),
            )

        self._fitted = True
        logger.success("DataTransformer fitted successfully")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")

        active = self._active_columns()
        self._check_columns(df, active)
        df = df[active].copy()

        parts: list[pd.DataFrame] = []

        # ── Numeric passthrough ───────────────────────────────────────────────
        parts.append(df[NUMERIC_PASSTHROUGH].copy())
        logger.debug("Numeric passthrough | cols={}", NUMERIC_PASSTHROUGH)

        # ── Binary encoding ───────────────────────────────────────────────────
        for col, mapping in BINARY_MAPPINGS.items():
            if col not in active:
                continue
            encoded = df[col].map(mapping)
            self._assert_no_unknowns(encoded, df[col], col, list(mapping.keys()))
            parts.append(encoded.rename(col).to_frame())
            logger.debug("Binary encoded | col={}", col)

        # ── Frequency encoding ────────────────────────────────────────────────
        for col in FREQUENCY_COLUMNS:
            freq_map = self._freq_maps[col]
            encoded = df[col].map(freq_map)
            n_unseen = int(encoded.isna().sum())
            if n_unseen:
                # Unseen categories at inference get the smallest known frequency
                # so the model treats them as rare (anomaly-leaning) rather than
                # silently dropping or zeroing them.
                fallback = min(freq_map.values())
                logger.warning(
                    "Unseen categories in '{}' | count={} | assigned fallback_freq={:.6f}",
                    col, n_unseen, fallback,
                )
                encoded = encoded.fillna(fallback)
            parts.append(encoded.rename(f"{col}_freq").to_frame())
            logger.debug("Frequency encoded | col={} → {}_freq", col, col)

        # ── One-Hot encoding ──────────────────────────────────────────────────
        if self.include_optional:
            for col, categories in OHE_COLUMNS.items():
                if col not in active:
                    continue
                for cat in categories:
                    dummy_name = f"{col}_{cat.lower()}"
                    parts.append((df[col] == cat).astype(int).rename(dummy_name).to_frame())
                logger.debug("OHE encoded | col={} | n_dummies={}", col, len(categories))

        result = pd.concat(parts, axis=1)
        logger.info("Transform complete | output_shape={}", result.shape)
        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "freq_maps": self._freq_maps,
            "include_optional": self.include_optional,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        logger.info("Transformer state saved | path={}", path)

    @classmethod
    def load(cls, path: str) -> "DataTransformer":
        with open(path) as f:
            state = json.load(f)
        obj = cls(include_optional=state["include_optional"])
        obj._freq_maps = state["freq_maps"]
        obj._fitted = True
        logger.info("Transformer state loaded | path={}", path)
        return obj

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _active_columns(self) -> list[str]:
        return CORE_COLUMNS + (OPTIONAL_COLUMNS if self.include_optional else [])

    @staticmethod
    def _check_columns(df: pd.DataFrame, required: list[str]) -> None:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Input DataFrame missing required columns: {missing}")

    @staticmethod
    def _assert_no_unknowns(
        encoded: pd.Series,
        original: pd.Series,
        col: str,
        valid: list[str],
    ) -> None:
        unknowns = original[encoded.isna()].unique().tolist()
        if unknowns:
            raise ValueError(
                f"Unknown values in '{col}': {unknowns}. Expected one of: {valid}"
            )
