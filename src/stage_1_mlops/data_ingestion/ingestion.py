import os

import pandas as pd
from loguru import logger

from .s3_connector import LocalFileLoader
from .data_validator import DataValidator
from .schema import MODE_IMPUTE_COLUMNS, CONSTANT_IMPUTE_COLUMNS, DATE_COLUMNS

INPUT_EXCEL_DIR = os.path.join(
    os.path.dirname(__file__), "..", "input_excel"
)


class DataIngestionPipeline:
    """Loads raw billing data from input_excel, validates, and cleans it."""

    def __init__(self, input_dir: str = INPUT_EXCEL_DIR):
        self.loader    = LocalFileLoader(input_dir=os.path.normpath(input_dir))
        self.validator = DataValidator()

    def run(self, filename: str | None = None) -> pd.DataFrame:
        logger.info("Data ingestion started | source_dir={}", self.loader.input_dir)

        # ── Load ──────────────────────────────────────────────────────────────
        df = self.loader.load(filename)
        logger.info(
            "Raw data loaded | rows={} cols={} | file={}",
            len(df), len(df.columns),
            filename or "<first file in dir>",
        )

        # ── Validate ──────────────────────────────────────────────────────────
        logger.info("Starting validation ...")
        self.validator.validate(df)          # raises on failure; validator logs detail
        logger.success("Validation passed — data meets all schema and business rules")

        # ── Clean ─────────────────────────────────────────────────────────────
        logger.info("Cleaning data (imputation + date parsing) ...")
        df = self._clean(df)
        logger.success(
            "Data ingestion complete | rows={} cols={} | ready for preprocessing",
            len(df), len(df.columns),
        )

        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for col in MODE_IMPUTE_COLUMNS:
            mode_val = df[col].mode()
            if not mode_val.empty:
                n_filled = int(df[col].isna().sum())
                df[col] = df[col].fillna(mode_val[0])
                if n_filled:
                    logger.debug(
                        "Mode imputation | col={} filled={} with value='{}'",
                        col, n_filled, mode_val[0],
                    )

        for col, fill_value in CONSTANT_IMPUTE_COLUMNS.items():
            n_filled = int(df[col].isna().sum())
            df[col] = df[col].fillna(fill_value)
            if n_filled:
                logger.debug(
                    "Constant imputation | col={} filled={} with value='{}'",
                    col, n_filled, fill_value,
                )

        for col, fmt in DATE_COLUMNS.items():
            df[col] = pd.to_datetime(df[col], format=fmt, errors="coerce")
            logger.debug("Date parsed | col={} format={}", col, fmt)

        return df
