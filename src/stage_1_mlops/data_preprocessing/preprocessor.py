import os

import pandas as pd
from loguru import logger

from .feature_engineering import DataTransformer

_ARTIFACT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "artifacts")
)
TRANSFORMER_STATE_PATH  = os.path.join(_ARTIFACT_DIR, "transformer_state.json")
TRANSFORMED_DATA_PATH   = os.path.join(_ARTIFACT_DIR, "transformed_data.pkl")


class DataPreprocessingPipeline:
    """
    Orchestrates column selection and encoding for the anomaly detection pipeline.

    Training  : fit() learns frequency maps on the full dataset, transforms it,
                saves the fitted state to transformer_state.json, and persists
                the encoded DataFrame to transformed_data.pkl for model training.
    Inference : loads the saved state and applies the same maps to new records,
                guaranteeing identical feature columns at scoring time.
    """

    def __init__(self, include_optional: bool = True):
        self.transformer = DataTransformer(include_optional=include_optional)

    def run_training(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(
            "DataPreprocessingPipeline | mode=training | rows={} cols={}",
            len(df), len(df.columns),
        )
        transformed = self.transformer.fit_transform(df)
        self.transformer.save(TRANSFORMER_STATE_PATH)
        self._save_pkl(transformed, TRANSFORMED_DATA_PATH)
        logger.success(
            "Preprocessing complete | output_shape={} | artifacts saved",
            transformed.shape,
        )
        return transformed

    def run_inference(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(
            "DataPreprocessingPipeline | mode=inference | rows={} cols={}",
            len(df), len(df.columns),
        )
        self.transformer = DataTransformer.load(TRANSFORMER_STATE_PATH)
        transformed = self.transformer.transform(df)
        logger.success(
            "Preprocessing complete | output_shape={}",
            transformed.shape,
        )
        return transformed

    @staticmethod
    def load_transformed_data() -> pd.DataFrame:
        """Load the pickled training data produced by run_training()."""
        if not os.path.exists(TRANSFORMED_DATA_PATH):
            raise FileNotFoundError(
                f"Transformed data not found at '{TRANSFORMED_DATA_PATH}'. "
                "Run run_training() first."
            )
        df = pd.read_pickle(TRANSFORMED_DATA_PATH)
        logger.info("Transformed data loaded | shape={} | path={}", df.shape, TRANSFORMED_DATA_PATH)
        return df

    @staticmethod
    def _save_pkl(df: pd.DataFrame, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_pickle(path)
        logger.info("Transformed data saved | shape={} | path={}", df.shape, path)
