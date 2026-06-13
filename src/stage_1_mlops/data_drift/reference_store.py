"""Saves and loads reference distribution statistics as a JSON file."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


class ReferenceStore:
    """Persists reference statistics needed for JSD and PSI drift tests."""

    _MAX_SAMPLE = 2000  # cap stored numeric samples
    _BINS = 10          # bins used for both PSI and numeric JSD

    def save(
        self,
        df: pd.DataFrame,
        numeric_cols: list[str],
        categorical_cols: list[str],
        path: str | Path,
    ) -> None:
        """Extract and persist reference statistics from the baseline dataset."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        ref: dict = {"row_count": len(df), "numeric": {}, "categorical": {}}

        for col in numeric_cols:
            if col not in df.columns:
                continue
            values = df[col].dropna().astype(float).values
            if len(values) > self._MAX_SAMPLE:
                rng = np.random.default_rng(42)
                values = rng.choice(values, size=self._MAX_SAMPLE, replace=False)
            # percentile-based bin edges guarantee equal-frequency bins
            percentiles = np.linspace(0, 100, self._BINS + 1)
            bin_edges = np.percentile(values, percentiles).tolist()
            ref["numeric"][col] = {
                "sample": values.tolist(),
                "bin_edges": bin_edges,
            }

        for col in categorical_cols:
            if col not in df.columns:
                continue
            # store absolute counts (not proportions) so JSD / PSI can scale correctly
            counts = df[col].value_counts(normalize=False, dropna=True)
            ref["categorical"][col] = counts.to_dict()

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(ref, fh, indent=2)

    def load(self, path: str | Path) -> dict:
        """Load previously saved reference statistics."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Reference distribution not found at: {path}\n"
                "Call DataDriftDetector.fit() on the training data first."
            )
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
