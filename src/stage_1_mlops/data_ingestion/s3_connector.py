import os
import pandas as pd


class LocalFileLoader:
    """Loads CSV or Excel files from a local directory (input_excel folder)."""

    SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

    def __init__(self, input_dir: str):
        self.input_dir = input_dir

    def load(self, filename: str | None = None) -> pd.DataFrame:
        """Load the first supported file in input_dir, or a specific filename."""
        path = self._resolve_path(filename)
        ext = os.path.splitext(path)[1].lower()

        if ext == ".csv":
            return pd.read_csv(path)
        elif ext in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _resolve_path(self, filename: str | None) -> str:
        if filename:
            return os.path.join(self.input_dir, filename)

        files = [
            f for f in os.listdir(self.input_dir)
            if os.path.splitext(f)[1].lower() in self.SUPPORTED_EXTENSIONS
        ]
        if not files:
            raise FileNotFoundError(
                f"No supported data file found in: {self.input_dir}"
            )
        return os.path.join(self.input_dir, sorted(files)[0])
