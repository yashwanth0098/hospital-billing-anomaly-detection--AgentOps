import json
import os
from datetime import datetime

from loguru import logger

_ARTIFACT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "artifacts")
)
REGISTRY_PATH = os.path.join(_ARTIFACT_DIR, "model_registry.json")


class ModelRegistry:
    """
    File-based model registry for the anomaly detection pipeline.

    Each call to register() appends a versioned record to
    artifacts/model_registry.json containing:
      - version       : timestamp-based ID, e.g. "v20260613_184814"
      - trained_at    : ISO-8601 datetime string
      - metrics       : tier distribution, thresholds, track-agreement rates
      - notes         : optional free-text annotation
      - artifact_dir  : path to the artifact folder for this version

    Typical use — register immediately after training + evaluation:
        registry = ModelRegistry()
        registry.register(metrics=eval_report, notes="initial training run")
    """

    def __init__(self):
        self._records: list[dict] = self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def register(self, metrics: dict, notes: str = "") -> str:
        """
        Register a newly trained model with its evaluation metrics.

        Args:
            metrics: Any dict — typically the evaluation_report.json content
                     (tier distribution, thresholds, track agreement rates).
            notes:   Optional annotation, e.g. "retrain after drift on charge_amount".

        Returns:
            Version string, e.g. "v20260613_184814".
        """
        version = "v" + datetime.now().strftime("%Y%m%d_%H%M%S")
        record  = {
            "version"     : version,
            "trained_at"  : datetime.now().isoformat(),
            "metrics"     : metrics,
            "notes"       : notes,
            "artifact_dir": _ARTIFACT_DIR,
        }
        self._records.append(record)
        self._save()
        logger.info(
            "ModelRegistry | registered version={} | notes={}",
            version, notes or "—",
        )
        return version

    def get_latest(self) -> dict | None:
        """Return the most recently registered record, or None if registry is empty."""
        return self._records[-1] if self._records else None

    def get_version(self, version: str) -> dict | None:
        """Return a specific version record by its version string."""
        for record in self._records:
            if record["version"] == version:
                return record
        return None

    def list_versions(self) -> list[dict]:
        """Return all registered versions in chronological order."""
        return list(self._records)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load(self) -> list[dict]:
        if not os.path.exists(REGISTRY_PATH):
            return []
        with open(REGISTRY_PATH, encoding="utf-8") as f:
            return json.load(f)

    def _save(self) -> None:
        os.makedirs(_ARTIFACT_DIR, exist_ok=True)
        with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
            json.dump(self._records, f, indent=2)
        logger.debug(
            "ModelRegistry saved | path={} | n_versions={}",
            REGISTRY_PATH, len(self._records),
        )
