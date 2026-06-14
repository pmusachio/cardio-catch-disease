"""Serving layer: load the serialized pipeline and expose the screening contract."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, artifact_path: Path = config.PIPELINE_PATH) -> None:
        import joblib

        if not Path(artifact_path).exists():
            raise FileNotFoundError(f"No artifact at {artifact_path}. Run `python -m src.pipeline` first.")
        art = joblib.load(artifact_path)
        self.pipeline = art["pipeline"]
        self.base_rate: float = art["base_rate"]
        self.threshold: float = art.get("threshold", 0.5)
        self.importances: List[Dict[str, Any]] = art.get("importances", [])
        self.best_model: str = art.get("best_model", "")

    def score(self, records: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(records)[:, 1]

    def score_one(self, features: Dict[str, Any]) -> float:
        return float(self.score(pd.DataFrame([features]))[0])

    def decision(self, score: float) -> str:
        return "elevated cardiovascular risk" if score >= self.threshold else "lower cardiovascular risk"

    def top_features(self, n: int = 6) -> List[Dict[str, Any]]:
        return self.importances[:n]
