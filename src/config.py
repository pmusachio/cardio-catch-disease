"""Central configuration: paths, dataset identity, modeling constants and the
Dracula palette shared by the pipeline, the serving layer and the dashboard.
"""
from __future__ import annotations

from pathlib import Path

BASE_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = BASE_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
SAMPLE_DIR: Path = DATA_DIR / "sample"
MODELS_DIR: Path = BASE_DIR / "models"

PIPELINE_PATH: Path = MODELS_DIR / "pipeline.joblib"
MODEL_CARD_PATH: Path = MODELS_DIR / "model_card.json"
PROCESSED_PATH: Path = PROCESSED_DIR / "train.parquet"

SAMPLE_FILENAME: str = "cardio_sample.csv"
SAMPLE_PATH: Path = SAMPLE_DIR / SAMPLE_FILENAME

KAGGLE_DATASET: str = "sulianova/cardiovascular-disease-dataset"
RAW_FILENAME: str = "cardio_train.csv"
CSV_SEP: str = ";"

TARGET: str = "cardio"
POSITIVE_LABEL: int = 1
ID_COLS: tuple[str, ...] = ("id",)
# No target leakage: every measurement is taken at the examination, before diagnosis.

# Plausible clinical bounds used to clip data-entry errors in the raw file.
AP_HI_BOUNDS: tuple[int, int] = (60, 250)
AP_LO_BOUNDS: tuple[int, int] = (30, 200)
BMI_BOUNDS: tuple[int, int] = (10, 80)

NUMERIC_FEATURES: tuple[str, ...] = (
    "age_years", "height", "weight", "bmi", "ap_hi", "ap_lo", "pulse_pressure",
)
ORDINAL_FEATURES: tuple[str, ...] = ("cholesterol", "gluc")  # 1=normal, 2=above, 3=well above
BINARY_FEATURES: tuple[str, ...] = ("gender", "smoke", "alco", "active", "high_blood_pressure")
ENGINEERED_FEATURES: tuple[str, ...] = ("age_years", "bmi", "pulse_pressure", "high_blood_pressure")

TEST_SIZE: float = 0.2
SEED: int = 42
CV_FOLDS: int = 5
TUNING_ITERS: int = 18
SCORING: str = "roc_auc"

DRACULA = {
    "background": "#282a36", "current_line": "#44475a", "foreground": "#f8f8f2",
    "comment": "#6272a4", "cyan": "#8be9fd", "green": "#50fa7b", "orange": "#ffb86c",
    "pink": "#ff79c6", "purple": "#bd93f9", "red": "#ff5555", "yellow": "#f1fa8c",
}
