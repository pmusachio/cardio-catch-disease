"""Transformation layer. Clinical feature engineering (age in years, BMI, pulse
pressure, hypertension flag) and clipping of data-entry errors live in a custom
first pipeline step so training and serving share the identical transform.
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config

logger = logging.getLogger(__name__)

SOURCE_COLUMNS = ["age", "gender", "height", "weight", "ap_hi", "ap_lo",
                  "cholesterol", "gluc", "smoke", "alco", "active"]
NUMERIC_MODEL = list(config.NUMERIC_FEATURES) + list(config.ORDINAL_FEATURES)
PASSTHROUGH = list(config.BINARY_FEATURES)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ("age", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc",
              "gender", "smoke", "alco", "active"):
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out["age_years"] = (out["age"] / 365.25).round(1)
    out["ap_hi"] = out["ap_hi"].clip(*config.AP_HI_BOUNDS)
    out["ap_lo"] = out["ap_lo"].clip(*config.AP_LO_BOUNDS)
    height_m = out["height"] / 100.0
    out["bmi"] = (out["weight"] / (height_m * height_m)).clip(*config.BMI_BOUNDS)
    out["pulse_pressure"] = (out["ap_hi"] - out["ap_lo"]).clip(lower=0)
    out["high_blood_pressure"] = ((out["ap_hi"] >= 140) | (out["ap_lo"] >= 90)).astype(float)
    out["gender"] = out["gender"].map({1: 0, 2: 1}).fillna(out["gender"]).astype(float)
    return out


class FeaturePrep(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X) -> pd.DataFrame:
        df = engineer(pd.DataFrame(X).copy())
        cols = NUMERIC_MODEL + PASSTHROUGH
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        return df[cols]


def build_column_transformer() -> ColumnTransformer:
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    return ColumnTransformer(
        [("num", numeric_pipe, NUMERIC_MODEL),
         ("bin", SimpleImputer(strategy="most_frequent"), PASSTHROUGH)],
        remainder="drop")


class Preprocessor:
    def __init__(self, processed_path=config.PROCESSED_PATH) -> None:
        self.processed_path = processed_path

    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        if config.TARGET not in df.columns:
            raise ValueError(f"Target '{config.TARGET}' missing")
        y = df[config.TARGET].astype(int)
        X = df[[c for c in SOURCE_COLUMNS if c in df.columns]].copy()
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned = engineer(df).copy()
        cleaned[config.TARGET] = y.values
        cleaned.to_parquet(self.processed_path, index=False)
        logger.info("Processed frame (%d rows) written to %s", len(cleaned), self.processed_path)
        return X, y
