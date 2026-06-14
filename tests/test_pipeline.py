"""Smoke tests for the data contract, engineering and the serving surface."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.predict import Predictor  # noqa: E402
from src.preprocessing import FeaturePrep, Preprocessor, engineer  # noqa: E402


@pytest.fixture(scope="module")
def sample():
    return pd.read_csv(config.SAMPLE_PATH, sep=config.CSV_SEP)


def test_target_present_and_balanced(sample):
    assert config.TARGET in sample.columns
    assert 0.4 < sample[config.TARGET].mean() < 0.6


def test_id_absent_and_engineering(sample):
    X, y = Preprocessor().run(sample)
    assert "id" not in X.columns
    eng = engineer(sample.head(200))
    for c in config.ENGINEERED_FEATURES:
        assert c in eng.columns
    assert eng["ap_hi"].between(*config.AP_HI_BOUNDS).all()
    assert set(y.unique()) <= {0, 1}


def test_feature_prep_fixed_columns(sample):
    a = FeaturePrep().fit_transform(sample.head(20))
    b = FeaturePrep().transform(sample.head(5))
    assert list(a.columns) == list(b.columns)


def test_predictor_contract():
    pred = Predictor()
    rec = {"age": 20000, "gender": 2, "height": 170, "weight": 90, "ap_hi": 150, "ap_lo": 95,
           "cholesterol": 3, "gluc": 1, "smoke": 1, "alco": 0, "active": 0}
    s = pred.score_one(rec)
    assert 0.0 <= s <= 1.0
    assert "risk" in pred.decision(s)
    assert len(pred.top_features(5)) >= 1


def test_higher_risk_above_lower():
    pred = Predictor()
    risky = {"age": 23000, "gender": 2, "height": 165, "weight": 100, "ap_hi": 180, "ap_lo": 110,
             "cholesterol": 3, "gluc": 3, "smoke": 1, "alco": 1, "active": 0}
    healthy = {"age": 12000, "gender": 1, "height": 175, "weight": 62, "ap_hi": 110, "ap_lo": 70,
               "cholesterol": 1, "gluc": 1, "smoke": 0, "alco": 0, "active": 1}
    assert pred.score_one(risky) > pred.score_one(healthy)
