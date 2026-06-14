"""Interactive cardiovascular-risk screening dashboard.

Scores a patient's cardiovascular disease risk and shows the sensitivity/specificity
trade-off at the chosen decision threshold, computed on the versioned sample.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.predict import Predictor  # noqa: E402

D = config.DRACULA
st.set_page_config(page_title="Cardio Risk Screening", layout="wide")
st.markdown(
    f"""<style>
    .stApp {{ background-color: {D['background']}; color: {D['foreground']}; }}
    section[data-testid="stSidebar"] {{ background-color: {D['current_line']}; }}
    h1, h2, h3 {{ color: {D['purple']}; }}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor() -> Predictor:
    return Predictor()


@st.cache_data
def load_sample() -> pd.DataFrame:
    return pd.read_csv(config.SAMPLE_PATH, sep=config.CSV_SEP) if config.SAMPLE_PATH.exists() else pd.DataFrame()


@st.cache_data
def sample_scores() -> np.ndarray:
    df = load_sample()
    return load_predictor().score(df) if not df.empty else np.array([])


def style_axes(ax):
    ax.set_facecolor(D["background"])
    for s in ax.spines.values():
        s.set_color(D["current_line"])
    ax.tick_params(colors=D["foreground"])
    ax.xaxis.label.set_color(D["foreground"])
    ax.yaxis.label.set_color(D["foreground"])
    ax.grid(True, color=D["current_line"], linestyle="--", alpha=0.4)


def tradeoff_chart(scores, y, threshold):
    ths = np.linspace(0.05, 0.95, 50)
    sens, spec = [], []
    pos, neg = (y == 1).sum(), (y == 0).sum()
    for t in ths:
        pred = scores >= t
        sens.append(((pred) & (y == 1)).sum() / pos if pos else 0)
        spec.append(((~pred) & (y == 0)).sum() / neg if neg else 0)
    fig, ax = plt.subplots(figsize=(6, 3.4), facecolor=D["background"])
    ax.plot(ths, sens, color=D["pink"], linewidth=2, label="Sensitivity")
    ax.plot(ths, spec, color=D["cyan"], linewidth=2, label="Specificity")
    ax.axvline(threshold, color=D["green"], linestyle=":", linewidth=1.5)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Rate")
    ax.legend(facecolor=D["current_line"], edgecolor=D["comment"], labelcolor=D["foreground"], fontsize=8)
    style_axes(ax)
    fig.tight_layout()
    return fig


def main():
    try:
        predictor = load_predictor()
    except FileNotFoundError:
        st.error("Model artifact not found. Run the pipeline before launching the app.")
        return

    st.title("Cardio Catch Disease — Cardiovascular Risk Screening")
    st.markdown(
        "Estimates a patient's cardiovascular disease risk from a routine examination to support "
        "triage. This is a decision-support tool, not a diagnosis."
    )

    with st.sidebar:
        st.header("Examination")
        age_years = st.slider("Age (years)", 20, 90, 55)
        gender = st.selectbox("Gender", ["Female", "Male"])
        height = st.slider("Height (cm)", 140, 210, 168)
        weight = st.number_input("Weight (kg)", 40.0, 200.0, 82.0, 1.0)
        ap_hi = st.slider("Systolic pressure (ap_hi)", 90, 220, 140)
        ap_lo = st.slider("Diastolic pressure (ap_lo)", 50, 140, 90)
        cholesterol = st.selectbox("Cholesterol", [1, 2, 3],
                                   format_func=lambda v: {1: "Normal", 2: "Above normal", 3: "Well above"}[v])
        gluc = st.selectbox("Glucose", [1, 2, 3],
                            format_func=lambda v: {1: "Normal", 2: "Above normal", 3: "Well above"}[v])
        smoke = st.selectbox("Smoker", [0, 1], format_func=lambda v: "Yes" if v else "No")
        alco = st.selectbox("Alcohol intake", [0, 1], format_func=lambda v: "Yes" if v else "No")
        active = st.selectbox("Physically active", [1, 0], format_func=lambda v: "Yes" if v else "No")
        run = st.button("Assess risk", type="primary")

    record = {"age": int(age_years * 365.25), "gender": 2 if gender == "Male" else 1, "height": height,
              "weight": weight, "ap_hi": ap_hi, "ap_lo": ap_lo, "cholesterol": cholesterol,
              "gluc": gluc, "smoke": smoke, "alco": alco, "active": active}

    if run:
        score = predictor.score_one(record)
        st.subheader("Risk assessment")
        c = st.columns(3)
        c[0].metric("Estimated risk", f"{score*100:.1f}%")
        c[1].metric("Assessment", "Elevated" if score >= predictor.threshold else "Lower")
        c[2].metric("Population prevalence", f"{predictor.base_rate*100:.0f}%")
        color = D["red"] if score >= predictor.threshold else D["green"]
        st.markdown(f"<span style='color:{color}'>{predictor.decision(score).capitalize()}.</span>",
                    unsafe_allow_html=True)
        st.subheader("Most influential features (model-wide)")
        imp = pd.DataFrame(predictor.top_features(6)).rename(
            columns={"feature": "Feature", "importance": "Permutation importance (ROC AUC drop)"})
        st.dataframe(imp, hide_index=True, width="stretch")

    df = load_sample()
    if not df.empty and config.TARGET in df.columns:
        st.subheader("Sensitivity / specificity trade-off (reference sample)")
        threshold = st.slider("Decision threshold", 0.10, 0.90, 0.50, 0.05)
        scores = sample_scores()
        y = df[config.TARGET].to_numpy()
        pred = scores >= threshold
        pos, neg = (y == 1).sum(), (y == 0).sum()
        left, right = st.columns([2, 1])
        with left:
            st.pyplot(tradeoff_chart(scores, y, threshold))
        with right:
            st.metric("Sensitivity", f"{((pred)&(y==1)).sum()/pos*100:.0f}%")
            st.metric("Specificity", f"{((~pred)&(y==0)).sum()/neg*100:.0f}%")


if __name__ == "__main__":
    main()
