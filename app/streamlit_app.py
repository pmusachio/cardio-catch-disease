"""Streamlit app for cardiovascular risk scoring."""

from __future__ import annotations

import os

import requests
import streamlit as st


st.set_page_config(page_title="Cardio Catch Disease", layout="centered")
st.title("Cardio Catch Disease")

_default_api = os.environ.get("API_URL", "http://127.0.0.1:8000")
api_url = st.text_input("API URL", f"{_default_api}/predict")

c1, c2 = st.columns(2)
with c1:
    age_years = st.number_input("Age", min_value=18, max_value=100, value=52)
    height = st.number_input("Height cm", min_value=120, max_value=230, value=168)
    weight = st.number_input("Weight kg", min_value=35, max_value=220, value=78)
    gender = st.selectbox("Gender", [1, 2], format_func=lambda value: "Female" if value == 1 else "Male")
    ap_hi = st.number_input("Systolic pressure", min_value=70, max_value=250, value=140)
    ap_lo = st.number_input("Diastolic pressure", min_value=40, max_value=180, value=90)
with c2:
    cholesterol = st.selectbox("Cholesterol", [1, 2, 3])
    gluc = st.selectbox("Glucose", [1, 2, 3])
    smoke = st.checkbox("Smoker")
    alco = st.checkbox("Alcohol intake")
    active = st.checkbox("Physically active", value=True)

record = {
    "age": int(age_years * 365.25),
    "height": height,
    "weight": weight,
    "gender": gender,
    "ap_hi": ap_hi,
    "ap_lo": ap_lo,
    "cholesterol": cholesterol,
    "gluc": gluc,
    "smoke": int(smoke),
    "alco": int(alco),
    "active": int(active),
}

if st.button("Score patient", type="primary"):
    response = requests.post(api_url, json={"records": [record]}, timeout=30)
    response.raise_for_status()
    payload = response.json()
    score = payload.get("score", [None])[0]
    prediction = payload.get("prediction", [None])[0]
    st.metric("Prediction", prediction)
    if score is not None:
        st.metric("Risk score", f"{float(score):.1%}")
    st.json(payload)
