# app.py
import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

st.set_page_config(page_title="Heart Failure Predictor", layout="centered")

PIPE_PATH = Path("artifacts/heart_failure_pipeline.pkl")
META_PATH = Path("artifacts/meta.json")

if not PIPE_PATH.exists():
    st.error("Pipeline not found. Run the notebook to create artifacts/heart_failure_pipeline.pkl")
    st.stop()

pipeline = joblib.load(PIPE_PATH)
meta = json.load(open(META_PATH))
features = meta.get("features", [])

st.title("Heart Failure Prediction")
st.write("Enter patient details or upload a CSV to get predictions.")

# Form for single patient
st.subheader("Single patient input")
user = {}
cols = st.columns(2)
for i, feat in enumerate(features):
    fname = feat.lower()
    if any(k in fname for k in ["sex","anaemia","diabetes","smoking","high_blood_pressure"]):
        user[feat] = cols[i%2].selectbox(feat, [0,1])
    else:
        user[feat] = cols[i%2].number_input(feat, value=0.0)

if st.button("Predict single"):
    df_in = pd.DataFrame([user], columns=features)
    proba = pipeline.predict_proba(df_in)[:,1][0]
    pred = int(pipeline.predict(df_in)[0])
    st.write(f"Predicted probability of DEATH_EVENT = 1: **{proba:.3f}**")
    if pred == 1:
        st.error("⚠️ HIGH RISK predicted.")
    else:
        st.success("✅ LOW RISK predicted.")

st.markdown("---")
st.subheader("Batch prediction (CSV upload)")
uploaded = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded:
    df_up = pd.read_csv(uploaded)
    missing = [c for c in features if c not in df_up.columns]
    if missing:
        st.error("CSV missing columns: " + ", ".join(missing))
    else:
        preds = pipeline.predict_proba(df_up[features])[:,1]
        df_up["pred_proba"] = preds
        df_up["pred"] = pipeline.predict(df_up[features])
        st.dataframe(df_up.head())
        csv = df_up.to_csv(index=False).encode('utf-8')
        st.download_button("Download predictions", csv, file_name="predictions.csv")
