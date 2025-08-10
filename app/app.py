import streamlit as st
import pandas as pd
import joblib
from src.config import MODELS_DIR

st.set_page_config(page_title="Fraud Detector", layout="wide")
st.title("Fraud Detection — demo")

uploaded = st.file_uploader("Upload CSV with features", type=["csv"])
model_choice = st.selectbox("Model", ["rf", "lgbm", "xgb"])

if st.button("Load model"):
    model = joblib.load(f"{MODELS_DIR}/{model_choice}_best.pkl")
    st.success("Model loaded")

if uploaded is not None and 'model' in locals():
    df = pd.read_csv(uploaded)
    preds = model.predict_proba(df)[:,1]
    df["fraud_probability"] = preds
    st.dataframe(df.head(50))
    st.download_button("Download predictions CSV", df.to_csv(index=False), file_name="predictions.csv")
else:
    st.info("Загрузите CSV и загрузите модель.")
