import joblib
import pandas as pd

def load_model(path):
    return joblib.load(path)

def predict_from_csv(model, csv_path):
    df = pd.read_csv(csv_path)
    preds = model.predict_proba(df)[:,1]
    df["fraud_prob"] = preds
    return df
