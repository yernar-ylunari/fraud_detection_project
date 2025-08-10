from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

def build_pipeline(preprocessor, model_name="rf"):
    if model_name == "rf":
        clf = RandomForestClassifier(n_jobs=-1, random_state=42)
    elif model_name == "lgbm":
        clf = LGBMClassifier(n_jobs=-1, random_state=42)
    elif model_name == "xgb":
        clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_jobs=-1, random_state=42)
    else:
        raise ValueError("Unknown model_name")

    pipe = Pipeline(steps=[("preprocessor", preprocessor),
                           ("classifier", clf)])
    return pipe
