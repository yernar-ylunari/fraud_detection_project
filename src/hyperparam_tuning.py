import optuna
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from src.pipeline_builder import build_pipeline

def objective_lgbm(trial, X, y, preprocessor):
    params = {
        "classifier__n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "classifier__num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "classifier__learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.2),
    }

    pipe = build_pipeline(preprocessor, model_name="lgbm")
    pipe.set_params(**params)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return np.mean(scores)

def run_optuna_study(X, y, preprocessor, n_trials=50):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective_lgbm(trial, X, y, preprocessor), n_trials=n_trials)
    return study
