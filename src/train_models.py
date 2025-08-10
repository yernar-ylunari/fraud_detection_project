import joblib
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from src.config import MODELS_DIR
from src.pipeline_builder import build_pipeline

def grid_search_train(X, y, preprocessor, model_name="rf"):
    pipe = build_pipeline(preprocessor, model_name=model_name)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    if model_name == "rf":
        param_grid = {
            "classifier__n_estimators": [100, 300],
            "classifier__max_depth": [None, 10, 20],
            "classifier__min_samples_split": [2, 5]
        }
    elif model_name == "lgbm":
        param_grid = {
            "classifier__n_estimators": [100, 300],
            "classifier__num_leaves": [31, 64],
            "classifier__learning_rate": [0.05, 0.1]
        }
    elif model_name == "xgb":
        param_grid = {
            "classifier__n_estimators": [100, 300],
            "classifier__max_depth": [3, 6],
            "classifier__learning_rate": [0.05, 0.1]
        }
    else:
        param_grid = {}

    gs = GridSearchCV(pipe, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=2)
    gs.fit(X, y)

    # save model
    fname = f"{MODELS_DIR}/{model_name}_best.pkl"
    joblib.dump(gs.best_estimator_, fname)

    return gs


if __name__ == "__main__":
    import os
    from src.data_preprocessing import load_data, build_preprocessor
    from src.config import RAW_DIR

    # Load demo data
    data_path = os.path.join(RAW_DIR, "sample.csv")
    df = load_data(data_path)

    numeric_features = ["amount", "transaction_hour"]
    categorical_features = ["country", "device_type"]
    X = df[numeric_features + categorical_features]
    y = df["is_fraud"]

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    for model_name in ["rf", "lgbm", "xgb"]:
        print(f"Training {model_name} ...")
        gs = grid_search_train(X, y, preprocessor, model_name=model_name)
        print(model_name, "best ROC-AUC:", gs.best_score_)
