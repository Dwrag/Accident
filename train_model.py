import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report,
)

from utils import FEATURE_COLUMNS, SEVERITY_LABELS

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from dl_model import TorchMLPClassifier
    HAS_DL = True
except ImportError:
    HAS_DL = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "historical_accidents.csv")
ARTIFACT_DIR = os.path.join(BASE_DIR, "model_artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python data/generate_historical_data.py` "
            "first, or drop your own historical_accidents.csv there with the same "
            "columns (see utils.FEATURE_COLUMNS + a 'severity' column)."
        )
    return pd.read_csv(DATA_PATH)


def get_candidate_models():
    models = {}

    # Logistic Regression
    models["Logistic Regression"] = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42
    )

    # Decision Tree
    models["Decision Tree"] = DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced"
    )

    # Random Forest
    models["Random Forest"] = RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=3,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    # Gradient Boosting
    models["Gradient Boosting"] = GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.08,
        random_state=42
    )

    # XGBoost
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="mlogloss",
            n_jobs=-1
        )

    # Deep Learning: PyTorch MLP (feed-forward net for tabular data)
    if HAS_DL:
        models["Deep Learning (MLP)"] = TorchMLPClassifier(
            hidden_dims=(128, 64, 32),
            dropout=0.25,
            lr=1e-3,
            weight_decay=1e-4,
            epochs=200,
            batch_size=64,
            patience=15,
            random_state=42,
            verbose=True,
        )

    return models

def evaluate(model, X_test, y_test, scaled_needed, scaler):
    Xt = scaler.transform(X_test) if scaled_needed else X_test
    preds = model.predict(Xt)
    probs = model.predict_proba(Xt) if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1_weighted": f1_score(y_test, preds, average="weighted"),
        "precision_weighted": precision_score(y_test, preds, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, preds, average="weighted"),
    }
    if probs is not None:
        try:
            metrics["roc_auc_ovr"] = roc_auc_score(y_test, probs, multi_class="ovr")
        except Exception:
            metrics["roc_auc_ovr"] = np.nan
    else:
        metrics["roc_auc_ovr"] = np.nan
    return metrics, preds


def main():
    print("Loading historical data ...")
    df = load_data()
    X = df[FEATURE_COLUMNS]
    y = df["severity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    NEEDS_SCALING = {"Logistic Regression", "Deep Learning (MLP)"}  # tree ensembles don't need scaling; neural nets do

    candidates = get_candidate_models()
    leaderboard = []
    trained_models = {}

    print(f"Comparing {len(candidates)} algorithms: {list(candidates.keys())}\n")
    for name, model in candidates.items():
        Xt_train = X_train_scaled if name in NEEDS_SCALING else X_train
        model.fit(Xt_train, y_train)
        trained_models[name] = model

        metrics, preds = evaluate(model, X_test, y_test, name in NEEDS_SCALING, scaler)
        metrics["model"] = name
        leaderboard.append(metrics)
        print(f"  {name:20s} | acc={metrics['accuracy']:.3f}  "
              f"f1={metrics['f1_weighted']:.3f}  "
              f"roc_auc={metrics['roc_auc_ovr']:.3f}")

    board_df = pd.DataFrame(leaderboard).sort_values("f1_weighted", ascending=False)
    board_df = board_df[["model", "accuracy", "f1_weighted", "precision_weighted",
                          "recall_weighted", "roc_auc_ovr"]]
    board_df.to_csv(os.path.join(ARTIFACT_DIR, "model_comparison.csv"), index=False)

    best_name = board_df.iloc[0]["model"]
    best_model = trained_models[best_name]
    print(f"\nWinner (highest weighted F1): {best_name}")
    print(classification_report(
        y_test,
        best_model.predict(scaler.transform(X_test) if best_name in NEEDS_SCALING else X_test),
        target_names=[SEVERITY_LABELS[i] for i in sorted(SEVERITY_LABELS)],
    ))

    joblib.dump(best_model, os.path.join(ARTIFACT_DIR, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.pkl"))

    meta = {
        "best_model_name": best_name,
        "needs_scaling": best_name in NEEDS_SCALING,
        "feature_columns": FEATURE_COLUMNS,
        "trained_on_rows": len(df),
    }
    with open(os.path.join(ARTIFACT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # feature importance if available
    if hasattr(best_model, "feature_importances_"):
        fi = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "importance": best_model.feature_importances_,
        }).sort_values("importance", ascending=False)
        fi.to_csv(os.path.join(ARTIFACT_DIR, "feature_importance.csv"), index=False)

    print(f"\nSaved model artifacts to {ARTIFACT_DIR}/")


if __name__ == "__main__":
    main()