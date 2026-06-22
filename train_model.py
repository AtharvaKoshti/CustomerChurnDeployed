"""
Offline training script. Run once before starting the Flask app.
Saves churn_model.pkl and preprocessor.pkl into ./models/
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

DATA_PATH  = os.path.join("data", "Bank_Churn_Classification_Dataset.csv")
MODEL_DIR  = "models"

NUMERIC_COLS = ["Tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_COLS = ["Gender", "Contract", "PaymentMethod"]
TARGET = "Churn"


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.drop(columns=["Unnamed: 0", "CustomerID"], errors="ignore", inplace=True)
    df.drop_duplicates(inplace=True)

    # Cap monthly charges outliers at 99th percentile
    cap = df["MonthlyCharges"].quantile(0.99)
    df["MonthlyCharges"] = df["MonthlyCharges"].clip(upper=cap)

    # Derived feature: charge trajectory (higher total vs monthly = long-term customer)
    df["ChargeRatio"] = df["TotalCharges"] / (df["Tenure"] + 1)

    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_features = NUMERIC_COLS + ["ChargeRatio"]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
        ],
        remainder="drop",
    )


def evaluate(model, X_test, y_test, label: str):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)

    print(f"\n{'='*40}")
    print(f"  {label}")
    print(f"{'='*40}")
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))
    print(f"ROC-AUC : {auc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    return auc


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_and_clean(DATA_PATH)
    X  = df.drop(columns=[TARGET])
    y  = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()

    # --- Logistic Regression baseline ---
    lr_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    lr_pipeline.fit(X_train, y_train)
    evaluate(lr_pipeline, X_test, y_test, "Logistic Regression")

    # --- XGBoost (primary model) ---
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )),
    ])
    xgb_pipeline.fit(X_train, y_train)
    auc = evaluate(xgb_pipeline, X_test, y_test, "XGBoost")

    # 5-fold CV AUC
    cv_scores = cross_val_score(xgb_pipeline, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
    print(f"\nCross-Val AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    joblib.dump(xgb_pipeline, os.path.join(MODEL_DIR, "churn_model.pkl"))
    print(f"\nModel saved → {MODEL_DIR}/churn_model.pkl")


if __name__ == "__main__":
    main()
