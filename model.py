
import pandas as pd
import shap
import joblib
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, f1_score
import numpy as np

def train_model(df):
    X = df.drop("OK_NG", axis=1)
    y = df["OK_NG"]
    weights = compute_sample_weight(class_weight={0: 1.0, 1: 2.0}, y=y)
    model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                          objective="binary:logistic", eval_metric="logloss", random_state=42)
    model.fit(X, y, sample_weight=weights)
    explainer = shap.Explainer(model, X)
    return model, explainer, list(X.columns)

def recommend_threshold(model, X, y):
    probs = model.predict_proba(X)[:, 1]
    thresholds = np.arange(0.1, 0.91, 0.01)
    best = 0.5
    best_f1 = 0
    for t in thresholds:
        preds = (probs >= t).astype(int)
        report = classification_report(y, preds, output_dict=True, zero_division=0)
        if report["1"]["recall"] >= 0.95:
            f1 = f1_score(y, preds)
            if f1 > best_f1:
                best_f1 = f1
                best = t
    return best
