import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, recall_score

def train_model(df):
    X = df.drop("OK_NG", axis=1)
    y = df["OK_NG"]
    weights = compute_sample_weight(class_weight={0: 1.0, 1: 2.0}, y=y)
    model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                          objective="binary:logistic", eval_metric="logloss", random_state=42)
    model.fit(X, y, sample_weight=weights)
    explainer = shap.Explainer(model, X)
    return model, explainer, list(X.columns), X, y

def recommend_threshold(model, X, y):
    y_proba = model.predict_proba(X)[:, 1]
    thresholds = np.linspace(0.01, 0.99, 100)
    best_thresh, best_f1 = 0.5, 0
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        f1 = f1_score(y, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh

def plot_performance(model, X, y):
    y_proba = model.predict_proba(X)[:, 1]
    thresholds = np.linspace(0.01, 0.99, 100)
    recall_0 = []
    recall_1 = []
    f1_scores = []

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        recall_0.append(recall_score(y, preds, pos_label=0))
        recall_1.append(recall_score(y, preds, pos_label=1))
        f1_scores.append(f1_score(y, preds))

    best_t = recommend_threshold(model, X, y)

    fig, ax = plt.subplots()
    ax.plot(thresholds, recall_0, 'r-', label="Recall of NG")
    ax.plot(thresholds, recall_1, 'g-', label="Recall of OK")
    ax.plot(thresholds, f1_scores, 'b-', label="F1 Score")
    ax.axvline(x=best_t, color='purple', linestyle='--', label=f"Recommended Threshold: {best_t:.2f}")
    ax.legend()
    ax.set_title("XGBoost Model Performance")
    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Metric Value")
    ax.grid(True)
    return fig
