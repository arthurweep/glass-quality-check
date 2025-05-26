import joblib
import shap
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

def train_model(df):
    print("Columns in the dataset:", df.columns)  # 打印列名，确保包含 OK_NG 列
    X = df.drop("OK_NG", axis=1)
    y = df["OK_NG"]
    weights = compute_sample_weight(class_weight={0: 1.0, 1: 2.0}, y=y)
    
    model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                          objective="binary:logistic", eval_metric="logloss", random_state=42)
    model.fit(X, y, sample_weight=weights)
    explainer = shap.Explainer(model, X)
    
    return model, explainer, list(X.columns)

def recommend_threshold(model, explainer_data, y_data):
    # Implement your logic for threshold recommendation based on the trained model and explainer
    # You can also use any criteria or SHAP value thresholds here
    return 0.5  # Example threshold
