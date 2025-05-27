from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import shap

def train_model(df):
    X = df.drop("OK_NG", axis=1)
    y = df["OK_NG"]
    weights = compute_sample_weight(class_weight={0: 1.0, 1: 2.0}, y=y)
    model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                          objective="binary:logistic", eval_metric="logloss", random_state=42)
    model.fit(X, y, sample_weight=weights)
    explainer = shap.Explainer(model, X)
    return model, explainer, list(X.columns)

def recommend_threshold(model, explainer_data, y):
    # This is a simplified version, in real cases, you'd compute the threshold based on performance metrics.
    return 0.5
