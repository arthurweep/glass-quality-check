import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import shap

def train_model(df):
    # 确保 X 是一个 pandas DataFrame，保留列名
    X = df.drop("OK_NG", axis=1)
    y = df["OK_NG"]
    
    # 使用样本权重来处理类别不平衡
    weights = compute_sample_weight(class_weight={0: 1.0, 1: 2.0}, y=y)
    
    # 训练 XGBoost 模型
    model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                          objective="binary:logistic", eval_metric="logloss", random_state=42)
    model.fit(X, y, sample_weight=weights)
    
    # 使用 shap.Explainer 创建 SHAP 解释器，并确保传入的是 pandas DataFrame
    explainer = shap.Explainer(model, X)
    
    # 返回模型、SHAP 解释器和特征列名
    return model, explainer, list(X.columns)

def recommend_threshold(model, explainer_data, y_data):
    # 这里假设推荐阈值的方法需要你根据模型输出计算合适的阈值
    # 例如，你可以根据某个指标（如准确率、召回率）来调整
    # 假设我们简单地返回 0.5 作为阈值，实际应用中可以根据需要调整
    return 0.5
