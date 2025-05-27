from flask import Flask, render_template, request
import pandas as pd
import shap
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import numpy as np

app = Flask(__name__)

model = None
explainer = None
features = []
y_true_global = None

def train_model(df):
    global y_true_global
    X = df.drop("OK_NG", axis=1)
    y = df["OK_NG"]
    y_true_global = y
    weights = compute_sample_weight(class_weight={0: 1.0, 1: 2.0}, y=y)
    model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                          objective="binary:logistic", eval_metric="logloss", random_state=42)
    model.fit(X, y, sample_weight=weights)
    explainer = shap.Explainer(model, X)
    return model, explainer, list(X.columns)

def recommend_threshold(model, X, y_true, target_recall=0.95):
    y_prob = model.predict_proba(X)[:, 1]
    thresholds = np.linspace(0.0, 1.0, 1000)
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        true_positive = ((y_pred == 1) & (y_true == 1)).sum()
        actual_positive = (y_true == 1).sum()
        recall = true_positive / actual_positive if actual_positive > 0 else 0
        if recall >= target_recall:
            return threshold
    return 0.5  # fallback

@app.route("/", methods=["GET", "POST"])
def index():
    global model, explainer, features, y_true_global
    message = ""
    pred_result = ""
    prob = None
    shap_plot = None

    if request.method == "POST":
        if "csv_file" in request.files:
            csv_file = request.files["csv_file"]
            df = pd.read_csv(csv_file)
            if "OK_NG" not in df.columns:
                message = "❌ CSV 文件中缺少 'OK_NG' 列。"
            else:
                model, explainer, features = train_model(df)
                message = "✅ 模型训练完成，可在下方输入参数进行预测。"
        elif model is not None:
            try:
                input_data = [float(request.form[f]) for f in features]
                df_input = pd.DataFrame([input_data], columns=features)
                prob = model.predict_proba(df_input)[0][1]
                threshold = recommend_threshold(model, explainer.data, y_true_global)
                pred_result = "✅ 合格" if prob >= threshold else "❌ 不合格"

                if pred_result == "❌ 不合格":
                    shap_values = explainer(df_input)
                    shap.plots.waterfall(shap_values[0], show=False)
                    buf = BytesIO()
                    plt.savefig(buf, format="png", bbox_inches="tight")
                    plt.close()
                    shap_plot = base64.b64encode(buf.getvalue()).decode("utf-8")

            except Exception as e:
                message = f"❌ 预测失败：{str(e)}"

    return render_template("index.html", message=message, features=features, result=pred_result, prob=prob, shap_plot=shap_plot)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
