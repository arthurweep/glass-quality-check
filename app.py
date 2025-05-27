from flask import Flask, render_template, request
import pandas as pd
import shap
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

app = Flask(__name__)
model = None
explainer = None
features = []
X_train = None
y_train = None

def train_model(df):
    global X_train, y_train
    X_train = df.drop("OK_NG", axis=1)
    y_train = df["OK_NG"]
    weights = compute_sample_weight(class_weight={0: 1.0, 1: 2.0}, y=y_train)
    model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                          objective="binary:logistic", eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train, sample_weight=weights)
    explainer = shap.Explainer(model, X_train)
    return model, explainer, list(X_train.columns)

def recommend_threshold(model, X, y):
    y_proba = model.predict_proba(X)[:, 1]
    thresholds = sorted(set(y_proba))
    best_threshold = 0.5
    best_f1 = 0
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tp = ((y_pred == 1) & (y == 1)).sum()
        fn = ((y_pred == 0) & (y == 1)).sum()
        fp = ((y_pred == 1) & (y == 0)).sum()
        if tp + fn == 0 or tp + fp == 0:
            continue
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        f1 = 2 * precision * recall / (precision + recall)
        if recall >= 0.95 and f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    return best_threshold

@app.route("/", methods=["GET", "POST"])
def index():
    global model, explainer, features, X_train, y_train
    message = ""
    result = ""
    prob = None
    shap_plot_url = None

    if request.method == "POST":
        if "csv_file" in request.files:
            csv_file = request.files["csv_file"]
            df = pd.read_csv(csv_file)
            if "OK_NG" not in df.columns:
                message = "❌ CSV 中必须包含 OK_NG 标签列"
            else:
                model, explainer, features = train_model(df)
                message = "✅ 模型训练成功，请输入参数进行预测"
        elif model is not None:
            try:
                input_data = [float(request.form[f]) for f in features]
                df_input = pd.DataFrame([input_data], columns=features)
                prob = model.predict_proba(df_input)[0][1]
                threshold = recommend_threshold(model, X_train, y_train)
                result = "✅ 合格" if prob >= threshold else "❌ 不合格"
                if result == "❌ 不合格":
                    shap_values = explainer(df_input)
                    shap.plots.waterfall(shap_values[0], show=False)
                    buf = BytesIO()
                    plt.savefig(buf, format="png", bbox_inches="tight")
                    plt.close()
                    buf.seek(0)
                    shap_plot_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception as e:
                message = f"❌ 预测失败：{e}"

    return render_template("index.html",
                           message=message,
                           features=features,
                           result=result,
                           prob=prob,
                           shap_plot_url=shap_plot_url)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
