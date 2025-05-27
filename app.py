from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import shap
from model import train_model, recommend_threshold
import io
import base64

app = Flask(__name__)
model = None
explainer = None
features = []
X_train = None
y_train = None

@app.route("/", methods=["GET", "POST"])
def index():
    global model, explainer, features, X_train, y_train
    message, pred_result, prob = "", "", None
    shap_plot_base64, perf_plot_base64, shap_table = None, None, None

    if request.method == "POST":
        if "csv_file" in request.files and request.files["csv_file"].filename.endswith(".csv"):
            csv_file = request.files["csv_file"]
            df = pd.read_csv(csv_file)
            model, explainer, features = train_model(df)
            X_train = df.drop("OK_NG", axis=1)
            y_train = df["OK_NG"]
            message = "✅ 模型训练完成，可在下方输入参数进行预测。"

            # 性能图
            import sklearn.metrics as metrics
            y_pred_prob = model.predict_proba(X_train)[:, 1]
            fpr, tpr, _ = metrics.roc_curve(y_train, y_pred_prob)
            auc = metrics.roc_auc_score(y_train, y_pred_prob)
            plt.figure()
            plt.plot(fpr, tpr, label=f"ROC Curve (AUC={auc:.2f})")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("模型性能 ROC 曲线")
            plt.legend()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            perf_plot_base64 = base64.b64encode(buf.read()).decode("utf-8")
            plt.close()

        elif model is not None:
            input_data = [float(request.form[f]) for f in features]
            df_input = pd.DataFrame([input_data], columns=features)
            prob = model.predict_proba(df_input)[0][1]
            threshold = recommend_threshold(model, X_train, y_train)
            pred_result = "✅ 合格" if prob >= threshold else "❌ 不合格"

            # SHAP 值图
            shap_values = explainer(df_input)
            plt.figure()
            shap.plots.waterfall(shap_values[0], show=False)
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            shap_plot_base64 = base64.b64encode(buf.read()).decode("utf-8")
            plt.close()

            # SHAP 表格
            sv = shap_values.values[0]
            shap_table = list(zip(features, sv))

    return render_template(
        "index.html",
        message=message,
        features=features,
        result=pred_result,
        prob=prob,
        shap_plot_base64=shap_plot_base64,
        perf_plot_base64=perf_plot_base64,
        shap_table=shap_table
    )

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
