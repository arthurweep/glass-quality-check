from flask import Flask, render_template, request
import pandas as pd
import shap
import matplotlib.pyplot as plt
import base64
import io
from model import train_model, recommend_threshold, plot_performance

app = Flask(__name__)
model = None
explainer = None
features = []
X_global = None
y_global = None
performance_plot_uri = None

@app.route("/", methods=["GET", "POST"])
def index():
    global model, explainer, features, X_global, y_global, performance_plot_uri
    message, pred_result, prob = "", "", None
    shap_table = None
    shap_plot_uri = None

    if request.method == "POST":
        if "csv_file" in request.files:
            csv_file = request.files["csv_file"]
            df = pd.read_csv(csv_file)
            model, explainer, features, X_global, y_global = train_model(df)

            # 绘制性能图并转为 base64 URI
            fig_perf = plot_performance(model, X_global, y_global)
            buffer = io.BytesIO()
            fig_perf.savefig(buffer, format="png", bbox_inches="tight")
            buffer.seek(0)
            performance_plot_uri = base64.b64encode(buffer.read()).decode("utf-8")
            plt.close(fig_perf)

            message = "✅ 模型训练完成，可在下方输入参数进行预测。"

        elif model is not None:
            input_data = [float(request.form[f]) for f in features]
            df_input = pd.DataFrame([input_data], columns=features)
            prob = model.predict_proba(df_input)[0][1]
            threshold = recommend_threshold(model, X_global, y_global)
            pred_result = "✅ 合格" if prob >= threshold else "❌ 不合格"

            # 计算 SHAP 值
            shap_values = explainer(df_input)
            shap_row = shap_values[0]
            shap_df = pd.DataFrame({
                "Feature": features,
                "SHAP Value": shap_row.values,
                "Impact on Probability": shap_row.values * shap_row.base_values
            })
            shap_df = shap_df.reindex(shap_df["SHAP Value"].abs().sort_values(ascending=False).index)
            shap_table = shap_df.head(10).to_dict(orient="records")

            # 绘制 SHAP waterfall 图
            fig_shap = shap.plots.waterfall(shap_row, show=False)
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", bbox_inches="tight")
            buffer.seek(0)
            shap_plot_uri = base64.b64encode(buffer.read()).decode("utf-8")
            plt.close()

    return render_template("index.html",
                           message=message,
                           features=features,
                           result=pred_result,
                           prob=prob,
                           performance_plot_uri=performance_plot_uri,
                           shap_plot_uri=shap_plot_uri,
                           shap_table=shap_table)
