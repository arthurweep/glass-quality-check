from flask import Flask, render_template, request
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import io
import base64
from model import train_model, recommend_threshold

app = Flask(__name__)
model = None
explainer = None
features = []
y_true = None

@app.route("/", methods=["GET", "POST"])
def index():
    global model, explainer, features, y_true
    message, pred_result, prob, shap_img = "", "", None, None

    if request.method == "POST":
        if "csv_file" in request.files:
            csv_file = request.files["csv_file"]
            df = pd.read_csv(csv_file)
            if "OK_NG" not in df.columns:
                message = "❌ 错误：CSV 文件中必须包含 'OK_NG' 列"
            else:
                model, explainer, features = train_model(df)
                y_true = df["OK_NG"]
                message = "✅ 模型训练完成，可在下方输入参数进行预测。"
        elif model is not None and explainer is not None:
            input_data = [float(request.form[f]) for f in features]
            df_input = pd.DataFrame([input_data], columns=features)
            prob = model.predict_proba(df_input)[0][1]
            threshold = recommend_threshold(model, explainer.data, y_true)
            pred_result = "✅ 合格" if prob >= threshold else "❌ 不合格"

            # 生成 SHAP 图像并转为 base64 编码
            shap_values = explainer(df_input)
            plt.figure()
            shap.plots.waterfall(shap_values[0], show=False)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            shap_img = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()

    return render_template("index.html",
                           message=message,
                           features=features,
                           result=pred_result,
                           prob=prob,
                           shap_img=shap_img)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
