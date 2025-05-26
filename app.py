from flask import Flask, render_template, request
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from model import train_model, recommend_threshold

app = Flask(__name__)
model = None
explainer = None
features = []

@app.route("/", methods=["GET", "POST"])
def index():
    global model, explainer, features
    message, pred_result, prob = "", "", None

    if request.method == "POST":
        if "csv_file" in request.files:
            csv_file = request.files["csv_file"]
            df = pd.read_csv(csv_file)
            model, explainer, features = train_model(df)
            message = "✅ 模型训练完成，可在下方输入参数进行预测。"
        elif model is not None:
            input_data = [float(request.form[f]) for f in features]
            df_input = pd.DataFrame([input_data], columns=features)
            prob = model.predict_proba(df_input)[0][1]
            threshold = recommend_threshold(model, explainer.data, explainer.data["OK_NG"])
            pred_result = "✅ 合格" if prob >= threshold else "❌ 不合格"
            shap_values = explainer(df_input)
            shap.plots.waterfall(shap_values[0], show=False)
            plt.savefig("static/shap_plot.png", bbox_inches="tight")
            plt.close()

    return render_template("index.html", message=message, features=features, result=pred_result, prob=prob)

# Ensure the application listens on the correct IP address and port
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
