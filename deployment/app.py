from flask import Flask, request, jsonify
from model_loader import load_model, predict

app = Flask(__name__)
model, scaler = load_model()


@app.route("/predict", methods=["POST"])
def predict_flood():
    data = request.json
    try:
        prob = predict(
            model,
            scaler,
            [
                data["rainfall"],
                data["river_level"],
                data["drainage"],
                data["urbanization"],
            ],
        )
        return jsonify({"probability": float(prob), "model": "CNN-LSTM Hybrid"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
