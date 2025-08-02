# src/serve.py (Flask)
import os
from pathlib import Path
from typing import List
from flask import Flask, jsonify, request
import mlflow.sklearn

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models" / "latest"

model = mlflow.sklearn.load_model(str(MODEL_DIR))
app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "sentiment-api",
        "endpoints": {
            "health": "/health",
            "predict": "/predict"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    texts: List[str] = payload.get("texts")
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        return jsonify({"error": "Request body must be JSON with key 'texts': List[str]."}), 400
    preds = model.predict(texts)
    return jsonify({"predictions": [int(p) for p in preds]})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
