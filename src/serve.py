import os
from pathlib import Path
from flask import Flask, request, jsonify
import mlflow.sklearn


# Locate project root and MLflow runs folder
BASE_DIR   = Path(__file__).parents[1]
# Load vendored artifacts
MODEL_PATH = BASE_DIR / "prod_model" / "model"
VECT_PATH  = BASE_DIR / "prod_model" / "vectorizer"

model      = mlflow.sklearn.load_model(str(MODEL_PATH))
vectorizer = mlflow.sklearn.load_model(str(VECT_PATH))

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    # Vectorize and predict
    vec  = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return jsonify({"prediction": int(pred)})

if __name__ == "__main__":
    # Use PORT env var or default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
