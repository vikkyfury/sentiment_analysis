# src/train.py
from pathlib import Path
import logging
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn

# ---------- Paths & tracking ----------
ROOT_DIR    = Path(__file__).resolve().parents[1]
DATA_PATH   = ROOT_DIR / "data" / "processed" / "cleaned_data.csv"
MLRUNS_DIR  = ROOT_DIR / "mlruns"         # experiment history (ignored in Git)
DEPLOY_DIR  = ROOT_DIR / "models" / "latest"  # single deployable copy
EXPERIMENT  = "sentiment_analysis"

MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
logging.getLogger("mlflow").setLevel(logging.ERROR)  # quiet deprecation chatter

# ---------- Helpers ----------
def load_data(path: Path):
    df = pd.read_csv(path)
    return df["clean_comment"], df["category"]

# ---------- Main ----------
def main():
    mlflow.set_experiment(EXPERIMENT)

    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Single pipeline (has .predict)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("clf",   LogisticRegression(C=1.0, max_iter=100)),
    ])

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # Train
        pipe.fit(X_train, y_train)

        # Evaluate
        preds = pipe.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        f1    = f1_score(y_test, preds, average="macro")

        # Log params & metrics
        mlflow.log_params({"vectorizer": "tfidf-5000", "model": "LogisticRegression",
                           "C": 1.0, "max_iter": 100})
        mlflow.log_metrics({"accuracy": acc, "f1_macro": f1})

        # Log to MLflow artifacts (history)
        input_example = ["Great movie!", "Worst plot ever"]
        signature = infer_signature(input_example, pipe.predict(input_example))
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            input_example=input_example,
            signature=signature,
        )

        # Save ONE deployable copy at repo-root/models/latest
        if DEPLOY_DIR.exists():
            shutil.rmtree(DEPLOY_DIR)
        DEPLOY_DIR.parent.mkdir(parents=True, exist_ok=True)
        mlflow.sklearn.save_model(sk_model=pipe, path=str(DEPLOY_DIR))

        print(f"[run={run_id}] accuracy={acc:.4f}, f1_macro={f1:.4f}")
        print(f"MLflow artifacts: {MLRUNS_DIR}")
        print(f"Deployable model: {DEPLOY_DIR}")

if __name__ == "__main__":
    main()
