# src/train.py

import os
import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

# 1) Ensure an `mlruns/` folder exists at your repo root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MLRUNS_DIR = os.path.join(ROOT_DIR, "mlruns")
pathlib.Path(MLRUNS_DIR).mkdir(parents=True, exist_ok=True)

# 2) Point MLflow at that local folder
mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")

# Paths & constants
DATA_PATH  = os.path.join(ROOT_DIR, "data", "processed", "cleaned_data.csv")
EXPERIMENT = "sentiment_analysis"

def load_data(path):
    df = pd.read_csv(path)
    return df["clean_comment"], df["category"]

def main():
    # Create or switch to our experiment
    mlflow.set_experiment(EXPERIMENT)

    # Load & split
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF vectorizer
    vect = TfidfVectorizer(max_features=5000)
    X_train_vec = vect.fit_transform(X_train)
    X_test_vec  = vect.transform(X_test)

    # Hyperparameters
    C        = 1.0
    max_iter = 100

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("vectorizer", "tfidf-5000")
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)

        # Train
        model = LogisticRegression(C=C, max_iter=max_iter)
        model.fit(X_train_vec, y_train)

        # Evaluate
        preds = model.predict(X_test_vec)
        acc   = accuracy_score(y_test, preds)
        f1    = f1_score(y_test, preds, average="macro")

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)

        # Log artifacts (model & vectorizer)
        mlflow.sklearn.log_model(model,       "model")
        mlflow.sklearn.log_model(vect,        "vectorizer")

        print(
            f"[MLflow run={mlflow.active_run().info.run_id}] "
            f"accuracy={acc:.4f}, f1_macro={f1:.4f}"
        )

if __name__ == "__main__":
    main()
