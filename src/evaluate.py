# src/evaluate.py
import json
import argparse
from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

def evaluate(model_dir: Path, data_csv: Path, out_path: Path):
    # Load data
    df = pd.read_csv(data_csv)
    X, y = df["clean_comment"], df["category"]
    # Same split recipe as train (for consistency)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Load the deployable model saved by train.py
    model = mlflow.sklearn.load_model(str(model_dir))

    # Predict & metrics
    preds = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1_macro": float(f1_score(y_test, preds, average="macro")),
        "n_test": int(len(X_test)),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote metrics -> {out_path}: {metrics}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="metrics.json")
    args = p.parse_args()

    evaluate(Path(args.model_dir), Path(args.data), Path(args.out))
