# tests/test_train.py
from pathlib import Path
import pandas as pd
import mlflow

def test_train_writes_models_latest(tmp_path, monkeypatch):
    # Make at least 10 rows so stratified test_size=0.2 yields >= 2 test samples
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv = data_dir / "cleaned_data.csv"

    texts = [
        "good film", "bad film", "average", "excellent", "awful",
        "great acting", "poor script", "loved it", "hated it", "meh"
    ]
    labels = [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]  # both classes have >= 2 samples

    pd.DataFrame({"clean_comment": texts, "category": labels}).to_csv(csv, index=False)

    import src.train as train

    # Point train.py to tmp paths
    monkeypatch.setattr(train, "DATA_PATH", csv)
    mlruns_dir = tmp_path / "mlruns"
    monkeypatch.setattr(train, "MLRUNS_DIR", mlruns_dir)
    train.mlflow.set_tracking_uri(f"file://{mlruns_dir}")

    # Your code uses DEPLOY_DIR (models/latest)
    deploy_dir = tmp_path / "models" / "latest"
    monkeypatch.setattr(train, "DEPLOY_DIR", deploy_dir)

    # Run training
    train.main()

    # Assertions
    assert deploy_dir.is_dir(), "models/latest directory not created"
    assert (deploy_dir / "MLmodel").exists(), "MLmodel file missing in models/latest"
