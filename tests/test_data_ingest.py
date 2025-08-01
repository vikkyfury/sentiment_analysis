# tests/test_data_ingest.py
from pathlib import Path
import pandas as pd
from src.data_ingest import load_and_label

def test_load_and_label_basic(tmp_path: Path):
    csv = tmp_path / "Reddit_Data.csv"
    pd.DataFrame(
        {"clean_comment": ["Nice movie", "Terrible plot"], "category": [1, 0]}
    ).to_csv(csv, index=False)

    out = load_and_label(csv, text_col="clean_comment", label_col="category")
    assert list(out.columns) == ["clean_comment", "category"]
    assert len(out) == 2
    assert set(out["category"].unique()) <= {0, 1}
