import os
import pandas as pd
from pathlib import Path

# Compute project root, two levels up from this file:
# src/data_ingest.py → src → project root
BASE_DIR      = Path(__file__).parents[1]
RAW_DIR       = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE   = PROCESSED_DIR / "cleaned_data.csv"

def load_and_label(path: Path, text_col: str, label_col: str):
    df = pd.read_csv(path)
    df = df.rename(columns={text_col: "clean_comment"})
    df["category"] = df[label_col].fillna(-1).astype(int)
    return df[["clean_comment", "category"]]

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    reddit = load_and_label(
        RAW_DIR / "Reddit_Data.csv",
        text_col="clean_comment",
        label_col="category"
    )
    twitter = load_and_label(
        RAW_DIR / "Twitter_Data.csv",
        text_col="clean_text",
        label_col="category"
    )

    df = pd.concat([reddit, twitter], ignore_index=True)
    print(f"Combined rows before cleaning: {len(df)}")

    df = df[(df["category"] >= 0) & df["clean_comment"].notna()]
    print(f"After dropping missing: {len(df)}")

    df = df.drop_duplicates(subset=["clean_comment"])
    print(f"After dedupe: {len(df)}")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Cleaned data written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
