# backend/scripts/verify_data.py
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data" / "rev_rest.parquet"

REQUIRED = [
    "business_id",
    "text",
    "stars_review",
    "name",
    "city",
    "state",
    "stars_business",
    "review_count",
    "categories",
]

def main() -> int:
    df = pd.read_parquet(PARQUET)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        print("FAIL: missing columns:", missing)
        print("columns:", sorted(df.columns.tolist()))
        return 1

    print("OK: rows:", len(df))
    print("OK: businesses:", df["business_id"].nunique())
    print("OK: cities top 10:")
    print(df["city"].value_counts().head(10).to_string())
    return 0

if __name__ == "__main__":
    raise SystemExit(main())