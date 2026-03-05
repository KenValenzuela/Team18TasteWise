"""
data.py — Load pre-trained artifacts produced by train.py

Paths are resolved relative to THIS file so the backend works
regardless of what directory uvicorn is launched from.

Layout:
  backend/
    app/
      core/
        data.py       ← this file
  data/
    rev_rest.parquet   ← written by train.py (project root)
"""

import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# Resolve relative to this file, not the working directory
_THIS     = Path(__file__).resolve()
_APP_ROOT = _THIS.parent.parent           # backend/app/
_BACKEND  = _APP_ROOT.parent              # backend/
_PROJECT  = _BACKEND.parent               # project root
PARQUET_PATH = _PROJECT / "data" / "rev_rest.parquet"


@lru_cache(maxsize=1)
def load_data() -> pd.DataFrame:
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"\n\nrev_rest.parquet not found at:\n  {PARQUET_PATH}\n\n"
            "Run the training script first:\n\n"
            "    python app/scripts/train.py\n\n"
            "This filters and joins your Yelp JSON files, runs VADER,\n"
            "fits BERTopic, and saves everything the backend needs.\n"
        )

    log.info("Loading rev_rest from %s…", PARQUET_PATH)
    df = pd.read_parquet(PARQUET_PATH)
    log.info(
        "Loaded %d reviews | %d businesses",
        len(df), df["business_id"].nunique(),
    )
    return df
