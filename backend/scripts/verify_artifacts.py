# backend/scripts/verify_artifacts.py
from __future__ import annotations

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]

FILES = [
    ROOT / "data" / "rev_rest.parquet",
    ROOT / "backend" / "app" / "models" / "topic_labels.json",
    ROOT / "backend" / "app" / "models" / "restaurant_profiles.json",
    ROOT / "backend" / "app" / "models" / "tfidf_lr" / "tfidf.pkl",
    ROOT / "backend" / "app" / "models" / "tfidf_lr" / "lr_clf.pkl",
    ROOT / "backend" / "app" / "models" / "bertopic" / "model",
]

def fmt_mtime(p: Path) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime))

def main() -> int:
    ok = True
    print("ROOT:", ROOT)

    for p in FILES:
        if not p.exists():
            print("MISSING:", p)
            ok = False
            continue
        if p.is_file() and p.stat().st_size == 0:
            print("EMPTY FILE:", p)
            ok = False
            continue

        size = p.stat().st_size if p.is_file() else 0
        print("OK:", p, "| size:", size, "| mtime:", fmt_mtime(p))

    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())