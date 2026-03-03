# backend/scripts/verify_schema.py
from __future__ import annotations

import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
TOPIC_LABELS = ROOT / "backend" / "app" / "models" / "topic_labels.json"
PROFILES = ROOT / "backend" / "app" / "models" / "restaurant_profiles.json"

NUM_RE = re.compile(r"^\d+$")

def main() -> int:
    labels = json.loads(TOPIC_LABELS.read_text(encoding="utf-8"))
    profiles = json.loads(PROFILES.read_text(encoding="utf-8"))

    # basic label sanity
    for k, v in list(labels.items())[:5]:
        assert "label" in v and "keywords" in v, f"bad topic_labels entry for {k}"

    # spot check 10 businesses
    bad = 0
    for i, (bid, obj) in enumerate(profiles.items()):
        if i >= 10:
            break

        top_id = obj.get("top_topic_id")
        top_label = obj.get("top_topic_label")

        if not top_id:
            print("BAD: missing top_topic_id for", bid)
            bad += 1
            continue

        expected = labels.get(str(top_id), {}).get("label", "")
        if expected and top_label != expected:
            print("BAD: label mismatch for", bid, "| top_id:", top_id, "| got:", top_label, "| expected:", expected)
            bad += 1

        ts = obj.get("topic_scores_by_id", {})
        if not ts:
            print("BAD: missing topic_scores_by_id for", bid)
            bad += 1
            continue

        # keys should be numeric strings
        for kk in list(ts.keys())[:8]:
            if not NUM_RE.match(str(kk)):
                print("BAD: non-numeric topic id key:", kk, "for", bid)
                bad += 1
                break

    if bad == 0:
        print("OK: schema looks consistent")
        return 0

    print("FAIL:", bad, "issues found")
    return 1

if __name__ == "__main__":
    raise SystemExit(main())