from __future__ import annotations

import json
from pathlib import Path

TOPIC_LABELS = {
    0: "Misc / General",
    1: "Mexican & Tacos",
    2: "Tucson Mentions",
    3: "Pizza",
    4: "Beer & Bar",
    5: "Chicken",
    6: "Overall Food Praise",
    7: "Burgers & Fries",
    8: "Gelato",
    9: "Wings",
    10: "Boba Tea",
    11: "Ice Cream",
    12: "Ramen",
    13: "Sushi",
    14: "Complaints / Not Good",
}

ROOT = Path(__file__).resolve().parents[1]  # project root
TOPIC_PATH = ROOT / "backend" / "app" / "models" / "topic_labels.json"
PROFILES_PATH = ROOT / "backend" / "app" / "models" / "restaurant_profiles.json"


def main() -> None:
    data = json.loads(TOPIC_PATH.read_text(encoding="utf-8"))

    # Update labels in topic_labels.json
    for tid, new_label in TOPIC_LABELS.items():
        k = str(tid)
        if k not in data:
            print(f"skip: topic {tid} not found in {TOPIC_PATH.name}")
            continue
        data[k]["label"] = new_label

    TOPIC_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"updated: {TOPIC_PATH}")

    # OPTIONAL: keep restaurant_profiles.json consistent (your profiles use label strings as keys)
    if PROFILES_PATH.exists():
        profiles = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))

        # Build old->new label mapping using existing keywords[0].title() behavior
        old_to_new = {}
        for tid, new_label in TOPIC_LABELS.items():
            k = str(tid)
            if k in data:
                # We don’t reliably know the *old* label after we overwrote it,
                # so we infer old label from your original JSON pattern when possible.
                # If you want perfect renames, run this script BEFORE overwriting labels,
                # or just regenerate profiles after renaming.
                pass

        # Best practice here: regenerate profiles instead of trying to rewrite keys.
        print("note: regenerate restaurant_profiles.json after renaming to keep keys consistent.")

    else:
        print(f"skip: {PROFILES_PATH} not found")

if __name__ == "__main__":
    main()