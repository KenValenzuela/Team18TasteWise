"""
Fine-tune DistilBERT for binary sentiment classification
=========================================================

Uses the same weak-label scheme as TF-IDF+LR in train.py:
  - stars >= 4 → positive (label 1)
  - stars <= 2 → negative (label 0)
  - stars == 3 → discarded (ambiguous)

Data source (in order of preference):
  1. Cached parquet from train.py  (data/rev_rest.parquet)
  2. Raw Yelp JSON files           (data/yelp_academic_dataset_*.json)

Output:
  backend/app/models/distilbert_sentiment/
    ├── config.json
    ├── tokenizer_config.json
    ├── tokenizer.json
    ├── vocab.txt
    ├── special_tokens_map.json
    └── model.safetensors

Once saved, the backend's SentimentEngine auto-detects it at startup
and upgrades from Tier 2 (TF-IDF+LR) to Tier 3 (DistilBERT).

Usage:
  python backend/scripts/train_distilbert.py                     # defaults
  python backend/scripts/train_distilbert.py --epochs 5          # more epochs
  python backend/scripts/train_distilbert.py --max-samples 20000 # smaller subset
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
THIS_FILE = Path(__file__).resolve()
SCRIPTS_DIR = THIS_FILE.parent
BACKEND_DIR = SCRIPTS_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
PARQUET_PATH = DATA_DIR / "rev_rest.parquet"
BUSINESS_JSON = DATA_DIR / "yelp_academic_dataset_business.json"
REVIEW_JSON = DATA_DIR / "yelp_academic_dataset_review.json"

OUT_DIR = BACKEND_DIR / "app" / "models" / "distilbert_sentiment"

BASE_MODEL = "distilbert-base-uncased"


# ── Dataset ──────────────────────────────────────────────────────────────────
class SentimentDataset(Dataset):
    """Simple PyTorch dataset for tokenized review texts + binary labels."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ── Data loading ─────────────────────────────────────────────────────────────
def load_reviews() -> pd.DataFrame:
    """Load reviews from parquet cache or raw Yelp JSON."""
    if PARQUET_PATH.exists():
        log.info("Loading cached parquet → %s", PARQUET_PATH)
        df = pd.read_parquet(PARQUET_PATH)
        if "stars_review" in df.columns and "text" in df.columns:
            return df[["text", "stars_review"]].copy()
        raise ValueError("Parquet missing expected columns (text, stars_review).")

    # Fall back to raw JSON (reuse logic from train.py)
    if not REVIEW_JSON.exists():
        raise FileNotFoundError(
            f"No data found. Run 'python backend/scripts/train.py' first to create "
            f"{PARQUET_PATH}, or place Yelp JSON files in {DATA_DIR}/"
        )

    log.info("Parquet not found — loading from raw Yelp JSON…")
    # Import helpers from sibling train script
    sys.path.insert(0, str(SCRIPTS_DIR))
    from train import load_and_filter, LoadOptions

    _, rev_rest = load_and_filter(LoadOptions())
    return rev_rest[["text", "stars_review"]].copy()


def prepare_labels(df: pd.DataFrame, max_samples: int | None = None) -> pd.DataFrame:
    """Apply weak-label scheme: stars>=4 → 1, stars<=2 → 0, discard 3-star."""
    df = df.copy()
    df["label"] = np.where(
        df["stars_review"] >= 4, 1,
        np.where(df["stars_review"] <= 2, 0, np.nan),
    )
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 10].copy()

    log.info("Labeled reviews — positive: %d | negative: %d",
             (df["label"] == 1).sum(), (df["label"] == 0).sum())

    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
        log.info("Subsampled to %d reviews", len(df))

    return df


# ── Training ─────────────────────────────────────────────────────────────────
def train(
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    max_length: int = 128,
    max_samples: int | None = None,
    test_size: float = 0.15,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # Load and label data
    raw = load_reviews()
    df = prepare_labels(raw, max_samples=max_samples)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size=test_size,
        random_state=42,
        stratify=df["label"].tolist(),
    )
    log.info("Train: %d | Test: %d", len(X_train), len(X_test))

    # Tokenize
    log.info("Loading tokenizer: %s", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    log.info("Tokenizing…")
    train_enc = tokenizer(X_train, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    test_enc = tokenizer(X_test, truncation=True, padding=True, max_length=max_length, return_tensors="pt")

    train_dataset = SentimentDataset(train_enc, y_train)
    test_dataset = SentimentDataset(test_enc, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Model
    log.info("Loading model: %s (2-class)", BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )
    model.to(device)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    # Training loop
    log.info("Training for %d epochs (%d steps)…", epochs, total_steps)
    model.train()

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0

        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

            if step % 50 == 0:
                avg_loss = running_loss / step
                acc = correct / total
                log.info("  Epoch %d | Step %d/%d | Loss: %.4f | Acc: %.3f",
                         epoch, step, len(train_loader), avg_loss, acc)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        log.info("Epoch %d complete — Loss: %.4f | Train Acc: %.3f", epoch, epoch_loss, epoch_acc)

    # Evaluation
    log.info("Evaluating on test set…")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    report = classification_report(
        all_labels, all_preds,
        target_names=["negative", "positive"],
        digits=3,
    )
    log.info("DistilBERT evaluation:\n%s", report)

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    log.info("Saved fine-tuned DistilBERT → %s", OUT_DIR)
    log.info(
        "Set SENTIMENT_MODEL_PATH in your .env or it will auto-detect from:\n  %s",
        OUT_DIR,
    )


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT for Yelp review sentiment (Tier 3)"
    )
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Training batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate (default: 2e-5)")
    parser.add_argument("--max-length", type=int, default=128,
                        help="Max token length per review (default: 128)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap on total labeled reviews (default: use all)")
    parser.add_argument("--test-size", type=float, default=0.15,
                        help="Fraction held out for evaluation (default: 0.15)")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        max_samples=args.max_samples,
        test_size=args.test_size,
    )


if __name__ == "__main__":
    main()
