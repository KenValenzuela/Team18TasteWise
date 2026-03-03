"""
Sentiment Engine — Three-tier approach
=======================================

  Tier 1 — VADER (always available, no training needed)
  Tier 2 — TF-IDF + Logistic Regression (saved by train.py)
  Tier 3 — Fine-tuned DistilBERT (saved by Colab notebook)

The engine loads the best available tier at startup.
Lower tiers are always used as fallbacks.

Model precedence (highest wins):
  DistilBERT > TF-IDF+LR > VADER
"""

import logging
import os
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from backend.app.core.config import settings

logger = logging.getLogger(__name__)

_THIS     = Path(__file__).resolve()
_APP_ROOT = _THIS.parent.parent
_MODELS   = _APP_ROOT / "models"
TFIDF_PATH = _MODELS / "tfidf_lr" / "tfidf.pkl"
LR_PATH    = _MODELS / "tfidf_lr" / "lr_clf.pkl"


@dataclass
class SentimentResult:
    label: str        # "positive" | "negative" | "neutral"
    positive: float   # probability 0–1
    negative: float
    neutral: float
    compound: float   # VADER compound or logit diff
    model_used: str   # "vader" | "tfidf_lr" | "distilbert"


class SentimentEngine:

    def __init__(self):
        self._vader = SentimentIntensityAnalyzer()
        self._tfidf = None
        self._lr_clf = None
        self._bert_tokenizer = None
        self._bert_model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model_used = "vader"

        # Load in order — each successful load upgrades the tier
        self._try_load_lr()
        self._try_load_bert()

    # ── Loaders ───────────────────────────────────────────────────

    def _try_load_lr(self):
        """Load TF-IDF + LR saved by train.py (Tier 2)."""
        if TFIDF_PATH.exists() and LR_PATH.exists():
            try:
                with open(TFIDF_PATH, "rb") as f:
                    self._tfidf = pickle.load(f)
                with open(LR_PATH, "rb") as f:
                    self._lr_clf = pickle.load(f)
                self._model_used = "tfidf_lr"
                logger.info("TF-IDF + LR classifier loaded (Tier 2).")
            except Exception as exc:
                logger.warning("Could not load TF-IDF+LR: %s. Staying on VADER.", exc)
        else:
            logger.info(
                "TF-IDF+LR not found at %s — run train.py to enable Tier 2. "
                "Using VADER.",
                TFIDF_PATH,
            )

    def _try_load_bert(self):
        """Load fine-tuned DistilBERT saved by Colab notebook (Tier 3)."""
        model_path = settings.SENTIMENT_MODEL_PATH
        if not os.path.exists(model_path):
            logger.info(
                "DistilBERT not found at '%s'. "
                "Finish Colab training and save the model there to enable Tier 3.",
                model_path,
            )
            return
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            logger.info("Loading fine-tuned DistilBERT from '%s'…", model_path)
            self._bert_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._bert_model = (
                AutoModelForSequenceClassification
                .from_pretrained(model_path)
                .to(self._device)
            )
            self._bert_model.eval()
            self._model_used = "distilbert"
            logger.info("DistilBERT loaded on %s (Tier 3 active).", self._device)
        except Exception as exc:
            logger.error(
                "DistilBERT load failed (%s). Staying on %s.",
                exc, self._model_used,
            )

    # ── Public API ────────────────────────────────────────────────

    def predict(self, text: str) -> SentimentResult:
        if self._bert_model is not None:
            return self._bert_predict([text])[0]
        if self._lr_clf is not None:
            return self._lr_predict([text])[0]
        return self._vader_predict(text)

    def predict_batch(self, texts: list, batch_size: int = 32) -> list:
        if self._bert_model is not None:
            results = []
            for i in range(0, len(texts), batch_size):
                results.extend(self._bert_predict(texts[i: i + batch_size]))
            return results
        if self._lr_clf is not None:
            return self._lr_predict(texts)
        return [self._vader_predict(t) for t in texts]

    @property
    def model_name(self) -> str:
        return self._model_used

    # ── Tier 1: VADER ─────────────────────────────────────────────

    def _vader_predict(self, text: str) -> SentimentResult:
        scores = self._vader.polarity_scores(text)
        c = scores["compound"]
        label = "positive" if c >= 0.05 else ("negative" if c <= -0.05 else "neutral")
        return SentimentResult(
            label=label,
            positive=scores["pos"],
            negative=scores["neg"],
            neutral=scores["neu"],
            compound=c,
            model_used="vader",
        )

    # ── Tier 2: TF-IDF + LR (from train.py) ──────────────────────

    def _lr_predict(self, texts: list) -> list:
        X = self._tfidf.transform([str(t) for t in texts])
        probs = self._lr_clf.predict_proba(X)  # (n, 2) [neg, pos]
        results = []
        for p in probs:
            neg_p, pos_p = float(p[0]), float(p[1])
            label = "positive" if pos_p > neg_p else "negative"
            results.append(SentimentResult(
                label=label,
                positive=pos_p,
                negative=neg_p,
                neutral=0.0,
                compound=float(pos_p - neg_p),
                model_used="tfidf_lr",
            ))
        return results

    # ── Tier 3: Fine-tuned DistilBERT ─────────────────────────────

    def _bert_predict(self, texts: list) -> list:
        enc = self._bert_tokenizer(
            texts, padding=True, truncation=True,
            max_length=128, return_tensors="pt",
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self._bert_model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        results = []
        for p in probs:
            neg_p, pos_p = float(p[0]), float(p[1])
            label = "positive" if pos_p > neg_p else "negative"
            results.append(SentimentResult(
                label=label,
                positive=pos_p,
                negative=neg_p,
                neutral=0.0,
                compound=float(pos_p - neg_p),
                model_used="distilbert",
            ))
        return results


@lru_cache(maxsize=1)
def get_sentiment_engine() -> SentimentEngine:
    return SentimentEngine()
