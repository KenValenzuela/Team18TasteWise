# backend/app/core/topics.py
"""
topics.py — Load pre-trained BERTopic model and restaurant profiles

Everything here is produced by train.py. The backend never fits a model.

Refactor goals:
- Make recommendations track the query harder:
  - Prefer BERTopic(query)->topic distribution when available
  - Add lexical match against restaurant name + categories
  - Reweight scoring toward relevance (topic_sim + lex_sim)
- Be defensive against older/partial profile schemas (no KeyError crashes)
- Expose model_loaded property (so /health never touches private internals)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Paths resolved relative to this file — works regardless of run directory
_THIS = Path(__file__).resolve()
_APP_ROOT = _THIS.parent.parent  # backend/app/
_MODELS = _APP_ROOT / "models"

PROFILES_PATH = _MODELS / "restaurant_profiles.json"
BERTOPIC_PATH = _MODELS / "bertopic" / "model"
TOPIC_LABELS_PATH = _MODELS / "topic_labels.json"

_WORD_RE = re.compile(r"[a-z0-9]+")

_STOP = {
    "a", "an", "the", "and", "or", "but", "with", "without", "for", "to", "of",
    "in", "on", "at", "near", "good", "best", "great", "nice", "place", "spots",
    "spot", "restaurant", "food"
}


@dataclass
class RankedRestaurant:
    business_id: str
    name: str
    city: str
    categories: str
    stars_business: float
    match_score: float
    sentiment_positive: float
    sentiment_negative: float
    sentiment_neutral: float
    topic_profile: dict[str, float]
    top_snippets: list[dict[str, Any]]
    n_reviews: int
    top_topic: str = ""


class TopicEngine:
    """
    Loads all artifacts saved by train.py and serves recommendations.
    No training happens here.
    """

    def __init__(self) -> None:
        self._profiles: dict[str, dict[str, Any]] = {}
        self._topic_labels: Any = {}
        self._bertopic = None

        self._check_artifacts()
        self._load()

    # ---------------- Public surface ----------------

    @property
    def model_loaded(self) -> bool:
        """True when BERTopic model was successfully loaded."""
        return bool(self._bertopic)

    def rank_by_query(
        self,
        query: str,
        top_n: int = 6,
        topic_weight_overrides: dict[str, float] | None = None,
        sentiment_threshold: float = 0.0,
    ) -> list[RankedRestaurant]:
        q = (query or "").strip()
        top_n = int(top_n) if top_n is not None else 6
        if top_n <= 0:
            return []

        # Normalize overrides to positive floats (if provided)
        weights: dict[str, float] | None = None
        if topic_weight_overrides:
            weights = {}
            for k, v in topic_weight_overrides.items():
                try:
                    fv = float(v)
                except Exception:
                    continue
                if fv <= 0:
                    continue
                weights[str(k)] = fv
            if not weights:
                weights = None

        # If no overrides, try BERTopic-derived query weights once
        bert_query_weights: dict[str, float] | None = None
        if not weights:
            bert_query_weights = self._query_topic_weights_from_bertopic(q, k=3, min_p=0.05)

        results: list[RankedRestaurant] = []

        for biz_id, raw_profile in self._profiles.items():
            profile = self._normalize_profile(biz_id, raw_profile)

            pos = float(profile["sentiment_positive"])
            if pos < float(sentiment_threshold or 0.0):
                continue

            topic_scores = profile["topic_scores"]

            # Topic similarity priority:
            # 1) Agent overrides
            # 2) BERTopic query distribution
            # 3) Keyword fallback
            if weights:
                topic_sim = self._weighted_sim(topic_scores, weights)
            elif bert_query_weights:
                topic_sim = self._weighted_sim(topic_scores, bert_query_weights)
            else:
                topic_sim = self._keyword_sim(q, topic_scores)

            # Lexical similarity against name/categories (hard constraint booster)
            lex_sim = self._lex_sim(q, str(profile.get("name", "")), str(profile.get("categories", "")))

            match = self._match_score_v2(
                sentiment_pos=pos,
                topic_sim=topic_sim,
                lex_sim=lex_sim,
                star_avg=float(profile.get("stars_business", 3.5) or 3.5),
            )

            results.append(
                RankedRestaurant(
                    business_id=biz_id,
                    name=str(profile.get("name", "")),
                    city=str(profile.get("city", "")),
                    categories=str(profile.get("categories", "")),
                    stars_business=float(profile.get("stars_business", 0.0) or 0.0),
                    match_score=round(float(match), 1),
                    sentiment_positive=round(float(pos), 3),
                    sentiment_negative=round(float(profile["sentiment_negative"]), 3),
                    sentiment_neutral=round(float(profile["sentiment_neutral"]), 3),
                    topic_profile=topic_scores,
                    top_snippets=profile["top_snippets"],
                    n_reviews=int(profile["n_reviews"]),
                    top_topic=str(profile.get("top_topic", "") or ""),
                )
            )

        results.sort(key=lambda r: r.match_score, reverse=True)
        return results[:top_n]

    def get_topic_list(self) -> list[dict[str, Any]]:
        """
        Returns:
          [{"id": 0, "label": "Ambience & Vibe"}, ...]
        Tolerates topic_labels.json being dict or list.
        """
        src = self._topic_labels

        # dict shape: {"0": {"id":0,"label":"..."}, ...} or {"0":"label", ...}
        if isinstance(src, dict):
            out: list[dict[str, Any]] = []
            for k, v in src.items():
                if isinstance(v, dict) and "label" in v:
                    try:
                        tid = int(v.get("id", k))
                    except Exception:
                        continue
                    lbl = str(v.get("label", ""))
                    if lbl:
                        out.append({"id": tid, "label": lbl})
                elif isinstance(v, str):
                    try:
                        tid = int(k)
                    except Exception:
                        continue
                    out.append({"id": tid, "label": v})
            out = [x for x in out if isinstance(x.get("id"), int) and x.get("label")]
            out.sort(key=lambda x: x["id"])
            return out

        # list shape: [{"id":0,"label":"..."}, ...]
        if isinstance(src, list):
            out2: list[dict[str, Any]] = []
            for v in src:
                if isinstance(v, dict) and "label" in v:
                    try:
                        tid = int(v.get("id", -1))
                    except Exception:
                        tid = -1
                    lbl = str(v.get("label", ""))
                    if tid >= 0 and lbl:
                        out2.append({"id": tid, "label": lbl})
            out2.sort(key=lambda x: x["id"])
            return out2

        return []

    def get_restaurant_profile(self, business_id: str) -> dict[str, Any] | None:
        p = self._profiles.get(business_id)
        if not isinstance(p, dict):
            return None
        return self._normalize_profile(business_id, p).copy()

    # ---------------- Internal loading ----------------

    def _check_artifacts(self) -> None:
        missing: list[str] = []
        for p in [PROFILES_PATH, TOPIC_LABELS_PATH]:
            if not p.exists():
                missing.append(str(p))
        if missing:
            raise FileNotFoundError(
                "\n\nMissing training artifacts:\n  - " + "\n  - ".join(missing) + "\n\n"
                "Run your training script to generate these files.\n"
            )

    def _load(self) -> None:
        log.info("Loading restaurant profiles from %s …", PROFILES_PATH)
        with open(PROFILES_PATH, "r", encoding="utf-8") as f:
            profiles_obj = json.load(f)

        # profiles expected: {business_id: {...}}, but tolerate list format too
        if isinstance(profiles_obj, dict):
            self._profiles = {str(k): v for k, v in profiles_obj.items() if isinstance(v, dict)}
        elif isinstance(profiles_obj, list):
            tmp: dict[str, dict[str, Any]] = {}
            for row in profiles_obj:
                if not isinstance(row, dict):
                    continue
                bid = row.get("business_id")
                if not bid:
                    continue
                tmp[str(bid)] = row
            self._profiles = tmp
        else:
            self._profiles = {}

        log.info("Loaded %d restaurant profiles.", len(self._profiles))

        log.info("Loading topic labels from %s …", TOPIC_LABELS_PATH)
        with open(TOPIC_LABELS_PATH, "r", encoding="utf-8") as f:
            self._topic_labels = json.load(f)

        if BERTOPIC_PATH.exists():
            try:
                from bertopic import BERTopic

                self._bertopic = BERTopic.load(str(BERTOPIC_PATH))
                log.info("BERTopic model loaded from %s.", BERTOPIC_PATH)
            except Exception as exc:
                self._bertopic = None
                log.warning("Could not load BERTopic (%s). Continuing without it.", exc)

        # Normalize all profiles once so ranking never crashes
        fixed = 0
        for bid, p in list(self._profiles.items()):
            before = dict(p) if isinstance(p, dict) else None
            self._profiles[bid] = self._normalize_profile(bid, p)
            if before is not None and self._profiles[bid] != before:
                fixed += 1
        if fixed:
            log.warning("Normalized %d profiles (filled missing fields).", fixed)

    def _normalize_profile(self, business_id: str, profile: dict[str, Any]) -> dict[str, Any]:
        """
        Ensures profile has all fields used by rankers/routers, with safe defaults.
        """
        p = profile if isinstance(profile, dict) else {}
        out: dict[str, Any] = dict(p)

        out.setdefault("name", "")
        out.setdefault("city", "")
        out.setdefault("categories", "")
        out.setdefault("stars_business", 0.0)

        # sentiment fields (share in [0..1])
        try:
            out["sentiment_positive"] = float(out.get("sentiment_positive", 0.0) or 0.0)
        except Exception:
            out["sentiment_positive"] = 0.0
        try:
            out["sentiment_negative"] = float(out.get("sentiment_negative", 0.0) or 0.0)
        except Exception:
            out["sentiment_negative"] = 0.0
        try:
            out["sentiment_neutral"] = float(out.get("sentiment_neutral", 0.0) or 0.0)
        except Exception:
            out["sentiment_neutral"] = 0.0

        # topic_scores MUST exist as dict[str, float]
        ts = out.get("topic_scores")
        if not isinstance(ts, dict):
            ts = {}
        clean_ts: dict[str, float] = {}
        for k, v in ts.items():
            try:
                clean_ts[str(k)] = float(v)
            except Exception:
                continue
        out["topic_scores"] = clean_ts

        # snippets
        sn = out.get("top_snippets")
        if not isinstance(sn, list):
            sn = []
        out["top_snippets"] = sn

        # review count
        try:
            out["n_reviews"] = int(out.get("n_reviews", 0) or 0)
        except Exception:
            out["n_reviews"] = 0

        # top_topic (optional)
        if not out.get("top_topic"):
            out["top_topic"] = self._infer_top_topic(out["topic_scores"])

        return out

    def _infer_top_topic(self, topic_scores: dict[str, float]) -> str:
        if not topic_scores:
            return ""
        best = max(topic_scores.items(), key=lambda kv: kv[1])
        return str(best[0])

    # ---------------- Query relevance helpers ----------------

    def _tokenize(self, s: str) -> list[str]:
        toks = _WORD_RE.findall((s or "").lower())
        return [t for t in toks if t not in _STOP and len(t) > 1]

    def _lex_sim(self, query: str, name: str, categories: str) -> float:
        q_toks = self._tokenize(query)
        if not q_toks:
            return 0.0
        hay = f"{name or ''} {categories or ''}".lower()
        uniq = set(q_toks)
        hits = sum(1 for t in uniq if t in hay)
        return float(hits / max(len(uniq), 1))

    def _topic_id_to_label(self, topic_id: int) -> str:
        """
        Convert a BERTopic numeric topic id to your human label from topic_labels.json.
        Works with:
          - {"0": {"id":0,"label":"..."}, ...}
          - {"0": "Ambience & Vibe", ...}
          - [{"id":0,"label":"..."}, ...]
        """
        tid = int(topic_id)
        src = self._topic_labels

        if isinstance(src, dict):
            key = str(tid)
            if key in src:
                v = src[key]
                if isinstance(v, dict) and v.get("label"):
                    return str(v["label"])
                if isinstance(v, str):
                    return v

            for v in src.values():
                if isinstance(v, dict):
                    try:
                        if int(v.get("id", -1)) == tid and v.get("label"):
                            return str(v["label"])
                    except Exception:
                        continue

        if isinstance(src, list):
            for v in src:
                if isinstance(v, dict):
                    try:
                        if int(v.get("id", -1)) == tid and v.get("label"):
                            return str(v["label"])
                    except Exception:
                        continue

        return ""

    def _query_topic_weights_from_bertopic(self, query: str, k: int = 3, min_p: float = 0.05) -> dict[str, float] | None:
        """
        If BERTopic is loaded, derive topic weights for the query and map them to your labels.
        If we can't confidently map probabilities to ids, return None and fall back.
        """
        if not self._bertopic or not (query or "").strip():
            return None

        try:
            topics, probs = self._bertopic.transform([query])
            if probs is None:
                return None

            p = np.asarray(probs[0], dtype=float)
            if p.ndim != 1 or p.size == 0:
                return None

            # Try to map probability indices -> topic ids safely.
            # Strategy:
            # 1) If indices correspond to contiguous topic ids [0..N-1], map directly.
            # 2) Else if size equals number of non-outlier topics sorted by id, use that order.
            model_topics = self._bertopic.get_topics()
            if not isinstance(model_topics, dict):
                return None

            topic_ids = sorted(int(t) for t in model_topics.keys() if int(t) != -1)
            if not topic_ids:
                return None

            idx_to_tid: list[int] = []

            # Case 1: contiguous ids (0..N-1)
            if len(topic_ids) == p.size and topic_ids[0] == 0 and topic_ids[-1] == p.size - 1:
                idx_to_tid = list(range(p.size))
            # Case 2: assume p aligns to sorted topic_ids
            elif len(topic_ids) == p.size:
                idx_to_tid = topic_ids
            else:
                return None

            # Top-k topics by probability
            idxs = np.argsort(p)[::-1][: max(1, int(k))]

            w: dict[str, float] = {}
            for i in idxs:
                pv = float(p[int(i)])
                if pv < float(min_p):
                    continue
                tid = idx_to_tid[int(i)]
                label = self._topic_id_to_label(tid)
                if label:
                    w[label] = pv

            s = sum(w.values())
            if s <= 0:
                return None
            return {kk: vv / s for kk, vv in w.items()}

        except Exception:
            return None

    # ---------------- Similarity + scoring ----------------

    def _weighted_sim(self, topic_scores: dict[str, float], weights: dict[str, float]) -> float:
        dot = 0.0
        wsum = 0.0
        for t, w in weights.items():
            tw = float(w)
            if tw <= 0:
                continue
            dot += float(topic_scores.get(t, 0.0)) * tw
            wsum += tw
        if wsum <= 0:
            return 0.0
        return float(min(dot / wsum, 1.0))

    def _keyword_sim(self, query: str, topic_scores: dict[str, float]) -> float:
        q = (query or "").lower()
        keyword_map: dict[str, list[str]] = {
            "Food Quality": ["food", "flavor", "taste", "authentic", "delicious"],
            "Service & Staff": ["service", "staff", "server", "attentive", "friendly"],
            "Value & Price": ["value", "price", "affordable", "cheap", "worth"],
            "Wait Time": ["wait", "slow", "fast", "quick", "speed"],
            "Ambience & Vibe": ["vibe", "atmosphere", "quiet", "cozy", "date", "romantic"],
            "Parking & Location": ["parking", "location", "convenient"],
            "Brunch & Breakfast": ["brunch", "breakfast", "mimosa"],
            "Drinks & Bar": ["drinks", "bar", "cocktail", "beer", "wine"],
        }

        sim = 0.0
        weight = 0.0
        for topic, kws in keyword_map.items():
            hits = sum(1 for kw in kws if kw in q)
            if hits:
                sim += float(topic_scores.get(topic, 0.0))
                weight += 1.0

        if weight == 0.0:
            vals = list(topic_scores.values())
            return float(np.mean(vals)) if vals else 0.0

        return float(min(sim / weight, 1.0))

    def _match_score_v2(self, sentiment_pos: float, topic_sim: float, lex_sim: float, star_avg: float) -> float:
        """
        Reweighted toward query relevance.
        Tune these weights to taste.
        """
        star_norm = (float(star_avg) - 1.0) / 4.0
        star_norm = max(0.0, min(star_norm, 1.0))

        score = (
            0.50 * float(topic_sim) +
            0.25 * float(lex_sim) +
            0.15 * float(sentiment_pos) +
            0.10 * float(star_norm)
        ) * 100.0

        return float(min(score, 99.0))


@lru_cache(maxsize=1)
def get_topic_engine() -> TopicEngine:
    return TopicEngine()