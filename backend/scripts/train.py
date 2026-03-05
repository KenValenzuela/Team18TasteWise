# backend/train.py
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd

from backend.app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Canonical paths
#   Team18Project/
#     data/
#     backend/
#       train.py
#       app/
#         models/
# ──────────────────────────────────────────────────────────────────────────────
THIS_FILE = Path(sys.argv[0]).resolve() if Path(sys.argv[0]).exists() else Path(__file__).resolve()
BACKEND_DIR = THIS_FILE.parent
PROJECT_ROOT = BACKEND_DIR.parent if BACKEND_DIR.name == "backend" else THIS_FILE.parent

DATA_DIR = PROJECT_ROOT / "data"
APP_MODELS_DIR = PROJECT_ROOT / "backend" / "app" / "models"

BUSINESS_JSON = DATA_DIR / "yelp_academic_dataset_business.json"
REVIEW_JSON = DATA_DIR / "yelp_academic_dataset_review.json"

OUT_PARQUET = DATA_DIR / "rev_rest.parquet"

OUT_TFIDF = APP_MODELS_DIR / "tfidf_lr" / "tfidf.pkl"
OUT_LR = APP_MODELS_DIR / "tfidf_lr" / "lr_clf.pkl"
OUT_BERTOPIC_DIR = APP_MODELS_DIR / "bertopic"
OUT_BERTOPIC_MODEL = OUT_BERTOPIC_DIR / "model"
OUT_TOPIC_LABELS = APP_MODELS_DIR / "topic_labels.json"
OUT_PROFILES = APP_MODELS_DIR / "restaurant_profiles.json"
OUT_DISTILBERT_DIR = APP_MODELS_DIR / "distilbert_sentiment"


# ──────────────────────────────────────────────────────────────────────────────
# Filters (keep simple; tweak if you want Phoenix/Scottsdale later)
# ──────────────────────────────────────────────────────────────────────────────
FOOD_DRINK_INCLUDE = {
    "Restaurants",
    "Food",
    "Coffee & Tea",
    "Cafes",
    "Bakeries",
    "Desserts",
    "Ice Cream & Frozen Yogurt",
    "Juice Bars & Smoothies",
    "Donuts",
    "Bagels",
    "Fast Food",
    "Food Trucks",
    "Food Delivery Services",
    "Caterers",
    "Delis",
    "Sandwiches",
    "Pizza",
    "Breakfast & Brunch",
    "Bars",
    "Pubs",
    "Breweries",
    "Wine Bars",
    "Cocktail Bars",
    "Beer Bar",
}
GROCERY_EXCLUDE = {"Grocery", "International Grocery", "Ethnic Grocery", "Ethical Grocery"}

FILTER_STATE = "AZ"
EXCLUDE_MODE = "soft"  # "soft" or "hard"


# ──────────────────────────────────────────────────────────────────────────────
# Options
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class LoadOptions:
    review_chunksize: int = 200_000
    use_cache_rev_rest: bool = False


@dataclass(frozen=True)
class RunOptions:
    skip_bertopic: bool = False
    profiles_only: bool = False
    topic_batch_size: int = 20_000
    lr_batch_size: int = 50_000
    n_topic_docs: int = 2000


# ──────────────────────────────────────────────────────────────────────────────
# JSON format helpers (array vs NDJSON)
# ──────────────────────────────────────────────────────────────────────────────
def _read_head_bytes(path: Path, n: int = 4096) -> bytes:
    with path.open("rb") as f:
        return f.read(n).lstrip()


def _is_json_array_file(path: Path) -> bool:
    return _read_head_bytes(path).startswith(b"[")


def _is_ndjson_file(path: Path) -> bool:
    head = _read_head_bytes(path)
    return head.startswith(b"{") or head.startswith(b'{"')


def read_json_auto(path: Path) -> pd.DataFrame:
    if _is_json_array_file(path):
        return pd.read_json(path)
    if _is_ndjson_file(path):
        return pd.read_json(path, lines=True)
    try:
        return pd.read_json(path, lines=True)
    except Exception:
        return pd.read_json(path)


def iter_reviews_auto(path: Path, *, chunksize: int) -> Iterator[pd.DataFrame]:
    if _is_json_array_file(path):
        yield pd.read_json(path)
        return
    for chunk in pd.read_json(path, lines=True, chunksize=chunksize):
        yield chunk


# ──────────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────────
def parse_categories(cat_str: str) -> set:
    return {c.strip() for c in str(cat_str).split(",") if c.strip()}


def is_grocery_store(catset: set) -> bool:
    if not (catset & GROCERY_EXCLUDE):
        return False
    if EXCLUDE_MODE == "hard":
        return True
    if EXCLUDE_MODE == "soft":
        return "Restaurants" not in catset
    raise ValueError("EXCLUDE_MODE must be 'soft' or 'hard'")


def _batch_slices(n: int, batch_size: int) -> Iterator[slice]:
    i = 0
    while i < n:
        j = min(i + batch_size, n)
        yield slice(i, j)
        i = j


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load + Filter (stream reviews)
# Key fix: rename columns so joins never collide:
#   - review stars   -> stars_review
#   - business stars -> stars_business
# ──────────────────────────────────────────────────────────────────────────────
def load_and_filter(opts: LoadOptions) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if opts.use_cache_rev_rest and OUT_PARQUET.exists():
        log.info("Loading cached rev_rest parquet → %s", OUT_PARQUET)
        rev_rest = pd.read_parquet(OUT_PARQUET)
        biz_cols = ["business_id", "name", "city", "state", "stars_business", "review_count", "categories"]
        biz_rest = rev_rest[biz_cols].drop_duplicates("business_id").copy()
        return biz_rest, rev_rest

    log.info("Loading business JSON…")
    biz = read_json_auto(BUSINESS_JSON)
    biz["categories"] = biz.get("categories", "").fillna("")
    biz["cat_set"] = biz["categories"].apply(parse_categories)

    biz_rest = biz[biz["cat_set"].apply(lambda s: bool(s & FOOD_DRINK_INCLUDE))].copy()
    biz_rest = biz_rest[biz_rest["state"] == FILTER_STATE].copy()
    biz_rest = biz_rest[~biz_rest["cat_set"].apply(is_grocery_store)].copy()

    log.info("Businesses after filter: %d", len(biz_rest))

    # business meta for join (rename stars -> stars_business)
    biz_meta = biz_rest[
        ["business_id", "name", "city", "state", "stars", "review_count", "categories"]
    ].copy()
    biz_meta = biz_meta.rename(columns={"stars": "stars_business"})
    biz_meta["business_id"] = biz_meta["business_id"].astype(str)
    biz_meta = biz_meta.drop_duplicates("business_id").set_index("business_id")

    keep_ids = set(biz_meta.index.tolist())

    log.info("Streaming review JSON in chunks (chunksize=%d)…", opts.review_chunksize)

    out_chunks: List[pd.DataFrame] = []
    total_in = 0
    total_kept = 0

    for chunk in iter_reviews_auto(REVIEW_JSON, chunksize=opts.review_chunksize):
        total_in += len(chunk)

        # prune columns early; rename review stars -> stars_review
        want_cols = ["review_id", "user_id", "business_id", "stars", "useful", "text", "date"]
        present = [c for c in want_cols if c in chunk.columns]
        chunk = chunk[present].copy()

        if "stars" in chunk.columns:
            chunk = chunk.rename(columns={"stars": "stars_review"})

        chunk["business_id"] = chunk["business_id"].astype(str)
        chunk = chunk[chunk["business_id"].isin(keep_ids)]
        if chunk.empty:
            continue

        # join (no overlap now)
        chunk = chunk.join(biz_meta, on="business_id", how="inner")

        if "date" in chunk.columns:
            chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")

        out_chunks.append(chunk)
        total_kept += len(chunk)

        if total_in % (opts.review_chunksize * 5) == 0:
            log.info("... scanned %d reviews, kept %d", total_in, total_kept)

    if not out_chunks:
        raise RuntimeError("No reviews matched filters. Check FILTER_STATE/categories.")

    rev_rest = pd.concat(out_chunks, ignore_index=True)

    # memory-friendly dtypes
    for col in ["business_id", "user_id"]:
        if col in rev_rest.columns:
            rev_rest[col] = rev_rest[col].astype("category")

    log.info(
        "rev_rest: %d reviews | %d businesses | %d unique users",
        len(rev_rest),
        rev_rest["business_id"].nunique(),
        rev_rest["user_id"].nunique() if "user_id" in rev_rest.columns else 0,
    )
    return biz_rest, rev_rest


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 — VADER
# ──────────────────────────────────────────────────────────────────────────────
def run_vader(rev_rest: pd.DataFrame) -> pd.DataFrame:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    log.info("Running VADER on %d reviews…", len(rev_rest))
    analyzer = SentimentIntensityAnalyzer()

    texts = rev_rest["text"].astype(str).tolist()
    scores = np.fromiter(
        (analyzer.polarity_scores(t)["compound"] for t in texts),
        dtype=np.float32,
        count=len(texts),
    )

    out = rev_rest.copy()
    out["vader_compound"] = scores

    labels = np.full(len(scores), "neutral", dtype=object)
    labels[scores >= 0.05] = "positive"
    labels[scores <= -0.05] = "negative"
    out["vader_label"] = labels

    log.info("VADER done — positive rate: %.1f%%", float((out["vader_label"] == "positive").mean()) * 100)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 — TF-IDF + LR
# ──────────────────────────────────────────────────────────────────────────────
def train_tfidf_lr(rev_rest: pd.DataFrame):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split

    if "stars_review" not in rev_rest.columns:
        raise ValueError("Expected 'stars_review' in rev_rest. Did you rename review stars correctly?")

    df = rev_rest[["text", "stars_review"]].copy()
    df["label"] = np.where(
        df["stars_review"] >= 4, 1,
        np.where(df["stars_review"] <= 2, 0, np.nan),
    )
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    log.info(
        "TF-IDF+LR labels — positive: %d | negative: %d",
        int((df["label"] == 1).sum()),
        int((df["label"] == 0).sum()),
    )

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].astype(str),
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    tfidf = TfidfVectorizer(
        stop_words="english",
        min_df=5,
        max_df=0.9,
        ngram_range=(1, 2),
    )

    Xtr = tfidf.fit_transform(X_train)
    Xte = tfidf.transform(X_test)

    clf = LogisticRegression(max_iter=300)
    clf.fit(Xtr, y_train)

    pred = clf.predict(Xte)
    log.info("TF-IDF+LR report:\n%s", classification_report(y_test, pred, digits=3))

    OUT_TFIDF.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TFIDF, "wb") as f:
        pickle.dump(tfidf, f)
    with open(OUT_LR, "wb") as f:
        pickle.dump(clf, f)

    log.info("Saved TF-IDF → %s", OUT_TFIDF)
    log.info("Saved LR    → %s", OUT_LR)
    return tfidf, clf


def load_tfidf_lr():
    if not OUT_TFIDF.exists() or not OUT_LR.exists():
        raise FileNotFoundError("Missing TF-IDF/LR artifacts. Run full training once.")
    with open(OUT_TFIDF, "rb") as f:
        tfidf = pickle.load(f)
    with open(OUT_LR, "rb") as f:
        clf = pickle.load(f)
    return tfidf, clf


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 — BERTopic
# ──────────────────────────────────────────────────────────────────────────────
def train_bertopic(rev_rest: pd.DataFrame, *, n_docs: int):
    from bertopic import BERTopic
    from umap import UMAP
    from sklearn.feature_extraction.text import CountVectorizer

    docs_df = rev_rest[["text"]].copy()
    docs_df["text"] = docs_df["text"].astype(str).str.strip()
    docs_df = docs_df[docs_df["text"].str.len() > 0].copy()

    if len(docs_df) > n_docs:
        docs_df = docs_df.sample(n=n_docs, random_state=42)

    docs = docs_df["text"].tolist()
    log.info("Fitting BERTopic on %d documents…", len(docs))

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        metric="cosine",
        random_state=42,
    )

    # remove stop-words from topic representations
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)

    topic_model = BERTopic(
        umap_model=umap_model,
        embedding_model="paraphrase-MiniLM-L3-v2",
        language="english",
        min_topic_size=15,
        vectorizer_model=vectorizer_model,
        verbose=True,
    )

    topic_model.fit(docs)

    topic_info = topic_model.get_topic_info()
    log.info("BERTopic topics:\n%s", topic_info.to_string())

    OUT_BERTOPIC_DIR.mkdir(parents=True, exist_ok=True)
    topic_model.save(str(OUT_BERTOPIC_MODEL))
    log.info("Saved BERTopic model → %s", OUT_BERTOPIC_MODEL)

    # topic_labels.json (you can manually rename "label" values later)
    topic_labels: Dict[str, dict] = {}
    for _, row in topic_info.iterrows():
        tid = int(row["Topic"])
        if tid == -1:
            continue
        words = topic_model.get_topic(tid)
        keywords = [w for w, _ in words[:5]] if words else []
        topic_labels[str(tid)] = {
            "id": tid,
            "keywords": keywords,
            "label": (keywords[0].replace("_", " ").title() if keywords else f"Topic {tid}"),
            "count": int(row["Count"]),
        }

    APP_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TOPIC_LABELS.write_text(json.dumps(topic_labels, indent=2), encoding="utf-8")

    log.info("Saved topic labels → %s", OUT_TOPIC_LABELS)
    log.info("Rename labels in that file, then run: python backend/train.py --profiles-only")
    return topic_model


def load_bertopic():
    if not OUT_BERTOPIC_MODEL.exists():
        raise FileNotFoundError(f"Missing BERTopic model at {OUT_BERTOPIC_MODEL}")
    from bertopic import BERTopic
    return BERTopic.load(str(OUT_BERTOPIC_MODEL))


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5 — Profiles
# Stable contract:
#   - topic_scores_by_id: { "3": 0.12, "4": 0.09, ... }
#   - top_topic_id: "3"
#   - top_topic_label: "Pizza"
# Also includes "topic_scores_by_label" as a convenience for current UI.
# ──────────────────────────────────────────────────────────────────────────────
def _infer_lr_probs_all(rev_rest: pd.DataFrame, tfidf, lr_clf, *, batch_size: int) -> np.ndarray:
    texts = rev_rest["text"].astype(str).to_numpy()
    n = len(texts)
    out = np.empty(n, dtype=np.float32)

    for s in _batch_slices(n, batch_size):
        X = tfidf.transform(texts[s].tolist())
        out[s] = lr_clf.predict_proba(X)[:, 1].astype(np.float32)

    return out


def _topic_ids_in_order(topic_model) -> List[int]:
    info = topic_model.get_topic_info()
    tids = [int(t) for t in info["Topic"].tolist() if int(t) != -1]
    return tids


def build_profiles(
    rev_rest: pd.DataFrame,
    topic_model,
    tfidf,
    lr_clf,
    *,
    topic_batch_size: int,
    lr_batch_size: int,
) -> None:
    if not OUT_TOPIC_LABELS.exists():
        raise FileNotFoundError(f"Missing {OUT_TOPIC_LABELS}")

    topic_labels: Dict[str, dict] = json.loads(OUT_TOPIC_LABELS.read_text(encoding="utf-8"))
    id_to_label = {int(k): v.get("label", f"Topic {k}") for k, v in topic_labels.items() if k != "-1"}
    known_topic_ids = sorted(id_to_label.keys())

    log.info("Building profiles (batched)…")

    biz_series = rev_rest["business_id"].astype(str)
    biz_ids = biz_series.drop_duplicates().tolist()
    biz_id_to_i = {bid: i for i, bid in enumerate(biz_ids)}
    n_biz = len(biz_ids)

    # LR inference
    log.info("LR inference on all reviews (batched)…")
    lr_probs = _infer_lr_probs_all(rev_rest, tfidf, lr_clf, batch_size=lr_batch_size)

    df = rev_rest.copy()
    df["_lr_pos_prob"] = lr_probs

    # sentiment aggregates
    if "vader_compound" in df.columns:
        comp = df["vader_compound"].astype(np.float32)
        df["_pos"] = (comp >= 0.05).astype(np.float32)
        df["_neg"] = (comp <= -0.05).astype(np.float32)
        df["_neu"] = ((comp > -0.05) & (comp < 0.05)).astype(np.float32)
    else:
        if "stars_review" not in df.columns:
            raise ValueError("Expected stars_review in df")
        s = df["stars_review"]
        df["_pos"] = (s >= 4).astype(np.float32)
        df["_neg"] = (s <= 2).astype(np.float32)
        df["_neu"] = (s == 3).astype(np.float32)

    # topic aggregates
    topic_ids = _topic_ids_in_order(topic_model)
    n_topics = len(topic_ids)

    sum_probs = np.zeros((n_biz, n_topics), dtype=np.float32)
    cnt_docs = np.zeros(n_biz, dtype=np.int32)

    text_series = df["text"].astype(str).str.strip()
    mask = text_series.str.len() > 10
    text_vals = text_series[mask].to_numpy()
    biz_vals = biz_series[mask].to_numpy()
    m = len(text_vals)

    log.info("BERTopic transform (batched)…")
    for s in _batch_slices(m, topic_batch_size):
        texts = text_vals[s].tolist()
        biz_chunk = biz_vals[s]
        biz_idx = np.fromiter((biz_id_to_i[b] for b in biz_chunk), dtype=np.int32, count=len(biz_chunk))

        try:
            _, probs = topic_model.transform(texts)
            probs_arr = None if probs is None else np.asarray(probs)
        except Exception:
            probs_arr = None

        np.add.at(cnt_docs, biz_idx, 1)

        if probs_arr is None or probs_arr.ndim != 2 or probs_arr.shape[0] != len(biz_idx):
            continue

        # best-effort alignment:
        # if probs has same number of columns as topic_ids, assume it aligns to that order
        if probs_arr.shape[1] >= n_topics:
            np.add.at(sum_probs, biz_idx, probs_arr[:, :n_topics].astype(np.float32))

    with np.errstate(divide="ignore", invalid="ignore"):
        avg_probs = sum_probs / np.maximum(cnt_docs[:, None], 1)

    group = df.groupby(biz_series, sort=False)

    mean_pos = group["_pos"].mean()
    mean_neg = group["_neg"].mean()
    mean_neu = group["_neu"].mean()
    mean_lr = group["_lr_pos_prob"].mean()

    meta_cols = ["name", "city", "categories", "stars_business", "review_count"]
    meta_first = group[meta_cols].first()

    sort_col = "useful" if "useful" in df.columns else "stars_review"

    profiles: Dict[str, dict] = {}

    for bid, g in group:
        bid_str = str(bid)
        i = biz_id_to_i.get(bid_str)
        if i is None:
            continue

        # topic scores by id + by label
        topic_scores_by_id: Dict[str, float] = {}
        topic_scores_by_label: Dict[str, float] = {}

        row_probs = avg_probs[i]
        for j, tid in enumerate(topic_ids):
            score = float(np.round(row_probs[j], 4))
            topic_scores_by_id[str(tid)] = score
            topic_scores_by_label[id_to_label.get(int(tid), f"Topic {tid}")] = score

        top_topic_id = ""
        if topic_scores_by_id:
            top_topic_id = max(topic_scores_by_id, key=topic_scores_by_id.get)

        top_topic_label = id_to_label.get(int(top_topic_id), "") if top_topic_id else ""

        # snippets
        snippets: List[dict] = []
        g_sorted = g.sort_values(sort_col, ascending=False).head(20)
        for _, r in g_sorted.iterrows():
            t = str(r.get("text", "")).strip()
            if len(t) < 30:
                continue
            preview = t  # store full review text — no truncation
            star = int(r.get("stars_review", 3))
            snippets.append(
                {
                    "text": preview,
                    "sentiment_label": "positive" if star >= 4 else ("negative" if star <= 2 else "neutral"),
                    "stars": star,
                }
            )
            if len(snippets) >= settings.SNIPPET_MAX_PER_RESTAURANT:
                break

        meta = meta_first.loc[bid]
        profiles[bid_str] = {
            "business_id": bid_str,
            "name": str(meta.get("name", "")),
            "city": str(meta.get("city", "")),
            "categories": str(meta.get("categories", "")),
            "stars_business": float(meta.get("stars_business", 0.0)),
            "n_reviews": int(len(g)),
            "sentiment_positive": float(np.round(mean_pos.loc[bid], 4)),
            "sentiment_negative": float(np.round(mean_neg.loc[bid], 4)),
            "sentiment_neutral": float(np.round(mean_neu.loc[bid], 4)),
            "lr_pos_prob": float(np.round(mean_lr.loc[bid], 4)),
            "topic_scores_by_id": topic_scores_by_id,
            "topic_scores_by_label": topic_scores_by_label,
            "top_topic_id": top_topic_id,
            "top_topic_label": top_topic_label,
            "top_snippets": snippets,
        }

    OUT_PROFILES.parent.mkdir(parents=True, exist_ok=True)
    OUT_PROFILES.write_text(json.dumps(profiles, indent=2), encoding="utf-8")
    log.info("Saved %d profiles → %s", len(profiles), OUT_PROFILES)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Tastewise training pipeline (optimized)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--review-chunksize", type=int, default=200_000)
    parser.add_argument("--use-cache-rev-rest", action="store_true")
    parser.add_argument("--skip-bertopic", action="store_true")
    parser.add_argument("--profiles-only", action="store_true")
    parser.add_argument("--topic-batch-size", type=int, default=20_000)
    parser.add_argument("--lr-batch-size", type=int, default=50_000)
    parser.add_argument("--n-topic-docs", type=int, default=2000)
    parser.add_argument("--train-distilbert", action="store_true",
                        help="Also fine-tune DistilBERT sentiment (Tier 3)")
    parser.add_argument("--distilbert-epochs", type=int, default=3)
    parser.add_argument("--distilbert-max-samples", type=int, default=None,
                        help="Cap labeled reviews for DistilBERT (default: all)")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print(f"[DEBUG] Script location : {THIS_FILE}")
    print(f"[DEBUG] PROJECT_ROOT    : {PROJECT_ROOT}")
    print(f"[DEBUG] DATA_DIR        : {DATA_DIR}")
    print(f"[DEBUG] MODELS_DIR      : {APP_MODELS_DIR}")
    print(f"[DEBUG] BUSINESS_JSON   : {BUSINESS_JSON} (exists={BUSINESS_JSON.exists()})")
    print(f"[DEBUG] REVIEW_JSON     : {REVIEW_JSON} (exists={REVIEW_JSON.exists()})")
    print(f"[DEBUG] OUT_PARQUET     : {OUT_PARQUET}")
    print(f"[DEBUG] OUT_PROFILES    : {OUT_PROFILES}")
    print(f"[DEBUG] OUT_TOPIC_LABELS: {OUT_TOPIC_LABELS}")
    print()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    APP_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_BERTOPIC_DIR.mkdir(parents=True, exist_ok=True)

    load_opts = LoadOptions(
        review_chunksize=args.review_chunksize,
        use_cache_rev_rest=args.use_cache_rev_rest,
    )
    run_opts = RunOptions(
        skip_bertopic=args.skip_bertopic,
        profiles_only=args.profiles_only,
        topic_batch_size=args.topic_batch_size,
        lr_batch_size=args.lr_batch_size,
        n_topic_docs=args.n_topic_docs,
    )

    # Profiles-only path: fastest iteration after renaming topic labels
    if run_opts.profiles_only:
        if not OUT_PARQUET.exists():
            raise FileNotFoundError(f"Missing {OUT_PARQUET}. Run full pipeline once first.")
        rev_rest = pd.read_parquet(OUT_PARQUET)
        tfidf, lr_clf = load_tfidf_lr()
        topic_model = load_bertopic()
        build_profiles(
            rev_rest,
            topic_model,
            tfidf,
            lr_clf,
            topic_batch_size=run_opts.topic_batch_size,
            lr_batch_size=run_opts.lr_batch_size,
        )
        return

    # Full pipeline
    if not BUSINESS_JSON.exists() or not REVIEW_JSON.exists():
        raise FileNotFoundError("Missing Yelp data files under data/ at project root.")

    biz_rest, rev_rest = load_and_filter(load_opts)
    rev_rest = run_vader(rev_rest)

    tfidf, lr_clf = train_tfidf_lr(rev_rest)

    if run_opts.skip_bertopic and OUT_BERTOPIC_MODEL.exists() and OUT_TOPIC_LABELS.exists():
        log.info("Skipping BERTopic fit — loading existing model…")
        topic_model = load_bertopic()
    else:
        topic_model = train_bertopic(rev_rest, n_docs=run_opts.n_topic_docs)

    build_profiles(
        rev_rest,
        topic_model,
        tfidf,
        lr_clf,
        topic_batch_size=run_opts.topic_batch_size,
        lr_batch_size=run_opts.lr_batch_size,
    )

    # DistilBERT fine-tuning (Tier 3 sentiment)
    if args.train_distilbert:
        log.info("── Fine-tuning DistilBERT (Tier 3) ──")
        from train_distilbert import train as train_distilbert
        train_distilbert(
            epochs=args.distilbert_epochs,
            max_samples=args.distilbert_max_samples,
        )
    elif not OUT_DISTILBERT_DIR.exists():
        log.info(
            "Tip: run with --train-distilbert to fine-tune DistilBERT (Tier 3), "
            "or run: python backend/scripts/train_distilbert.py"
        )

    # Save cache last
    rev_rest.to_parquet(OUT_PARQUET, index=False)
    log.info("Saved rev_rest → %s", OUT_PARQUET)


if __name__ == "__main__":
    main()