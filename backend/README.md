# Tastewise — Backend
**Team 18 · CIS 509 · Spring 2026**

---

## How everything runs — the honest picture

Training and serving are completely separate.
You run `train.py` once (in your local environment).
The backend (`uvicorn`) never trains anything — it only loads what `train.py` saved.

```
Notebooks (training)          train.py (glue)         Backend (serving)
─────────────────────         ───────────────         ─────────────────
EDA notebook           ──►    mirrors cells 11,       loads saved files,
 - Filter + join               17, 18 exactly,        serves API
 - VADER scoring               saves artifacts
 - TF-IDF + LR fit   ──►    ──────────────────
 - BERTopic fit                data/rev_rest.parquet
                               models/tfidf_lr/
Colab notebook         ──►    models/bertopic/
 - Fine-tune DistilBERT        models/topic_labels.json
                    ──►        models/restaurant_profiles.json
                               models/distilbert_sentiment/  ← from Colab
```

---

## What files you need from Canvas

Download `yelp_dataset_new.zip` from the course module on Canvas.
Extract it. You only need two files:

```
tastewise-backend/
└── data/
    ├── yelp_academic_dataset_business.json   ← put here
    └── yelp_academic_dataset_review.json     ← put here
```

The other three files (user, tip, checkin) are not used by the backend.

---

## First-time setup — step by step

### 1. Create virtual environment
```bash
cd tastewise-backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Put Yelp data in place
```bash
mkdir data
# copy your two JSON files into data/
```

### 3. Configure environment
```bash
cp .env.example .env
# open .env and fill in ANTHROPIC_API_KEY if you have one
```

### 4. Run train.py  ← THIS IS WHERE TRAINING HAPPENS
```bash
python train.py
```

This takes about 3–5 minutes and does everything:
- Filters and joins the Yelp JSON files (mirrors EDA Cell 11)
- Scores every review with VADER (Cell 17 §7.2)
- Trains TF-IDF + Logistic Regression on star-based weak labels (Cell 17 §7.3)
- Fits BERTopic with the exact same config as Cell 18
- Builds per-restaurant profiles (sentiment + topics + snippets)
- Saves everything to disk

When it finishes you will see a message like:
```
*** IMPORTANT — open models/topic_labels.json and rename the 'label'
    fields to human-readable names based on the keywords shown above. ***
```

Open `models/topic_labels.json`. You will see entries like:
```json
"0": {
  "id": 0,
  "keywords": ["food", "great", "chicken", "order", "sauce"],
  "label": "Food",     <-- rename this to something like "Food Quality"
  "count": 312
}
```
Read the keywords, rename each label to match (Service & Staff, Value & Price,
Wait Time, Ambience & Vibe, etc.), save the file. This is the only manual step.

### 5. Add DistilBERT (when your Colab training finishes)
After running the Colab fine-tuning notebook, download the saved model
folder from Google Drive and place it at:
```
models/distilbert_sentiment/
  config.json
  tokenizer_config.json
  vocab.txt
  model.safetensors
```
The backend auto-detects it on the next restart and upgrades from
TF-IDF+LR to DistilBERT. No code changes needed.

### 6. Start the backend
```bash
uvicorn app.main:app --reload --port 8000
```

Check it works:
```bash
curl http://localhost:8000/health
```

Expected output:
```json
{
  "status": "ok",
  "sentiment_model": "tfidf_lr",
  "topic_model_loaded": true,
  "data_loaded": true,
  "n_restaurants": 251,
  "n_reviews": 3900
}
```

Once you add DistilBERT, `sentiment_model` will say `"distilbert"`.

---

## Sentiment model tiers

| Tier | Model | When active | Source |
|---|---|---|---|
| 1 | VADER | Always (fallback) | No training needed |
| 2 | TF-IDF + Logistic Regression | After `python train.py` | EDA notebook Cell 17 |
| 3 | Fine-tuned DistilBERT | After Colab + model saved | Colab notebook |

The backend loads the best available tier automatically at startup.

---

## Project structure

```
tastewise-backend/
├── train.py                  ← RUN THIS FIRST (one time only)
├── app/
│   ├── main.py
│   ├── core/
│   │   ├── config.py         settings + paths
│   │   ├── data.py           loads data/rev_rest.parquet
│   │   ├── sentiment.py      3-tier sentiment engine
│   │   ├── topics.py         loads saved BERTopic + profiles
│   │   └── agent.py          Anthropic orchestration
│   ├── models/schemas.py
│   └── routers/
│       ├── health.py         GET /health
│       ├── recommend.py      POST /recommend
│       ├── restaurants.py    GET /restaurants
│       └── topics.py         GET /topics
├── data/                     ← put Yelp JSON files here
├── models/                   ← train.py writes here; backend reads here
├── .env.example
└── requirements.txt
```

---

## Running again after train.py

If you only want to re-run sentiment scoring or profiles without
re-fitting BERTopic (which takes the longest):

```bash
python train.py --skip-bertopic
```
