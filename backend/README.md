# Tastewise — Backend
**Team 18 · CIS 509 · Spring 2026**

---

## How everything runs — the honest picture

Training and serving are completely separate.
You run `train.py` once (from the `backend/` directory).
The backend (`uvicorn`) never trains anything — it only loads what `train.py` saved.

```
scripts/train.py (training)           Backend (serving)
────────────────────────────          ─────────────────
1. Filter + join Yelp JSON      ──►   loads saved files,
2. VADER scoring                       serves API
3. TF-IDF + LR fit              ──►
4. BERTopic fit                        app/data/rev_rest.parquet
5. Build per-restaurant profiles ──►  app/models/tfidf_lr/
                                        app/models/bertopic/
                                        app/models/topic_labels.json
                                        app/models/restaurant_profiles.json
                                        app/models/distilbert_sentiment/  ← from Colab
```

---

## What files you need from Canvas

Download `yelp_dataset_new.zip` from the course module on Canvas.
Extract it. You only need two files:

```
Team18TasteWise/
└── data/
    ├── yelp_academic_dataset_business.json   ← put here
    └── yelp_academic_dataset_review.json     ← put here
```

The other three files (user, tip, checkin) are not used.

---

## First-time setup — step by step

### 1. Create virtual environment
```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Put Yelp data in place
```bash
# From project root (Team18TasteWise/)
mkdir data
# copy your two JSON files into data/
```

### 3. Configure environment
```bash
# Copy and edit the env file
cp .env .env.bak   # if you already have one
# Create backend/.env with at least:
# OPENAI_API_KEY=sk-...         ← required for LLM agent features
# SENTIMENT_MODEL_PATH=...      ← set after Colab DistilBERT training
# AGENT_ENABLED=true
```

### 4. Run train.py  ← THIS IS WHERE TRAINING HAPPENS
```bash
# From backend/
python scripts/train.py
```

This takes about 3–5 minutes and does everything:
- Filters and joins the Yelp JSON files (AZ restaurants only)
- Scores every review with VADER
- Trains TF-IDF + Logistic Regression on star-based weak labels
- Fits BERTopic (8 topics, paraphrase-MiniLM-L3-v2 embeddings)
- Builds per-restaurant profiles (sentiment + topics + snippets)
- Saves everything to `backend/app/data/` and `backend/app/models/`

When it finishes you will see a message like:
```
*** IMPORTANT — open app/models/topic_labels.json and rename the 'label'
    fields to human-readable names based on the keywords shown above. ***
```

Open `app/models/topic_labels.json`. You will see entries like:
```json
"0": {
  "id": 0,
  "keywords": ["food", "great", "chicken", "order", "sauce"],
  "label": "Food",     ← rename to something like "Food Quality"
  "count": 312
}
```
Read the keywords, rename each label (Service & Staff, Value & Price,
Wait Time, Ambience & Vibe, etc.), save the file. You can also use the helper:
```bash
python scripts/rename_topics.py
```

### 5. Add DistilBERT (when your Colab training finishes)
After running the Colab fine-tuning notebook, download the saved model
folder from Google Drive and place it at:
```
backend/app/models/distilbert_sentiment/
  config.json
  tokenizer_config.json
  vocab.txt
  model.safetensors
```
Set `SENTIMENT_MODEL_PATH=app/models/distilbert_sentiment` in `backend/.env`.
The backend auto-detects it on the next restart and upgrades from
TF-IDF+LR to DistilBERT. No code changes needed.

### 6. Start the backend
```bash
# From backend/
uvicorn app.main:app --reload --port 8000
```

Check it works:
```bash
curl http://localhost:8000/health
```

Expected output (TF-IDF+LR tier):
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
| 2 | TF-IDF + Logistic Regression | After `python scripts/train.py` | train.py |
| 3 | Fine-tuned DistilBERT | After Colab + model saved + path set in .env | Colab notebook |

The backend loads the best available tier automatically at startup.

---

## Match score formula

```
score = (0.50 × topic_sim + 0.25 × lex_sim + 0.15 × sentiment_pos + 0.10 × star_norm) × 100
```

- **topic_sim** — BERTopic query distribution vs. restaurant profile (or keyword fallback)
- **lex_sim** — query token overlap against restaurant name + categories
- **sentiment_pos** — fraction of positive reviews for that restaurant
- **star_norm** — star rating normalized to [0, 1]

Tune weights in `app/core/topics.py → _match_score_v2`.

---

## Project structure

```
backend/
├── scripts/
│   ├── train.py              ← RUN THIS FIRST (one time only)
│   ├── rename_topics.py      helper to rename topic labels interactively
│   ├── verify_artifacts.py   check that all model files exist
│   ├── verify_data.py        check Yelp JSON files
│   └── verify_schema.py      check parquet schema
├── app/
│   ├── main.py               FastAPI app + CORS + router registration
│   ├── core/
│   │   ├── config.py         pydantic-settings (.env support)
│   │   ├── data.py           loads app/data/rev_rest.parquet (LRU cached)
│   │   ├── sentiment.py      3-tier sentiment engine (VADER → TF-IDF+LR → DistilBERT)
│   │   ├── topics.py         TopicEngine: rank_by_query, load BERTopic + profiles
│   │   └── agent.py          OpenAI tool-calling: parse_query, narrate, summarize
│   ├── models/
│   │   ├── schemas.py        Pydantic request/response schemas
│   │   ├── tfidf_lr/         tfidf.pkl + lr_clf.pkl  ← written by train.py
│   │   ├── bertopic/         saved BERTopic model    ← written by train.py
│   │   ├── topic_labels.json human-editable topic names ← written by train.py
│   │   ├── restaurant_profiles.json per-restaurant aggregated data ← written by train.py
│   │   └── distilbert_sentiment/  ← add manually from Colab
│   ├── data/
│   │   └── rev_rest.parquet  ← written by train.py
│   └── routers/
│       ├── health.py         GET /health
│       ├── recommend.py      POST /recommend
│       ├── restaurants.py    GET /restaurants/{id}, GET /restaurants/
│       └── topics.py         GET /topics
└── requirements.txt
```

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Model + data status |
| `POST` | `/recommend` | Main recommendation endpoint |
| `GET` | `/restaurants/{id}` | Single restaurant detail |
| `GET` | `/restaurants/` | Paginated restaurant list (optional `?city=`) |
| `GET` | `/topics` | List of BERTopic cluster labels |

### POST /recommend

**Request:**
```json
{ "query": "quiet date night good wine", "top_n": 6 }
```

**Response includes:**
```json
{
  "query": "...",
  "model_used": "tfidf_lr",
  "results": [...],
  "total_results": 6,
  "intent_summary": "Finding a quiet romantic spot with good wine",
  "agent_used": true,
  "agent_response": "3-5 sentence LLM summary grounded in review snippets",
  "agent_citations": [{ "restaurant": "...", "snippet": "..." }]
}
```

---

## Running again after train.py

Re-run with shortcuts to skip slow steps:

```bash
# Reuse cached parquet (skip JSON parsing)
python scripts/train.py --use-cache-rev-rest

# Skip re-fitting BERTopic (fastest full re-run)
python scripts/train.py --skip-bertopic

# Only rebuild restaurant profiles (requires existing parquet + BERTopic)
python scripts/train.py --profiles-only

# Tune batch sizes for your RAM
python scripts/train.py --topic-batch-size 30000 --lr-batch-size 60000

# Use more docs when fitting BERTopic (better topics, slower)
python scripts/train.py --n-topic-docs 2500
```

---

## Environment variables (backend/.env)

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | `""` | Required for LLM agent (parse, narrate, summarize) |
| `AGENT_MODEL` | `"gpt-4o-mini"` | OpenAI model for agent calls |
| `AGENT_ENABLED` | `true` | Set `false` to use keyword fallback only |
| `SENTIMENT_MODEL_PATH` | `""` | Path to DistilBERT model dir (relative to `backend/app/`) |
| `FILTER_STATE` | `"AZ"` | Yelp state filter |
| `TOP_N` | `10` | Max results per recommend call |
| `ALLOWED_ORIGINS` | `["http://localhost:3000"]` | CORS allowed origins |
