# Tastewise
**Team 18 · CIS 509 · Spring 2026**

Yelp-powered restaurant recommender for the Arizona dataset. Type a natural language query ("quiet date night with good wine") and Tastewise ranks ~250 local restaurants using sentiment analysis and BERTopic topic modeling, then generates a grounded conversational summary via an OpenAI agent.

---

## Repo layout

```
Team18TasteWise/
├── data/              ← create this manually; add Yelp JSON files from Canvas (gitignored)
├── backend/           ← FastAPI + training pipeline (Python)
├── frontend/          ← Next.js UI (TypeScript + Tailwind)
└── test_main.http     ← quick HTTP smoke tests for the API
```

---

## How it works

```
User query (Next.js :3000)
        ↓  POST /api/recommend
FastAPI backend (:8000)
  ├─ OpenAI agent   → parse intent into topic weights
  ├─ TopicEngine    → rank ~250 AZ restaurants by query
  ├─ OpenAI agent   → generate one-sentence reason per card
  └─ OpenAI agent   → grounded conversational summary + citations
        ↓
Cards + agent summary rendered in browser
```

Training and serving are decoupled — `scripts/train.py` is run once to fit models and build restaurant profiles. The API only loads what was saved to disk; it never retrains.

---

## Sentiment model tiers

The backend auto-selects the best available tier at startup:

| Tier | Model | Requires |
|------|-------|---------|
| 1 | VADER (always active, fallback) | nothing |
| 2 | TF-IDF + Logistic Regression | `python scripts/train.py` |
| 3 | Fine-tuned DistilBERT | Colab training + model saved to `backend/app/models/distilbert_sentiment/` |

---

## Prerequisites

- **Python 3.10+** (backend + training)
- **Node.js 18+** (frontend)
- **Yelp dataset** — download `yelp_dataset_new.zip` from Canvas and extract the two JSON files below
- **OpenAI API key** — required for agent features; without it the app falls back to keyword matching and template summaries

---

## Quick start

### 1. Get the data
```bash
mkdir data
# place these two files in data/:
#   yelp_academic_dataset_business.json
#   yelp_academic_dataset_review.json
```

### 2. Backend — install dependencies
```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment
```bash
# Create backend/.env and add at minimum:
# OPENAI_API_KEY=sk-...
```

### 4. Run the training pipeline (one time, ~3-5 min)
```bash
# from backend/
python scripts/train.py
```

This filters and joins the Yelp JSON files, runs VADER, trains TF-IDF + Logistic Regression, fits BERTopic (8 topics), builds per-restaurant profiles, and saves everything to `backend/app/data/` and `backend/app/models/`.

After it finishes, open `backend/app/models/topic_labels.json` and rename the auto-generated topic labels to human-readable names based on the printed keywords (e.g. `"Food Quality"`, `"Ambience & Vibe"`). Use the helper script if you prefer a guided prompt:
```bash
python scripts/rename_topics.py
```

### 5. Start the backend
```bash
# from backend/
uvicorn app.main:app --reload --port 8000
```

Verify:
```bash
curl http://localhost:8000/health
```

### 6. Start the frontend
```bash
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

---

## Sub-READMEs

- **`backend/README.md`** — full backend setup, `train.py` CLI options, all API endpoints, environment variable reference
- **`frontend/README.md`** — Next.js setup, component map, TypeScript API types

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 16, React 18, TypeScript, Tailwind CSS |
| Backend | FastAPI, Uvicorn, Pydantic |
| Sentiment | VADER · TF-IDF + Logistic Regression · DistilBERT (fine-tuned) |
| Topic modeling | BERTopic + UMAP + HDBSCAN + paraphrase-MiniLM-L3-v2 |
| Agent | OpenAI `gpt-4o-mini` (tool-calling) |
| Data | Yelp Academic Dataset — Arizona subset (~250 restaurants, ~3,900 reviews) |
