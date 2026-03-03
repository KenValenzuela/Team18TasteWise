# Tastewise — Next.js Frontend
**Team 18 · CIS 509 · Spring 2026**

## Stack
| | |
|---|---|
| Framework | Next.js 16 (App Router) |
| Language | TypeScript |
| Styling | Tailwind CSS |
| Fonts | Cormorant Garamond (display) + Syne (body) |
| API | FastAPI backend via `/api/*` proxy |

---

## Project Structure

```
frontend/
├── app/
│   ├── layout.tsx        # Root layout, metadata ("Tastewise — Restaurant Intelligence")
│   ├── page.tsx          # Main page — orchestrates all state
│   └── globals.css       # Tailwind base + grain texture + bar animations
├── components/
│   ├── SearchStage.tsx   # Full-screen hero → collapses to sticky nav
│   ├── RestaurantCard.tsx # Card in the grid
│   ├── SentimentBars.tsx  # Animated pos/neg/neu bars (IntersectionObserver)
│   ├── SentimentDonut.tsx # SVG donut chart for the drawer
│   ├── DetailDrawer.tsx   # Slide-in detail panel
│   ├── AgentResponse.tsx  # LLM-generated summary with citations
│   └── InfoSection.tsx    # Stats strip + pipeline + bias journal at bottom
├── lib/
│   └── api.ts            # Typed fetch client → FastAPI
├── .env.local            # NEXT_PUBLIC_API_URL (optional; defaults to localhost:8000)
├── next.config.js        # /api/* proxy to FastAPI
└── tailwind.config.ts
```

---

## Setup

### 1 — Install
```bash
npm install
```

### 2 — Start the FastAPI backend first
```bash
# In backend/
uvicorn app.main:app --reload --port 8000
```

### 3 — Start Next.js
```bash
npm run dev
# → http://localhost:3000
```

---

## How It Connects to FastAPI

`next.config.js` proxies every `/api/*` request to `http://localhost:8000/*`
(configurable via `NEXT_PUBLIC_API_URL`):

```
POST /api/recommend  →  POST http://localhost:8000/recommend
GET  /api/health     →  GET  http://localhost:8000/health
GET  /api/topics     →  GET  http://localhost:8000/topics
```

This means **no CORS issues** in development, and in production you just
change `NEXT_PUBLIC_API_URL` to your deployed FastAPI URL.

---

## UX Flow

```
1. Full-screen hero search
      ↓  user types query + hits Enter (min 3 characters)
2. Stage collapses into sticky nav (search bar remains visible)
3. POST /api/recommend called with query + top_n=6
4. AgentResponse renders LLM-generated summary + citations above results
5. Cards stagger in with animated sentiment bars
6. Click any card → slide-in DetailDrawer with:
     - Match score (50% topic sim + 25% lexical + 15% sentiment + 10% stars)
     - Sentiment donut (pos/neg/neu percentages)
     - BERTopic topic bar chart (8 topic clusters)
     - Top review snippets with sentiment labels
     - Model details (VADER / TF-IDF+LR / DistilBERT tier in use)
7. Below cards: stats strip + model cards + pipeline + bias journal
```

---

## API Types (`lib/api.ts`)

```typescript
TopicScore        { label: string; score: number }
ReviewSnippet     { text: string; sentiment_label: "positive"|"negative"|"neutral"; stars: number }
AgentCitation     { restaurant: string; snippet: string }

RestaurantResult  {
  business_id, name, city, categories, stars_business,
  match_score,                          // 0–100
  sentiment_positive/negative/neutral,  // 0–1 each
  topic_profile: TopicScore[],          // 8 BERTopic clusters
  top_snippets: ReviewSnippet[],
  n_reviews, reason                     // LLM one-sentence reason
}

RecommendResponse {
  query, model_used, results: RestaurantResult[], total_results,
  intent_summary, agent_used,
  agent_response: string,               // 3-5 sentence LLM summary
  agent_citations: AgentCitation[]      // 2-4 grounded evidence snippets
}

HealthResponse { status, sentiment_model, topic_model_loaded, data_loaded, n_restaurants, n_reviews }
```

---

## Customization

- **Colors** — all in `tailwind.config.ts` under `colors`
- **Match score weights** — in `backend/app/core/topics.py → _match_score_v2`
- **Number of results** — change `top_n` in `app/page.tsx → handleSearch`
- **Topic keyword fallback** — in `backend/app/core/topics.py → _keyword_sim`
- **Topic labels** — edit `backend/app/models/topic_labels.json` after running `train.py`
  (use `backend/scripts/rename_topics.py` for guided renaming)
