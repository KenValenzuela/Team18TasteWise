# Tastewise — Next.js Frontend
**Team 18 · CIS 509 · Spring 2026**

## Stack
| | |
|---|---|
| Framework | Next.js 14 (App Router) |
| Language | TypeScript |
| Styling | Tailwind CSS |
| Fonts | Cormorant Garamond (display) + Syne (body) |
| API | FastAPI backend via `/api/*` proxy |

---

## Project Structure

```
tastewise-frontend/
├── app/
│   ├── layout.tsx        # Root layout, fonts, metadata
│   ├── page.tsx          # Main page — orchestrates all state
│   └── globals.css       # Tailwind base + grain texture + bar animations
├── components/
│   ├── SearchStage.tsx   # Full-screen hero → collapses to sticky nav
│   ├── RestaurantCard.tsx # Card in the grid
│   ├── SentimentBars.tsx  # Animated pos/neg/neu bars
│   ├── SentimentDonut.tsx # SVG donut chart for the drawer
│   ├── DetailDrawer.tsx   # Slide-in detail panel
│   └── InfoSection.tsx    # Stats + pipeline + bias journal at bottom
├── lib/
│   └── api.ts            # Typed fetch client → FastAPI
├── .env.local            # NEXT_PUBLIC_API_URL
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
# In tastewise-backend/
uvicorn app.main:app --reload --port 8000
```

### 3 — Start Next.js
```bash
npm run dev
# → http://localhost:3000
```

---

## How It Connects to FastAPI

`next.config.js` proxies every `/api/*` request to `http://localhost:8000/*`:

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
      ↓  user types query + hits Enter
2. Stage collapses into sticky nav (with search still visible)
3. POST /api/recommend called with query
4. Cards stagger in with animated sentiment bars
5. Click any card → slide-in DetailDrawer with:
     - Match score
     - Sentiment donut (pos/neg/neu)
     - BERTopic topic bar chart
     - Top review snippets with sentiment labels
     - Model details (VADER vs DistilBERT)
6. Below cards: stats strip + model cards + pipeline + bias journal
```

---

## Customization

- **Colors** — all in `tailwind.config.ts` under `colors`
- **Match score weights** — in `tastewise-backend/app/core/topics.py → _compute_match_score`
- **Number of cards** — change `top_n` in `app/page.tsx → handleSearch`
- **Topic labels** — in `tastewise-backend/app/core/topics.py → TOPIC_LABELS`
  (update after inspecting your fitted BERTopic model's `get_topic_info()`)
