"""
Tastewise — FastAPI Backend
Team 18 · CIS 509

Entry point. Run with:
    uvicorn app.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.routers import recommend, restaurants, topics, health, eda
from backend.app.core.config import settings

app = FastAPI(
    title="Tastewise API",
    description="Yelp-powered restaurant recommender using sentiment analysis and BERTopic.",
    version="1.0.0",
)

# ── CORS (allow Next.js dev server) ──────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────
app.include_router(health.router, tags=["Health"])
app.include_router(recommend.router, prefix="/recommend", tags=["Recommend"])
app.include_router(restaurants.router, prefix="/restaurants", tags=["Restaurants"])
app.include_router(topics.router, prefix="/topics", tags=["Topics"])
app.include_router(eda.router, prefix="/eda", tags=["EDA"])
