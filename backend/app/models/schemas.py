"""
Pydantic schemas for all API request and response bodies.
"""

from pydantic import BaseModel, Field


class AgentCitationSchema(BaseModel):
    restaurant: str
    snippet: str

# ── /recommend ────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        examples=["fast lunch with great value", "quiet date night vibes"],
        description="Natural language description of what you're looking for.",
    )
    top_n: int = Field(default=6, ge=1, le=20)


class TopicScore(BaseModel):
    label: str
    score: float = Field(ge=0.0, le=1.0)


class ReviewSnippet(BaseModel):
    text: str
    sentiment_label: str   # "positive" | "negative" | "neutral"
    stars: int


class RestaurantResult(BaseModel):
    business_id: str
    name: str
    city: str
    categories: str
    stars_business: float
    match_score: float          # 0-100
    sentiment_positive: float
    sentiment_negative: float
    sentiment_neutral: float
    topic_profile: list[TopicScore]
    top_snippets: list[ReviewSnippet]
    n_reviews: int
    reason: str = ""            # agent-generated match reason


class RecommendResponse(BaseModel):
    query: str
    model_used: str
    results: list[RestaurantResult]
    total_results: int
    intent_summary: str
    agent_used: bool

    # ── NEW fields (default to empty so existing clients don't break) ──
    agent_response: str = ""
    agent_citations: list[AgentCitationSchema] = []


# ── /restaurants ──────────────────────────────────────────────────

class RestaurantDetail(BaseModel):
    business_id: str
    name: str
    city: str
    state: str
    categories: str
    stars_business: float
    review_count: int
    sentiment_positive: float
    sentiment_negative: float
    sentiment_neutral: float
    top_topics: list[TopicScore]
    top_snippets: list[ReviewSnippet]


# ── /topics ───────────────────────────────────────────────────────

class TopicItem(BaseModel):
    id: int
    label: str


class TopicsResponse(BaseModel):
    topics: list[TopicItem]
    model: str = "BERTopic + UMAP + HDBSCAN"
    n_topics: int


# ── /health ───────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    sentiment_model: str
    topic_model_loaded: bool
    data_loaded: bool
    n_restaurants: int
    n_reviews: int

