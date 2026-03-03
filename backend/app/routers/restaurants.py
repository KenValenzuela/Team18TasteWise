from fastapi import APIRouter, HTTPException, Query

from backend.app.core.data import load_data
from backend.app.core.topics import get_topic_engine
from backend.app.models.schemas import RestaurantDetail, TopicScore, ReviewSnippet

router = APIRouter()


@router.get("/{business_id}", response_model=RestaurantDetail)
async def get_restaurant(business_id: str):
    """Return full sentiment + topic detail for a single restaurant."""
    rev_rest = load_data()
    engine = get_topic_engine()

    if "business_id" not in rev_rest.columns:
        raise HTTPException(status_code=500, detail="rev_rest dataset missing business_id column.")

    biz_reviews = rev_rest[rev_rest["business_id"] == business_id]
    if biz_reviews.empty:
        raise HTTPException(status_code=404, detail="Restaurant not found.")

    row0 = biz_reviews.iloc[0]

    # Sentiment aggregates
    if "stars_review" in biz_reviews.columns:
        stars = biz_reviews["stars_review"]
    elif "stars" in biz_reviews.columns:
        stars = biz_reviews["stars"]
    else:
        stars = None

    if stars is not None:
        pos = float((stars >= 4).mean())
        neg = float((stars <= 2).mean())
        neu = float((stars == 3).mean())
    else:
        pos, neg, neu = 0.0, 0.0, 0.0

    # Topic profile
    profile = engine.get_restaurant_profile(business_id) or {}
    topic_scores = profile.get("topic_scores")
    if not isinstance(topic_scores, dict):
        topic_scores = {}

    topic_list = [
        TopicScore(label=k, score=v)
        for k, v in sorted(topic_scores.items(), key=lambda x: -x[1])
    ]

    # Snippets (top by useful votes)
    df = biz_reviews.sort_values("useful", ascending=False) if "useful" in biz_reviews.columns else biz_reviews
    snippets: list[ReviewSnippet] = []
    for _, r in df.head(9).iterrows():
        text = str(r.get("text", "")).strip()
        if len(text) < 30:
            continue
        preview = text[:220].rsplit(" ", 1)[0] + "…" if len(text) > 220 else text
        star = int(r.get("stars_review", r.get("stars", 3)))
        snippets.append(
            ReviewSnippet(
                text=preview,
                sentiment_label="positive" if star >= 4 else ("negative" if star <= 2 else "neutral"),
                stars=star,
            )
        )
        if len(snippets) >= 5:
            break

    return RestaurantDetail(
        business_id=business_id,
        name=str(row0.get("name", "")),
        city=str(row0.get("city", "")),
        state=str(row0.get("state", "")),
        categories=str(row0.get("categories", "")),
        stars_business=float(row0.get("stars_business", 0)),
        review_count=int(len(biz_reviews)),
        sentiment_positive=round(pos, 3),
        sentiment_negative=round(neg, 3),
        sentiment_neutral=round(neu, 3),
        top_topics=topic_list[:5],
        top_snippets=snippets,
    )


@router.get("", response_model=list[dict])
async def list_restaurants(
    city: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    """
    Paginated list of all AZ restaurants.
    Optional ?city=Phoenix filter.
    """
    rev_rest = load_data()
    if "business_id" not in rev_rest.columns:
        raise HTTPException(status_code=500, detail="rev_rest dataset missing business_id column.")

    # rev_rest is review-level; build business-level view by de-duping.
    business_df = rev_rest.drop_duplicates(subset=["business_id"]).copy()

    df = business_df
    if city and "city" in df.columns:
        df = df[df["city"].str.lower() == city.lower()]

    start = (page - 1) * page_size
    page_df = df.iloc[start : start + page_size]

    cols = ["business_id", "name", "city", "stars_business", "review_count", "categories"]
    cols = [c for c in cols if c in page_df.columns]
    return page_df[cols].to_dict(orient="records")