from fastapi import APIRouter
from backend.app.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    """
    Returns the status of all backend components.
    Call this from your frontend to check if the API is ready.
    """
    from backend.app.core.data import load_data
    from backend.app.core.sentiment import get_sentiment_engine
    from backend.app.core.topics import get_topic_engine

    try:
        rev_rest = load_data()  # load_data() returns ONE dataframe
        n_reviews = int(len(rev_rest))
        n_restaurants = int(rev_rest["business_id"].nunique()) if "business_id" in rev_rest.columns else 0
        data_ok = True
    except Exception:
        n_reviews = 0
        n_restaurants = 0
        data_ok = False

    sentiment_engine = get_sentiment_engine()
    topic_engine = get_topic_engine()

    # TopicEngine uses _bertopic
    topic_model_loaded = bool(getattr(topic_engine, "_bertopic", None))

    return HealthResponse(
        status="ok" if data_ok else "degraded",
        sentiment_model=sentiment_engine.model_name,
        topic_model_loaded=topic_model_loaded,
        data_loaded=data_ok,
        n_restaurants=n_restaurants,
        n_reviews=n_reviews,
    )