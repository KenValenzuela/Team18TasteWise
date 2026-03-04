"""
GET /topics — list all BERTopic clusters with labels
"""

from fastapi import APIRouter
from backend.app.core.topics import get_topic_engine
from backend.app.core.config import settings
from backend.app.models.schemas import TopicsResponse, TopicItem

router = APIRouter()


@router.get("", response_model=TopicsResponse)
async def get_topics():
    """Return all discovered BERTopic clusters and their human-readable labels."""
    engine = get_topic_engine()
    topics = [TopicItem(**t) for t in engine.get_topic_list()]
    return TopicsResponse(
        topics=topics,
        n_topics=settings.N_TOPICS,
    )
