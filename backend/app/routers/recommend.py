"""
POST /recommend

Agent-enhanced recommendation endpoint:
  1. Agent parses query → structured ParsedIntent (topic weights + intent summary)
  2. Ranker uses those weights instead of raw keyword matching
  3. Agent narrates each result → one-sentence personalised reason
  4. Agent summarizes results → grounded conversational response   ← NEW
  5. Response includes intent_summary + per-restaurant reason + agent summary
"""

import re

from fastapi import APIRouter, HTTPException

from backend.app.core.agent import get_agent
from backend.app.core.sentiment import get_sentiment_engine
from backend.app.core.topics import get_topic_engine
from backend.app.models.schemas import (
    AgentCitationSchema,
    RecommendRequest,
    RecommendResponse,
    RestaurantResult,
    ReviewSnippet,
    TopicScore,
)

router = APIRouter()

_STOP = {"a", "an", "the", "and", "or", "but", "with", "for", "to", "of", "in", "on", "at"}
_WORD_RE = re.compile(r"[a-z0-9]+")


def _sort_snippets_by_relevance(snippets: list, query: str) -> list:
    """Return snippets sorted so the most query-relevant ones come first."""
    q_toks = {t for t in _WORD_RE.findall(query.lower()) if t not in _STOP and len(t) > 2}
    if not q_toks:
        return snippets

    def score(s: dict) -> int:
        text = s.get("text", "").lower()
        return sum(1 for tok in q_toks if tok in text)

    return sorted(snippets, key=score, reverse=True)


@router.post("", response_model=RecommendResponse)
async def recommend(body: RecommendRequest):
    """
    Rank restaurants by matching a natural language query.

    With OPENAI_API_KEY set:
      - LLM parses the query into topic weights + intent summary
      - LLM writes a one-sentence reason per result
      - LLM generates a grounded conversational summary with citations

    Without key (or AGENT_ENABLED=False):
      - Falls back to keyword matching + template reasons + template summary
    """
    try:
        engine = get_topic_engine()
        sentiment_engine = get_sentiment_engine()
        agent = get_agent()

        # Step 1: Parse query intent
        intent = agent.parse_query(body.query)

        # Step 2: Rank restaurants with agent-provided topic weights
        ranked = engine.rank_by_query(
            query=body.query,
            top_n=body.top_n,
            topic_weight_overrides=intent.topic_weights,
            sentiment_threshold=intent.sentiment_threshold,
        )

        # Step 3: Narrate results (per-card reasons)
        narrate_inputs = [
            {
                "business_id": r.business_id,
                "name": r.name,
                "top_topic": r.top_topic,
                "sentiment_positive": r.sentiment_positive,
                "match_score": r.match_score,
            }
            for r in ranked
        ]
        narratives = agent.narrate_results(body.query, narrate_inputs)

        # Step 4: Summarize results (grounded conversational response)  ← NEW
        summarize_inputs = [
            {
                "business_id": r.business_id,
                "name": r.name,
                "city": r.city,
                "categories": r.categories,
                "stars_business": r.stars_business,
                "match_score": r.match_score,
                "sentiment_positive": r.sentiment_positive,
                "sentiment_negative": r.sentiment_negative,
                "sentiment_neutral": r.sentiment_neutral,
                "n_reviews": r.n_reviews,
                "top_topic": r.top_topic,
                "top_snippets": r.top_snippets,  # pass full snippets for evidence
            }
            for r in ranked
        ]
        summary = agent.summarize_results(body.query, summarize_inputs)

    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    # Build response results
    results = []
    for r in ranked:
        topic_list = [
            TopicScore(label=k, score=v)
            for k, v in sorted(r.topic_profile.items(), key=lambda x: -x[1])
        ]
        sorted_snips = _sort_snippets_by_relevance(r.top_snippets, body.query)
        snippets = [
            ReviewSnippet(
                text=s["text"],
                sentiment_label=s["sentiment_label"],
                stars=s["stars"],
            )
            for s in sorted_snips
        ]
        results.append(RestaurantResult(
            business_id=r.business_id,
            name=r.name,
            city=r.city,
            categories=r.categories,
            stars_business=r.stars_business,
            match_score=r.match_score,
            sentiment_positive=r.sentiment_positive,
            sentiment_negative=r.sentiment_negative,
            sentiment_neutral=r.sentiment_neutral,
            topic_profile=topic_list,
            top_snippets=snippets,
            n_reviews=r.n_reviews,
            reason=narratives.get(r.business_id, ""),
        ))

    return RecommendResponse(
        query=body.query,
        intent_summary=intent.intent_summary,
        agent_used=intent.agent_used,
        model_used=sentiment_engine.model_name,
        results=results,
        total_results=len(results),
        # ── NEW fields ──
        agent_response=summary.response,
        agent_citations=[
            AgentCitationSchema(restaurant=c.restaurant, snippet=c.snippet)
            for c in summary.citations
        ],
    )