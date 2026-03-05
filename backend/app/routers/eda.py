"""
GET /eda — Exploratory Data Analysis endpoint

Returns aggregated statistics about topic and sentiment distributions
across all restaurant profiles for visualization.
"""

from fastapi import APIRouter
from backend.app.core.topics import get_topic_engine

router = APIRouter()


@router.get("")
async def get_eda():
    """Return EDA statistics for topic and sentiment distributions."""
    engine = get_topic_engine()
    profiles = engine._profiles

    topic_labels_list = engine.get_topic_list()
    all_labels = [t["label"] for t in topic_labels_list]

    # --- Topic distribution across all restaurants ---
    topic_avg: dict[str, float] = {label: 0.0 for label in all_labels}
    topic_max: dict[str, float] = {label: 0.0 for label in all_labels}
    topic_dominant_count: dict[str, int] = {label: 0 for label in all_labels}

    sentiment_buckets = {"positive": 0, "negative": 0, "neutral": 0, "mixed": 0}
    sentiment_scatter: list[dict] = []
    city_counts: dict[str, int] = {}
    city_sentiment: dict[str, dict[str, float]] = {}
    stars_distribution: dict[str, int] = {}
    restaurant_details: list[dict] = []

    n = len(profiles)
    if n == 0:
        return {"n_restaurants": 0, "topics": [], "restaurants": []}

    for bid, p in profiles.items():
        ts = p.get("topic_scores", {})
        pos = float(p.get("sentiment_positive", 0))
        neg = float(p.get("sentiment_negative", 0))
        neu = float(p.get("sentiment_neutral", 0))
        name = p.get("name", "")
        city = p.get("city", "")
        stars = float(p.get("stars_business", 0))
        n_reviews = int(p.get("n_reviews", 0))
        categories = p.get("categories", "")

        # Topic aggregates
        for label in all_labels:
            score = float(ts.get(label, 0.0))
            topic_avg[label] += score
            if score > topic_max[label]:
                topic_max[label] = score

        # Dominant topic
        if ts:
            dominant = max(ts, key=ts.get)
            if dominant in topic_dominant_count:
                topic_dominant_count[dominant] += 1

        # Sentiment classification
        if pos >= 0.6:
            sentiment_buckets["positive"] += 1
        elif neg >= 0.4:
            sentiment_buckets["negative"] += 1
        elif neu >= 0.4:
            sentiment_buckets["neutral"] += 1
        else:
            sentiment_buckets["mixed"] += 1

        # Sentiment scatter data
        sentiment_scatter.append({
            "name": name,
            "positive": round(pos, 3),
            "negative": round(neg, 3),
            "neutral": round(neu, 3),
            "stars": stars,
            "n_reviews": n_reviews,
        })

        # City distribution
        if city:
            city_counts[city] = city_counts.get(city, 0) + 1
            if city not in city_sentiment:
                city_sentiment[city] = {"pos_sum": 0.0, "neg_sum": 0.0, "count": 0}
            city_sentiment[city]["pos_sum"] += pos
            city_sentiment[city]["neg_sum"] += neg
            city_sentiment[city]["count"] += 1

        # Stars distribution
        star_key = str(round(stars * 2) / 2)  # bucket by 0.5
        stars_distribution[star_key] = stars_distribution.get(star_key, 0) + 1

        # Per-restaurant topic profile for detail table
        top_topics = sorted(ts.items(), key=lambda x: -x[1])[:3]
        restaurant_details.append({
            "business_id": bid,
            "name": name,
            "city": city,
            "categories": categories[:60],
            "stars": stars,
            "n_reviews": n_reviews,
            "sentiment_positive": round(pos, 3),
            "sentiment_negative": round(neg, 3),
            "top_topic": top_topics[0][0] if top_topics else "",
            "top_topic_score": round(top_topics[0][1], 3) if top_topics else 0,
        })

    # Finalize averages
    for label in all_labels:
        topic_avg[label] = round(topic_avg[label] / n, 4)

    # Build topic summary list
    topics_summary = []
    for label in all_labels:
        topics_summary.append({
            "label": label,
            "avg_score": topic_avg[label],
            "max_score": round(topic_max[label], 4),
            "dominant_count": topic_dominant_count.get(label, 0),
            "dominant_pct": round(topic_dominant_count.get(label, 0) / n * 100, 1),
        })
    topics_summary.sort(key=lambda x: -x["dominant_count"])

    # City summary
    city_summary = []
    for city, count in sorted(city_counts.items(), key=lambda x: -x[1])[:15]:
        cs = city_sentiment[city]
        city_summary.append({
            "city": city,
            "count": count,
            "avg_positive": round(cs["pos_sum"] / cs["count"], 3),
            "avg_negative": round(cs["neg_sum"] / cs["count"], 3),
        })

    # Sort restaurants by sentiment_positive descending
    restaurant_details.sort(key=lambda x: -x["sentiment_positive"])

    return {
        "n_restaurants": n,
        "n_topics": len(all_labels),
        "topics": topics_summary,
        "sentiment_buckets": sentiment_buckets,
        "sentiment_scatter": sentiment_scatter,
        "city_distribution": city_summary,
        "stars_distribution": dict(sorted(stars_distribution.items())),
        "restaurants": restaurant_details,
    }
