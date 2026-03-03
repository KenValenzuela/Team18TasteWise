"""
Central config — reads from .env (or environment variables).
Copy .env.example → .env and fill in your values.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Paths to your Yelp JSON files (relative to project root)
    BUSINESS_JSON: str = "data/yelp_academic_dataset_business.json"
    REVIEW_JSON: str = "data/yelp_academic_dataset_review.json"

    # Filter scope — must match your EDA (AZ only)
    FILTER_STATE: str = "AZ"
    MIN_BIZ_REVIEWS: int = 5          # ignore businesses with fewer reviews

    # BERTopic
    BERTOPIC_MODEL_PATH: str = "models/bertopic_model"   # saved after first run
    N_TOPICS: int = 8                  # matches your EDA target k
    BERTOPIC_DOCS_SAMPLE: int = 2000   # same as EDA cell (N_DOCS = 2000)

    # Sentiment model
    SENTIMENT_MODEL: str = "distilbert-base-uncased"
    # Use absolute path or relative to backend/app/
    SENTIMENT_MODEL_PATH: str = ""  # set in .env once you have the model

    # Recommendation
    TOP_N: int = 10                    # max results returned
    SNIPPET_MAX_PER_RESTAURANT: int = 5

    # AI Agent
    OPENAI_API_KEY: str = ""          # set in .env — never commit the real key
    AGENT_MODEL: str = "gpt-4o-mini" #
    AGENT_ENABLED: bool = True           # set False to skip agent and use keyword ranker only

    # CORS
    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    class Config:
        env_file = ".env"


settings = Settings()
