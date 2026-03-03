"""
Agent Layer — OpenAI tool-calling integration
============================================

Three responsibilities:

  1. parse_query(query) → ParsedIntent
     Converts free-text like "quiet date night, good wine"
     into structured weights the ranker can use directly.
     Falls back to keyword matching if agent is disabled or key is missing.

  2. narrate_results(query, results) → dict[business_id, reason]
     Generates a one-sentence personalised reason for each restaurant
     explaining WHY it matches the query.
     Falls back to a template string if agent is disabled.

  3. summarize_results(query, results) → AgentSummary     ← NEW
     Generates a grounded conversational response (3-5 sentences)
     that synthesises the top results with evidence from review snippets.
     Falls back to a template summary if agent is disabled.

Uses OpenAI function/tool calling so responses are always machine-readable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional

from backend.app.core.config import settings

logger = logging.getLogger(__name__)

# Topics must match your downstream ranker/topic labels
KNOWN_TOPICS = [
    "Food Quality",
    "Service & Staff",
    "Value & Price",
    "Wait Time",
    "Ambience & Vibe",
    "Parking & Location",
    "Brunch & Breakfast",
    "Drinks & Bar",
]


# ── Data classes ─────────────────────────────────────────────────

@dataclass
class ParsedIntent:
    topic_weights: dict[str, float] = field(default_factory=dict)
    sentiment_threshold: float = 0.0
    keywords: list[str] = field(default_factory=list)
    intent_summary: str = ""
    agent_used: bool = False


@dataclass
class AgentCitation:
    """A single evidence reference linking a restaurant to a review snippet."""
    restaurant: str
    snippet: str


@dataclass
class AgentSummary:
    """The grounded conversational response returned to the frontend."""
    response: str
    citations: list[AgentCitation] = field(default_factory=list)


# ── Agent class ───────────────────────────────────────────────────

class TastewiseAgent:
    """
    Thin wrapper around the OpenAI client.
    Uses tool calls to guarantee structured output.
    """

    # ── Tool schemas ──────────────────────────────────────────────

    PARSE_TOOL = {
        "type": "function",
        "function": {
            "name": "parse_restaurant_query",
            "description": (
                "Parse a user's natural language restaurant query into structured "
                "intent weights for a recommendation engine."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic_weights": {
                        "type": "object",
                        "description": (
                            "Map topic label -> importance weight (0.0-1.0). "
                            "Only include up to 3 non-zero topics. Weights must sum to 1.0. "
                            f"Valid topics: {', '.join(KNOWN_TOPICS)}."
                        ),
                        "additionalProperties": {"type": "number"},
                    },
                    "sentiment_threshold": {
                        "type": "number",
                        "description": (
                            "Minimum acceptable positive sentiment ratio (0.0-1.0). "
                            "Use 0.0 unless the user implies 'only very positive'."
                        ),
                    },
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Up to 8 key concepts extracted from the query.",
                    },
                    "intent_summary": {
                        "type": "string",
                        "description": (
                            "One short sentence (max 12 words) describing what the user wants. "
                            "Start with a verb."
                        ),
                    },
                },
                "required": ["topic_weights", "sentiment_threshold", "keywords", "intent_summary"],
                "additionalProperties": False,
            },
        },
    }

    NARRATE_TOOL = {
        "type": "function",
        "function": {
            "name": "narrate_restaurant_matches",
            "description": "Generate a personalised one-sentence reason for each restaurant match.",
            "parameters": {
                "type": "object",
                "properties": {
                    "narratives": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "business_id": {"type": "string"},
                                "reason": {
                                    "type": "string",
                                    "description": (
                                        "One sentence (max 20 words) explaining why this restaurant matches."
                                    ),
                                },
                            },
                            "required": ["business_id", "reason"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["narratives"],
                "additionalProperties": False,
            },
        },
    }

    # ── NEW: Summarize tool schema ────────────────────────────────

    SUMMARIZE_TOOL = {
        "type": "function",
        "function": {
            "name": "summarize_recommendations",
            "description": (
                "Generate a grounded conversational summary of the top restaurant "
                "recommendations, citing specific review evidence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": (
                            "A 3-5 sentence conversational paragraph explaining why these "
                            "restaurants match the user's query. Mention specific restaurant "
                            "names and reference concrete details from review snippets. "
                            "Every claim must be supported by a provided snippet."
                        ),
                    },
                    "citations": {
                        "type": "array",
                        "description": "Evidence snippets used in the summary. Include 2-4.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "restaurant": {
                                    "type": "string",
                                    "description": "Restaurant name.",
                                },
                                "snippet": {
                                    "type": "string",
                                    "description": "The review snippet text used as evidence.",
                                },
                            },
                            "required": ["restaurant", "snippet"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["summary", "citations"],
                "additionalProperties": False,
            },
        },
    }

    # ── Init ──────────────────────────────────────────────────────

    def __init__(self) -> None:
        self._client = None
        if settings.AGENT_ENABLED and getattr(settings, "OPENAI_API_KEY", ""):
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
                logger.info("OpenAI agent ready — model: %s", settings.AGENT_MODEL)
            except Exception as exc:
                logger.warning("OpenAI client failed to init: %s. Agent disabled.", exc)
                self._client = None
        else:
            logger.info(
                "Agent disabled (AGENT_ENABLED=%s, key present=%s). Using keyword fallback.",
                settings.AGENT_ENABLED,
                bool(getattr(settings, "OPENAI_API_KEY", "")),
            )

    @property
    def enabled(self) -> bool:
        return self._client is not None

    # ── Public API ────────────────────────────────────────────────

    def parse_query(self, query: str) -> ParsedIntent:
        if not self.enabled:
            return self._keyword_fallback(query)
        try:
            return self._agent_parse(query)
        except Exception as exc:
            logger.warning("Agent parse failed (%s) — using keyword fallback.", exc)
            return self._keyword_fallback(query)

    def narrate_results(self, query: str, results: list[dict]) -> dict[str, str]:
        if not self.enabled or not results:
            return self._template_narratives(results)
        try:
            return self._agent_narrate(query, results)
        except Exception as exc:
            logger.warning("Agent narrate failed (%s) — using templates.", exc)
            return self._template_narratives(results)

    def summarize_results(self, query: str, results: list[dict]) -> AgentSummary:
        """
        NEW — Generate a grounded conversational summary of recommendations.

        Args:
            query:   The user's original natural-language query.
            results: List of ranked RestaurantResult dicts (must include
                     name, match_score, sentiment_positive, top_snippets, etc.).

        Returns:
            AgentSummary with .response (str) and .citations (list[AgentCitation]).
        """
        if not self.enabled or not results:
            return self._template_summary(query, results)
        try:
            return self._agent_summarize(query, results)
        except Exception as exc:
            logger.warning("Agent summarize failed (%s) — using template.", exc)
            return self._template_summary(query, results)

    # ── OpenAI calls ───────────────────────────────────────────────

    def _agent_parse(self, query: str) -> ParsedIntent:
        system = (
            "You parse restaurant queries into structured intent for ranking.\n"
            "Rules:\n"
            " - Choose at most 3 non-zero topics.\n"
            " - topic_weights must sum to 1.0.\n"
            " - Only use valid topics.\n"
            " - Keep keywords short; max 8.\n"
            " - intent_summary max 12 words.\n"
            "Always call the parse_restaurant_query tool."
        )

        resp = self._client.chat.completions.create(
            model=settings.AGENT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
            tools=[self.PARSE_TOOL],
            tool_choice={"type": "function", "function": {"name": "parse_restaurant_query"}},
            temperature=0.2,
        )

        data = self._extract_tool_args(resp, "parse_restaurant_query")
        intent = ParsedIntent(
            topic_weights=self._postprocess_topic_weights(data.get("topic_weights", {})),
            sentiment_threshold=self._clamp01(data.get("sentiment_threshold", 0.0)),
            keywords=self._postprocess_keywords(data.get("keywords", [])),
            intent_summary=str(data.get("intent_summary", "") or "").strip(),
            agent_used=True,
        )
        return intent

    def _agent_narrate(self, query: str, results: list[dict]) -> dict[str, str]:
        payload = [
            {
                "business_id": r.get("business_id", ""),
                "name": r.get("name", ""),
                "top_topic": r.get("top_topic", ""),
                "sentiment_positive": round(float(r.get("sentiment_positive", 0.0) or 0.0), 2),
                "match_score": r.get("match_score", 0),
            }
            for r in results[:6]
        ]

        system = (
            "You write one-sentence match reasons for a restaurant recommender.\n"
            "Rules:\n"
            " - 1 sentence, <= 20 words\n"
            " - Mention at least one provided field (top_topic/sentiment_positive/match_score)\n"
            " - No invented details\n"
            "Always call the narrate_restaurant_matches tool."
        )

        user_msg = (
            f'User query: "{query}"\n\n'
            f"Matched restaurants (JSON):\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
        )

        resp = self._client.chat.completions.create(
            model=settings.AGENT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            tools=[self.NARRATE_TOOL],
            tool_choice={"type": "function", "function": {"name": "narrate_restaurant_matches"}},
            temperature=0.3,
        )

        data = self._extract_tool_args(resp, "narrate_restaurant_matches")
        narratives = data.get("narratives", []) or []
        out: dict[str, str] = {}
        for n in narratives:
            if not isinstance(n, dict):
                continue
            bid = str(n.get("business_id", "")).strip()
            reason = str(n.get("reason", "")).strip()
            if bid:
                out[bid] = reason
        return out

    # ── NEW: Summarize call ───────────────────────────────────────

    def _agent_summarize(self, query: str, results: list[dict]) -> AgentSummary:
        """
        Calls the LLM with ranked results + review snippets as context.
        Uses tool calling to get structured {summary, citations} back.
        """
        # Build context: top results with their best review snippets
        context_items = []
        for i, r in enumerate(results[:6], 1):
            snippets = r.get("top_snippets", [])[:3]
            snippet_lines = "\n".join(
                f'      - [{s.get("sentiment_label", "?").upper()}] '
                f'"{s.get("text", "")}"'
                for s in snippets
                if isinstance(s, dict) and s.get("text")
            )
            item = (
                f'{i}. {r.get("name", "Unknown")} '
                f'(match: {r.get("match_score", 0)}, '
                f'stars: {float(r.get("stars_business", 0)):.1f}, '
                f'{r.get("n_reviews", 0)} reviews)\n'
                f'   Categories: {r.get("categories", "")}\n'
                f'   Sentiment: {float(r.get("sentiment_positive", 0)):.0%} positive\n'
                f'   Top review snippets:\n{snippet_lines}'
            )
            context_items.append(item)

        context = "\n\n".join(context_items)

        system = (
            "You are Tastewise, a restaurant recommendation assistant for Arizona.\n"
            "You explain WHY the recommended restaurants match the user's query.\n\n"
            "RULES:\n"
            " - Every claim MUST be supported by a review snippet provided below.\n"
            " - If no snippet supports a claim, do NOT make it.\n"
            " - Write 3-5 conversational sentences.\n"
            " - Mention specific restaurant names.\n"
            " - Reference concrete details from snippets (speed, price, vibe, etc.).\n"
            " - Do NOT invent facts or reviews.\n"
            " - Include 2-4 citations (the strongest evidence snippets).\n"
            "Always call the summarize_recommendations tool."
        )

        user_msg = (
            f'User query: "{query}"\n\n'
            f"Top-ranked restaurants with review evidence:\n\n{context}"
        )

        resp = self._client.chat.completions.create(
            model=settings.AGENT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            tools=[self.SUMMARIZE_TOOL],
            tool_choice={
                "type": "function",
                "function": {"name": "summarize_recommendations"},
            },
            temperature=0.3,
        )

        data = self._extract_tool_args(resp, "summarize_recommendations")

        # Parse citations
        raw_citations = data.get("citations", []) or []
        citations: list[AgentCitation] = []
        for c in raw_citations:
            if not isinstance(c, dict):
                continue
            name = str(c.get("restaurant", "")).strip()
            snip = str(c.get("snippet", "")).strip()
            if name and snip:
                citations.append(AgentCitation(restaurant=name, snippet=snip))

        return AgentSummary(
            response=str(data.get("summary", "") or "").strip(),
            citations=citations,
        )

    # ── Tool-call extraction + validation ──────────────────────────

    def _extract_tool_args(self, resp: Any, fn_name: str) -> dict[str, Any]:
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []
        call = None
        for tc in tool_calls:
            if getattr(tc, "type", None) == "function" and tc.function and tc.function.name == fn_name:
                call = tc
                break
        if call is None:
            raise ValueError(f"No tool call for {fn_name}")

        args_str = call.function.arguments or "{}"
        return self._safe_json_loads(args_str)

    def _safe_json_loads(self, s: str) -> dict[str, Any]:
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            s2 = (s or "").strip()
            if s2.startswith("```"):
                s2 = s2.strip("`")
            try:
                obj = json.loads(s2)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}

    def _clamp01(self, x: Any) -> float:
        try:
            v = float(x)
        except Exception:
            return 0.0
        return max(0.0, min(v, 1.0))

    def _postprocess_topic_weights(self, tw: Any) -> dict[str, float]:
        if not isinstance(tw, dict):
            return {}
        cleaned: dict[str, float] = {}
        for k, v in tw.items():
            topic = str(k).strip()
            if topic not in KNOWN_TOPICS:
                continue
            try:
                w = float(v)
            except Exception:
                continue
            if w <= 0:
                continue
            cleaned[topic] = min(w, 1.0)

        items = sorted(cleaned.items(), key=lambda kv: kv[1], reverse=True)[:3]
        if not items:
            return {}

        s = sum(w for _, w in items)
        if s <= 0:
            return {}
        return {k: round(w / s, 4) for k, w in items}

    def _postprocess_keywords(self, kws: Any) -> list[str]:
        if not isinstance(kws, list):
            return []
        out: list[str] = []
        for x in kws:
            s = str(x).strip()
            if not s:
                continue
            out.append(s[:32])
            if len(out) >= 8:
                break
        return out

    # ── Fallbacks ─────────────────────────────────────────────────

    def _keyword_fallback(self, query: str) -> ParsedIntent:
        q = (query or "").lower()

        keyword_map = {
            "Food Quality": ["food", "flavor", "taste", "authentic", "delicious", "fresh", "quality"],
            "Service & Staff": ["service", "staff", "server", "waiter", "waitress", "attentive", "friendly"],
            "Value & Price": ["value", "price", "affordable", "cheap", "worth", "budget", "inexpensive"],
            "Wait Time": ["wait", "slow", "fast", "quick", "speed", "efficient", "immediate"],
            "Ambience & Vibe": ["vibe", "atmosphere", "ambience", "quiet", "cozy", "date", "romantic", "intimate"],
            "Parking & Location": ["parking", "location", "distance", "close", "convenient", "park"],
            "Brunch & Breakfast": ["brunch", "breakfast", "mimosa", "eggs", "morning"],
            "Drinks & Bar": ["drinks", "bar", "cocktail", "beer", "wine", "happy hour"],
        }

        raw: dict[str, float] = {}
        for topic, words in keyword_map.items():
            hits = sum(1 for w in words if w in q)
            if hits:
                raw[topic] = min(hits * 0.35, 1.0)

        tw = self._postprocess_topic_weights(raw)

        sentiment = 0.0
        if any(x in q for x in ["best", "top", "amazing", "incredible", "excellent"]):
            sentiment = 0.5

        return ParsedIntent(
            topic_weights=tw,
            sentiment_threshold=sentiment,
            keywords=[w for w in q.split() if len(w) > 3][:8],
            intent_summary="",
            agent_used=False,
        )

    def _template_narratives(self, results: list[dict]) -> dict[str, str]:
        out: dict[str, str] = {}
        for r in results:
            bid = str(r.get("business_id", "")).strip()
            if not bid:
                continue
            top = str(r.get("top_topic", "the dining experience") or "the dining experience")
            pct = round(float(r.get("sentiment_positive", 0) or 0) * 100)
            out[bid] = f"Strong on {top.lower()} — about {pct}% positive reviews."
        return out

    # ── NEW: Template summary fallback ────────────────────────────

    def _template_summary(self, query: str, results: list[dict]) -> AgentSummary:
        """
        Deterministic summary when LLM is unavailable.
        Uses real data from ranked results — no invented claims.
        """
        if not results:
            return AgentSummary(
                response="No matching restaurants found for your query.",
                citations=[],
            )

        top = results[0]
        citations: list[AgentCitation] = []

        # Pick the best positive snippet from #1
        top_snippet = self._best_positive_snippet(top)

        parts = [
            f'For "{query}", {top.get("name", "the top result")} is the strongest match '
            f'with a score of {top.get("match_score", 0)} and '
            f'{float(top.get("sentiment_positive", 0)):.0%} positive sentiment '
            f'across {top.get("n_reviews", 0)} reviews.'
        ]

        if top_snippet:
            snip_text = top_snippet.get("text", "")
            display = f'{snip_text[:120]}…' if len(snip_text) > 120 else snip_text
            parts.append(f'Reviewers highlight: "{display}"')
            citations.append(AgentCitation(
                restaurant=top.get("name", ""),
                snippet=snip_text,
            ))

        if len(results) > 1:
            runner = results[1]
            parts.append(
                f'{runner.get("name", "The runner-up")} is also worth considering '
                f'with a {runner.get("match_score", 0)} match score and '
                f'{float(runner.get("stars_business", 0)):.1f} stars.'
            )
            runner_snippet = self._best_positive_snippet(runner)
            if runner_snippet:
                citations.append(AgentCitation(
                    restaurant=runner.get("name", ""),
                    snippet=runner_snippet.get("text", ""),
                ))

        if len(results) > 2:
            third = results[2]
            parts.append(
                f'{third.get("name", "Another strong option")} rounds out the '
                f'top three with {float(third.get("sentiment_positive", 0)):.0%} '
                f'positive sentiment.'
            )

        return AgentSummary(
            response=" ".join(parts),
            citations=citations,
        )

    def _best_positive_snippet(self, result: dict) -> dict | None:
        """Return the first positive snippet, or the first snippet if none are positive."""
        snippets = result.get("top_snippets", [])
        if not snippets:
            return None
        for s in snippets:
            if isinstance(s, dict) and s.get("sentiment_label") == "positive":
                return s
        return snippets[0] if isinstance(snippets[0], dict) else None


@lru_cache(maxsize=1)
def get_agent() -> TastewiseAgent:
    return TastewiseAgent()