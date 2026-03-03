// lib/api.ts
// Typed API client for the Tastewise FastAPI backend.
// All types mirror app/models/schemas.py exactly.

const BASE = "/api"; // proxied by next.config.js → http://localhost:8000

// ── Types ────────────────────────────────────────────────────────

export interface TopicScore {
  label: string;
  score: number; // 0-1
}

export interface ReviewSnippet {
  text: string;
  sentiment_label: "positive" | "negative" | "neutral";
  stars: number;
}

export interface AgentCitation {
  restaurant: string; // name of cited restaurant
  snippet: string;    // review snippet used as evidence
}

export interface RestaurantResult {
  business_id: string;
  name: string;
  city: string;
  categories: string;
  stars_business: number;
  match_score: number; // 0-100
  sentiment_positive: number;
  sentiment_negative: number;
  sentiment_neutral: number;
  topic_profile: TopicScore[];
  top_snippets: ReviewSnippet[];
  n_reviews: number;
  reason: string;           // agent-generated match reason
}

export interface RecommendResponse {
  query: string;
  model_used: string;
  results: RestaurantResult[];
  total_results: number;
  intent_summary: string;   // "Finding a quiet date spot with excellent wine…"
  agent_used: boolean;

  // ── NEW: LLM agent response ──
  agent_response: string;        // grounded natural-language summary
  agent_citations?: AgentCitation[]; // evidence snippets the agent cited
}

export interface HealthResponse {
  status: string;
  sentiment_model: string;
  topic_model_loaded: boolean;
  data_loaded: boolean;
  n_restaurants: number;
  n_reviews: number;
}

export interface TopicItem {
  id: number;
  label: string;
}

// ── API calls ─────────────────────────────────────────────────────

export async function recommend(
  query: string,
  top_n = 6
): Promise<RecommendResponse> {
  const res = await fetch(`${BASE}/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_n }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err?.detail ?? `API error ${res.status}`);
  }

  return res.json();
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BASE}/health`);
  if (!res.ok) throw new Error("Health check failed");
  return res.json();
}

export async function getTopics(): Promise<TopicItem[]> {
  const res = await fetch(`${BASE}/topics`);
  if (!res.ok) throw new Error("Topics fetch failed");
  const data = await res.json();
  return data.topics;
}