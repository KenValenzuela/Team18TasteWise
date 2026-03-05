"use client";

import { useState, useEffect, useCallback } from "react";
import { recommend, getHealth } from "@/lib/api";
import type { RestaurantResult, RecommendResponse, HealthResponse } from "@/lib/api";

import SearchStage from "@/components/SearchStage";
import RestaurantCard from "@/components/RestaurantCard";
import DetailDrawer from "@/components/DetailDrawer";
import InfoSection from "@/components/InfoSection";
import AgentResponse from "@/components/AgentResponse"; // ← NEW

export default function Home() {
  const [searched, setSearched] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [response, setResponse] = useState<RecommendResponse | null>(null);
  const [selected, setSelected] = useState<RestaurantResult | null>(null);
  const [selectedIndex, setSelectedIndex] = useState<number>(-1);
  const [health, setHealth] = useState<HealthResponse | null>(null);

  // Fetch health on mount (shows stats in footer)
  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => {}); // non-fatal
  }, []);

  const handleSearch = useCallback(async (query: string) => {
    setLoading(true);
    setError(null);

    try {
      const data = await recommend(query, 6);
      setResponse(data);
      setSearched(true);
    } catch (err: unknown) {
      const msg =
        err instanceof Error
          ? err.message
          : "Could not reach the backend. Make sure FastAPI is running on :8000.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, []);

  const results: RestaurantResult[] = response?.results ?? [];

  return (
    <main>
      {/* ── Search Stage (full-screen → sticky nav) ── */}
      <SearchStage
        collapsed={searched}
        loading={loading}
        onSearch={handleSearch}
      />

      {/* ── Error banner ── */}
      {error && (
        <div className="mx-6 mt-4 px-4 py-3 border border-sentiment-neg/40 bg-sentiment-neg/10 rounded-sm text-[0.8rem] text-sentiment-neg">
          {error}
        </div>
      )}

      {/* ── Results ── */}
      {searched && (
        <section
          className={`transition-all duration-600 ease-out ${
            results.length > 0
              ? "opacity-100 translate-y-0"
              : "opacity-0 translate-y-4"
          }`}
        >
          {/* Query header */}
          <div className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-1 px-6 pt-6 pb-3">
            <div className="flex items-center gap-3">
              <span className="text-[0.68rem] tracking-[0.14em] uppercase text-[#6b6455]">
                Top matches
              </span>
              {response?.agent_used && (
                <span className="text-[0.58rem] tracking-[0.1em] uppercase px-2 py-0.5 rounded-full border border-gold-dim text-gold bg-[rgba(200,151,62,0.07)]">
                  ✦ AI-ranked
                </span>
              )}
            </div>
            <div className="flex flex-col items-end gap-0.5">
              {response && (
                <span className="font-display text-[1rem] italic text-gold-light">
                  &ldquo;{response.query}&rdquo;
                </span>
              )}
              {response?.intent_summary && (
                <span className="text-[0.7rem] text-[#6b6455]">
                  {response.intent_summary}
                </span>
              )}
            </div>
          </div>

          {/* ── NEW: Agent LLM Response ── */}
          {response?.agent_response && (
            <AgentResponse
              text={response.agent_response}
              citations={response.agent_citations}
              streaming={loading}
              visible={!!response.agent_response}
            />
          )}

          {/* Card grid */}
          {results.length > 0 ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 border-t border-l border-border mx-0">
              {results.map((r, i) => (
                <RestaurantCard
                  key={r.business_id}
                  restaurant={r}
                  rank={i + 1}
                  delay={i * 75}
                  onClick={() => { setSelected(r); setSelectedIndex(i); }}
                />
              ))}
            </div>
          ) : (
            !loading && (
              <div className="px-6 py-12 text-center text-[#6b6455] text-[0.85rem]">
                No results found. Try a broader query.
              </div>
            )
          )}
        </section>
      )}

      {/* ── Info section (bottom) ── */}
      <InfoSection health={health} visible={searched} />

      {/* ── Detail drawer ── */}
      <DetailDrawer
        restaurant={selected}
        modelUsed={response?.model_used ?? "vader"}
        onClose={() => { setSelected(null); setSelectedIndex(-1); }}
        currentIndex={selectedIndex}
        totalResults={results.length}
        onNavigate={(dir) => {
          const next = dir === "prev" ? selectedIndex - 1 : selectedIndex + 1;
          if (next >= 0 && next < results.length) {
            setSelected(results[next]);
            setSelectedIndex(next);
          }
        }}
        currentQuery={response?.query ?? ""}
      />
    </main>
  );
}