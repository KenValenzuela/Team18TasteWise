"use client";

import { useEffect, useMemo } from "react";
import clsx from "clsx";
import type { RestaurantResult, ReviewSnippet } from "@/lib/api";
import SentimentDonut from "./SentimentDonut";

const SENT_LABEL = {
  positive: "Positive",
  negative: "Negative",
  neutral: "Neutral",
};

const SENT_COLOR = {
  positive: "text-sentiment-pos bg-[rgba(90,158,114,0.12)]",
  negative: "text-sentiment-neg bg-[rgba(184,82,82,0.12)]",
  neutral: "text-sentiment-neu bg-[rgba(107,127,158,0.12)]",
};

interface Props {
  restaurant: RestaurantResult | null;
  modelUsed: string;
  onClose: () => void;
  currentIndex?: number;
  totalResults?: number;
  onNavigate?: (dir: "prev" | "next") => void;
  currentQuery?: string;
}

/** Return the meaningful tokens from the current query */
function queryTokens(query: string): string[] {
  const stop = new Set([
    "a", "an", "the", "and", "or", "is", "in", "of", "for", "to", "at",
    "what", "nice", "good", "me", "some", "any", "my",
  ]);
  return query
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 2 && !stop.has(t));
}

/** True when the snippet text contains any query keyword */
function snippetMatchesQuery(snippet: ReviewSnippet, toks: string[]): boolean {
  if (!toks.length) return false;
  const text = snippet.text.toLowerCase();
  return toks.some((t) => text.includes(t));
}

/** Render snippet text with query keywords highlighted in gold */
function HighlightedText({ text, toks }: { text: string; toks: string[] }) {
  if (!toks.length) return <>{text}</>;

  const escaped = toks.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const re = new RegExp(`(${escaped.join("|")})`, "gi");
  const parts = text.split(re);

  return (
    <>
      {parts.map((part, i) =>
        re.test(part) ? (
          <mark key={i} className="bg-gold/20 text-gold-light rounded-[2px] px-[1px] not-italic">
            {part}
          </mark>
        ) : (
          <span key={i}>{part}</span>
        )
      )}
    </>
  );
}

export default function DetailDrawer({
  restaurant: r,
  modelUsed,
  onClose,
  currentIndex = -1,
  totalResults = 0,
  onNavigate,
  currentQuery = "",
}: Props) {
  const open = r !== null;
  const qToks = useMemo(() => queryTokens(currentQuery), [currentQuery]);

  // Close on Escape; navigate on arrow keys
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowLeft" && onNavigate) onNavigate("prev");
      if (e.key === "ArrowRight" && onNavigate) onNavigate("next");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, onNavigate]);

  // Lock body scroll when open
  useEffect(() => {
    document.body.style.overflow = open ? "hidden" : "";
    return () => { document.body.style.overflow = ""; };
  }, [open]);

  const hasPrev = currentIndex > 0;
  const hasNext = currentIndex >= 0 && currentIndex < totalResults - 1;

  // Snippets: query-matching ones float to the top
  const sortedSnippets = useMemo(() => {
    if (!r?.top_snippets) return [];
    const matching = r.top_snippets.filter((s) => snippetMatchesQuery(s, qToks));
    const rest = r.top_snippets.filter((s) => !snippetMatchesQuery(s, qToks));
    return [...matching, ...rest];
  }, [r, qToks]);

  const hasQueryMatches = sortedSnippets.some((s) => snippetMatchesQuery(s, qToks));

  return (
    <>
      {/* Overlay */}
      <div
        onClick={onClose}
        className={clsx(
          "fixed inset-0 z-[200] bg-black/60 backdrop-blur-sm",
          "transition-opacity duration-300",
          open ? "opacity-100 pointer-events-auto" : "opacity-0 pointer-events-none"
        )}
      />

      {/* Drawer */}
      <div
        className={clsx(
          "fixed right-0 top-0 bottom-0 z-[201]",
          "w-full max-w-[440px] bg-[#111009] border-l border-border",
          "overflow-y-auto transition-transform duration-[450ms] ease-[cubic-bezier(0.16,1,0.3,1)]",
          open ? "translate-x-0" : "translate-x-full"
        )}
      >
        {r && (
          <div className="p-7">
            {/* Top bar: close + navigation */}
            <div className="flex items-center justify-between mb-6">
              <button
                onClick={onClose}
                className="w-8 h-8 rounded-full border border-border text-[#6b6455] text-sm flex items-center justify-center hover:border-gold hover:text-gold transition-all duration-200"
              >
                ✕
              </button>

              {totalResults > 1 && onNavigate && (
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => onNavigate("prev")}
                    disabled={!hasPrev}
                    className={clsx(
                      "w-7 h-7 rounded-full border flex items-center justify-center text-[0.75rem] transition-all duration-150",
                      hasPrev
                        ? "border-border text-cream-dim hover:border-gold hover:text-gold"
                        : "border-border/30 text-[#6b6455]/30 cursor-not-allowed"
                    )}
                    title="Previous result (← arrow key)"
                  >
                    ←
                  </button>
                  <span className="text-[0.62rem] text-[#6b6455] tabular-nums select-none">
                    {currentIndex + 1} / {totalResults}
                  </span>
                  <button
                    onClick={() => onNavigate("next")}
                    disabled={!hasNext}
                    className={clsx(
                      "w-7 h-7 rounded-full border flex items-center justify-center text-[0.75rem] transition-all duration-150",
                      hasNext
                        ? "border-border text-cream-dim hover:border-gold hover:text-gold"
                        : "border-border/30 text-[#6b6455]/30 cursor-not-allowed"
                    )}
                    title="Next result (→ arrow key)"
                  >
                    →
                  </button>
                </div>
              )}
            </div>

            {/* Header */}
            <h2 className="font-display text-[1.9rem] font-semibold text-cream leading-tight mb-1">
              {r.name}
            </h2>
            <p className="text-[0.76rem] text-[#6b6455] tracking-wide mb-6">
              {r.categories.split(",").slice(0, 3).join(" · ")} &middot; {r.city}
            </p>

            {/* Match score */}
            <div className="flex items-baseline gap-2 mb-8">
              <span className="font-display text-[2.8rem] text-gold-light leading-none font-semibold">
                {r.match_score}
              </span>
              <span className="text-[0.65rem] tracking-[0.12em] uppercase text-gold-dim">
                match score
              </span>
            </div>

            {/* ── Sentiment ── */}
            <SectionLabel>Sentiment Breakdown</SectionLabel>
            <div className="flex items-center gap-5 mb-2">
              <SentimentDonut
                positive={r.sentiment_positive}
                negative={r.sentiment_negative}
                neutral={r.sentiment_neutral}
                size={100}
              />
              <div className="flex flex-col gap-2">
                <LegendItem color="#5a9e72" label="Positive" value={r.sentiment_positive} />
                <LegendItem color="#b85252" label="Negative" value={r.sentiment_negative} />
                <LegendItem color="#6b7f9e" label="Neutral" value={r.sentiment_neutral} />
              </div>
            </div>

            {/* ── Topics ── */}
            <SectionLabel>BERTopic Distribution</SectionLabel>
            {r.topic_profile.length > 0 ? (
              <div className="flex flex-col gap-2 mb-1">
                {r.topic_profile.map((t) => (
                  <div key={t.label} className="flex items-center gap-3">
                    <span className="text-[0.76rem] text-cream-dim w-28 flex-shrink-0">
                      {t.label}
                    </span>
                    <div className="flex-1 h-[6px] bg-border rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${Math.min(t.score * 100 * 3.5, 100)}%`,
                          background: "linear-gradient(to right, #9b7230, #e8b96a)",
                        }}
                      />
                    </div>
                    <span className="text-[0.68rem] text-[#6b6455] w-8 text-right tabular-nums">
                      {Math.round(t.score * 100)}%
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-[0.74rem] text-[#6b6455] mb-3">No topic data available.</p>
            )}

            {/* ── Snippets ── */}
            <SectionLabel>
              Review Snippets
              {hasQueryMatches && (
                <span className="text-[0.54rem] px-1.5 py-0.5 rounded-full bg-gold/10 border border-gold/20 text-gold-light tracking-normal normal-case">
                  matched first
                </span>
              )}
            </SectionLabel>
            <div className="flex flex-col gap-2">
              {sortedSnippets.map((s, i) => {
                const isMatch = snippetMatchesQuery(s, qToks);
                return (
                  <div
                    key={i}
                    className={clsx(
                      "border-l-2 pl-3 pr-3 py-2.5 rounded-r-sm",
                      isMatch
                        ? "bg-gold/[0.04] border-gold-dim"
                        : "bg-white/[0.025] border-border-light"
                    )}
                  >
                    <p className="text-[0.79rem] italic text-cream-dim leading-relaxed">
                      &ldquo;
                      <HighlightedText text={s.text} toks={isMatch ? qToks : []} />
                      &rdquo;
                    </p>
                    <span
                      className={clsx(
                        "inline-block mt-1.5 text-[0.58rem] tracking-[0.08em] uppercase px-2 py-0.5 rounded-full font-medium",
                        SENT_COLOR[s.sentiment_label]
                      )}
                    >
                      {SENT_LABEL[s.sentiment_label]}
                    </span>
                  </div>
                );
              })}
            </div>

            {/* ── Model info ── */}
            <SectionLabel>Model Details</SectionLabel>
            <p className="text-[0.76rem] text-[#6b6455] leading-relaxed">
              Sentiment via{" "}
              <strong className="text-cream-dim">
                {modelUsed === "distilbert"
                  ? "fine-tuned DistilBERT (91% acc, F1 0.90)"
                  : "VADER lexicon"}
              </strong>
              . Topics via{" "}
              <strong className="text-cream-dim">BERTopic</strong> with UMAP +
              HDBSCAN. Match = 30% topic + 30% review keywords + 20% lexical + 12% sentiment + 8% stars.
            </p>

            {/* Stats */}
            <div className="mt-4 flex gap-4">
              <StatPill label="Reviews" value={String(r.n_reviews)} />
              <StatPill label="Stars" value={r.stars_business.toFixed(1)} />
            </div>
          </div>
        )}
      </div>
    </>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <p className="flex items-center gap-2 text-[0.6rem] tracking-[0.16em] uppercase text-gold mt-6 mb-3">
      {children}
    </p>
  );
}

function LegendItem({
  color,
  label,
  value,
}: {
  color: string;
  label: string;
  value: number;
}) {
  return (
    <div className="flex items-center gap-2 text-[0.76rem] text-cream-dim">
      <span
        className="w-2 h-2 rounded-full flex-shrink-0"
        style={{ background: color }}
      />
      {label} · {Math.round(value * 100)}%
    </div>
  );
}

function StatPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="border border-border px-3 py-1.5 rounded-sm text-center min-w-[72px]">
      <span className="font-display text-[1.2rem] text-gold-light block leading-none">
        {value}
      </span>
      <span className="text-[0.58rem] tracking-wider uppercase text-[#6b6455]">
        {label}
      </span>
    </div>
  );
}
