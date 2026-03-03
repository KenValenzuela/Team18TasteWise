"use client";

import { useRef, useState, KeyboardEvent } from "react";
import clsx from "clsx";

const CHIPS = [
  "fast lunch, good value",
  "quiet date night vibes",
  "authentic flavors, easy parking",
  "best brunch in the valley",
  "attentive service, worth the price",
];

interface Props {
  collapsed: boolean;
  loading: boolean;
  onSearch: (query: string) => void;
}

export default function SearchStage({ collapsed, loading, onSearch }: Props) {
  const [query, setQuery] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const submit = () => {
    if (query.trim().length < 3) return;
    onSearch(query.trim());
  };

  const fillChip = (q: string) => {
    setQuery(q);
    inputRef.current?.focus();
    onSearch(q);
  };

  const onKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") submit();
  };

  return (
    <div
      className={clsx(
        "transition-all duration-700 ease-[cubic-bezier(0.16,1,0.3,1)]",
        collapsed
          ? // ── Sticky nav ──────────────────────────────────────────
            [
              "sticky top-0 z-50 flex flex-row items-center justify-between gap-4",
              "px-6 py-3 border-b border-border",
              "bg-[rgba(10,9,8,0.94)] backdrop-blur-xl",
            ]
          : // ── Full-screen hero ─────────────────────────────────────
            [
              "min-h-screen flex flex-col items-center justify-center",
              "px-6 pb-16",
            ]
      )}
    >
      {/* ── Brand ── */}
      <div className={clsx("flex-shrink-0", !collapsed && "mb-10 text-center")}>
        <span
          className={clsx(
            "font-display tracking-wide text-gold transition-all duration-500",
            collapsed ? "text-xl" : "text-3xl"
          )}
        >
          Taste
          <em className="not-italic text-cream">wise</em>
        </span>
        {!collapsed && (
          <p className="mt-1 text-[0.62rem] tracking-[0.18em] uppercase text-[#6b6455]">
            Yelp Review Intelligence · Team 18 · CIS&nbsp;509
          </p>
        )}
      </div>

      {/* ── Hero text (only when expanded) ── */}
      {!collapsed && (
        <div className="text-center mb-8 animate-fadeUp">
          <h1 className="font-display text-[clamp(2.6rem,6vw,4.8rem)] font-semibold leading-[1.06] text-cream mb-3">
            Find restaurants by
            <br />
            <em className="text-gold-light">what people actually say</em>
          </h1>
          <p className="text-[0.88rem] text-[#6b6455] tracking-wide leading-relaxed max-w-md mx-auto">
            Fine-tuned DistilBERT sentiment&nbsp;·&nbsp;BERTopic modeling
            <br />
            on 3,900 Arizona Yelp reviews
          </p>
        </div>
      )}

      {/* ── Search bar ── */}
      <div
        className={clsx(
          "flex items-stretch overflow-hidden border border-border-light bg-surface-card",
          "transition-all duration-500 focus-within:border-gold-dim",
          "focus-within:shadow-[0_0_0_3px_rgba(200,151,62,0.1)]",
          collapsed ? "flex-1 max-w-xl rounded-sm" : "w-full max-w-[580px] rounded-sm"
        )}
      >
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={onKey}
          placeholder={
            collapsed
              ? "Search restaurants…"
              : 'e.g. "fast lunch, good value"'
          }
          className={clsx(
            "flex-1 bg-transparent outline-none px-4 font-body font-light",
            "text-cream placeholder:text-[#6b6455] text-[0.88rem] tracking-wide",
            collapsed ? "py-2.5" : "py-3.5"
          )}
        />
        <button
          onClick={submit}
          disabled={loading}
          className={clsx(
            "px-5 bg-gold text-[#0a0908] font-body font-medium text-[0.75rem]",
            "tracking-[0.1em] uppercase transition-colors duration-200",
            "hover:bg-gold-light disabled:opacity-50 disabled:cursor-not-allowed",
            "flex items-center gap-2 whitespace-nowrap"
          )}
        >
          {loading ? (
            <>
              <span className="inline-block w-3 h-3 border border-[#0a0908] border-t-transparent rounded-full animate-spin" />
              Analyzing
            </>
          ) : (
            "Find →"
          )}
        </button>
      </div>

      {/* ── Query chips (only when expanded) ── */}
      {!collapsed && (
        <div className="flex flex-wrap gap-2 justify-center mt-4 animate-fadeUp">
          {CHIPS.map((c) => (
            <button
              key={c}
              onClick={() => fillChip(c)}
              className={clsx(
                "text-[0.72rem] px-3 py-1 rounded-full border border-border",
                "text-[#6b6455] font-body hover:border-gold-dim hover:text-gold",
                "hover:bg-[rgba(200,151,62,0.06)] transition-all duration-200"
              )}
            >
              {c}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
