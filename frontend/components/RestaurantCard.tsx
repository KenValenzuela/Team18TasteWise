"use client";

import clsx from "clsx";
import type { RestaurantResult } from "@/lib/api";
import SentimentBars from "./SentimentBars";

interface Props {
  restaurant: RestaurantResult;
  rank: number;
  delay?: number;
  onClick: () => void;
}

export default function RestaurantCard({
  restaurant: r,
  rank,
  delay = 0,
  onClick,
}: Props) {
  // Top 3 topics for card display
  const topTopics = r.topic_profile.slice(0, 4);

  return (
    <div
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === "Enter" && onClick()}
      className={clsx(
        "group relative bg-surface-card px-6 pt-5 pb-5 cursor-pointer",
        "border-r border-b border-border last:border-r-0",
        "transition-colors duration-200 hover:bg-surface-hover",
        "animate-cardIn outline-none"
      )}
      style={{ animationDelay: `${delay}ms`, animationFillMode: "both" }}
    >
      {/* Gold left accent on hover */}
      <div
        className={clsx(
          "absolute left-0 top-0 bottom-0 w-[2px] bg-gold",
          "scale-y-0 group-hover:scale-y-100 origin-bottom",
          "transition-transform duration-300 ease-out"
        )}
      />

      {/* Rank number (background) */}
      <span className="absolute top-4 right-4 font-display text-[2rem] font-semibold text-border-light leading-none group-hover:text-gold-dim transition-colors duration-200 select-none">
        {String(rank).padStart(2, "0")}
      </span>

      {/* Name + meta */}
      <div className="pr-12 mb-1">
        <h3 className="font-display text-[1.2rem] font-semibold text-cream leading-tight">
          {r.name}
        </h3>
        <p className="text-[0.71rem] text-[#6b6455] mt-0.5 tracking-wide">
          {r.categories.split(",").slice(0, 2).join(" · ")} &middot; {r.city}
        </p>
      </div>

      {/* Match score badge */}
      <div className="inline-flex items-baseline gap-1 border border-gold-dim bg-[rgba(200,151,62,0.07)] px-2 py-0.5 rounded-sm mb-3 mt-2">
        <span className="font-display text-[1.15rem] text-gold-light leading-none">
          {r.match_score}
        </span>
        <span className="text-[0.58rem] tracking-[0.1em] uppercase text-gold-dim">
          match
        </span>
      </div>

      {/* Agent reason */}
      {r.reason && (
        <p className="text-[0.74rem] text-[#6b6455] italic leading-relaxed mb-3">
          {r.reason}
        </p>
      )}

      {/* Sentiment bars */}
      <SentimentBars
        positive={r.sentiment_positive}
        negative={r.sentiment_negative}
        neutral={r.sentiment_neutral}
      />

      {/* Topic tags */}
      <div className="flex flex-wrap gap-1.5 mt-4">
        {topTopics.map((t) => (
          <span
            key={t.label}
            className={clsx(
              "text-[0.64rem] px-2 py-0.5 rounded-full border transition-colors duration-200",
              t.score > 0.25
                ? "border-[rgba(200,151,62,0.4)] text-gold-light bg-[rgba(200,151,62,0.07)]"
                : "border-border text-[#6b6455]"
            )}
          >
            {t.label}
          </span>
        ))}
      </div>
    </div>
  );
}
