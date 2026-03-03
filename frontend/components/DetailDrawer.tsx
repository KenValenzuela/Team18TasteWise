"use client";

import { useEffect } from "react";
import clsx from "clsx";
import type { RestaurantResult } from "@/lib/api";
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
}

export default function DetailDrawer({ restaurant: r, modelUsed, onClose }: Props) {
  const open = r !== null;

  // Close on Escape
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // Lock body scroll when open
  useEffect(() => {
    document.body.style.overflow = open ? "hidden" : "";
    return () => { document.body.style.overflow = ""; };
  }, [open]);

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
            {/* Close */}
            <button
              onClick={onClose}
              className="w-8 h-8 rounded-full border border-border text-[#6b6455] text-sm flex items-center justify-center hover:border-gold hover:text-gold transition-all duration-200 mb-6"
            >
              ✕
            </button>

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
            <div className="flex flex-col gap-2 mb-1">
              {r.topic_profile.map((t) => (
                <div key={t.label} className="flex items-center gap-3">
                  <span className="text-[0.76rem] text-cream-dim w-28 flex-shrink-0">
                    {t.label}
                  </span>
                  <div className="flex-1 h-[4px] bg-border rounded-full overflow-hidden">
                    <div
                      className="bar-fill h-full rounded-full bg-gradient-to-r from-gold-dim to-gold-light"
                      style={{ width: `${t.score * 100}%` }}
                    />
                  </div>
                  <span className="text-[0.68rem] text-[#6b6455] w-8 text-right tabular-nums">
                    {Math.round(t.score * 100)}%
                  </span>
                </div>
              ))}
            </div>

            {/* ── Snippets ── */}
            <SectionLabel>Review Snippets</SectionLabel>
            <div className="flex flex-col gap-2">
              {r.top_snippets.map((s, i) => (
                <div
                  key={i}
                  className="bg-white/[0.025] border-l-2 border-border-light pl-3 pr-3 py-2.5 rounded-r-sm"
                >
                  <p className="text-[0.79rem] italic text-cream-dim leading-relaxed">
                    &ldquo;{s.text}&rdquo;
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
              ))}
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
              HDBSCAN. Match&nbsp;= 40%&nbsp;sentiment&nbsp;+&nbsp;40%&nbsp;topic&nbsp;similarity&nbsp;+&nbsp;20%&nbsp;stars.
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
    <p className="text-[0.6rem] tracking-[0.16em] uppercase text-gold mt-6 mb-3">
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
