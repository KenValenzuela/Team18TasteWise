"use client";

import type { HealthResponse } from "@/lib/api";

const PIPELINE_STEPS = [
  { icon: "📥", title: "Data Ingestion", desc: "Yelp JSON filtered to AZ food businesses" },
  { icon: "🔧", title: "Preprocessing", desc: "Tokenize · lemmatize · remove stop words" },
  { icon: "💬", title: "Sentiment", desc: "VADER → Naive Bayes → fine-tuned DistilBERT" },
  { icon: "🗂", title: "BERTopic", desc: "UMAP + HDBSCAN → 8 topic clusters" },
  { icon: "🎯", title: "Ranking", desc: "Sentiment × topic similarity → match score" },
];

const BIASES = [
  {
    title: "Self-Selection Bias",
    text: "Unhappy diners write more reviews. Model calibrates via star-rating weak labels rather than raw sentiment prevalence.",
  },
  {
    title: "Geographic Scope",
    text: "Arizona only. Recommendations may not generalize to other regions with different dining cultures or price norms.",
  },
  {
    title: "Temporal Skew",
    text: "Dataset predates COVID-19. Restaurants that closed or changed significantly post-pandemic may score higher than they should.",
  },
  {
    title: "Category Imbalance",
    text: "High-volume categories (American, Mexican) dominate topic modeling. Niche cuisines may produce weaker topic clusters.",
  },
];

interface Props {
  health: HealthResponse | null;
  visible: boolean;
}

export default function InfoSection({ health, visible }: Props) {
  return (
    <section
      className={`mt-16 border-t border-border transition-all duration-700 delay-300 ${
        visible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-5"
      }`}
    >
      {/* ── Stats strip ── */}
      <div className="flex border-b border-border overflow-x-auto">
        {[
          { n: health?.n_reviews ? health.n_reviews.toLocaleString() : "—", l: "AZ reviews analyzed" },
          { n: "8", l: "BERTopic clusters" },
          { n: "91%", l: "DistilBERT accuracy" },
          { n: "0.78", l: "Avg coherence score" },
        ].map((s) => (
          <div
            key={s.l}
            className="flex-1 min-w-[130px] px-6 py-5 border-r border-border last:border-r-0"
          >
            <span className="font-display text-[2rem] font-semibold text-gold block mb-1 leading-none">
              {s.n}
            </span>
            <span className="text-[0.7rem] text-[#6b6455] tracking-wide">{s.l}</span>
          </div>
        ))}
      </div>

      {/* ── Model cards ── */}
      <div className="flex border-b border-border overflow-x-auto">
        <ModelCard
          role="Baseline"
          name="VADER Lexicon"
          desc="Rule-based scorer. Fast and interpretable. No training required. Used for initial labeling and comparison."
        />
        <ModelCard
          role="Main Classifier"
          name="Fine-tuned DistilBERT"
          metrics={[
            { v: "91%", l: "Accuracy" },
            { v: "0.90", l: "F1" },
            { v: "3 ep.", l: "Training" },
          ]}
        />
        <ModelCard
          role="Topic Modeling"
          name="BERTopic"
          desc="Sentence embeddings + UMAP dimensionality reduction + HDBSCAN clustering → 8 interpretable topic clusters."
        />
      </div>

      {/* ── Pipeline ── */}
      <div className="px-6 py-8 border-b border-border">
        <p className="text-[0.6rem] tracking-[0.16em] uppercase text-gold mb-4">
          Analytics Pipeline
        </p>
        <div className="flex gap-0 overflow-x-auto">
          {PIPELINE_STEPS.map((s, i) => (
            <div
              key={s.title}
              className="flex-1 min-w-[120px] bg-surface-card border border-border p-4 relative hover:border-gold-dim transition-colors duration-200 group"
            >
              {i < PIPELINE_STEPS.length - 1 && (
                <span className="absolute -right-[11px] top-1/2 -translate-y-1/2 text-gold-dim text-lg z-10 bg-[#0a0908] px-0.5">
                  ›
                </span>
              )}
              <span className="text-xl mb-2 block">{s.icon}</span>
              <p className="text-[0.75rem] font-medium text-cream mb-1">{s.title}</p>
              <p className="text-[0.68rem] text-[#6b6455] leading-relaxed">{s.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* ── Bias journal ── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 border-b border-border">
        {BIASES.map((b) => (
          <div
            key={b.title}
            className="bg-surface-card border-r border-border last:border-r-0 p-5"
          >
            <p className="text-[0.62rem] tracking-[0.12em] uppercase text-gold mb-2">
              {b.title}
            </p>
            <p className="text-[0.76rem] text-[#6b6455] leading-relaxed">{b.text}</p>
          </div>
        ))}
      </div>

      {/* ── Footer ── */}
      <div className="px-6 py-4 flex items-center justify-between">
        <span className="text-[0.65rem] text-[#6b6455] tracking-wide">
          Tastewise · Team 18 · CIS 509 · Spring 2026
        </span>
        <span className="text-[0.65rem] text-[#6b6455]">
          {health
            ? `${health.n_restaurants} restaurants · ${health.sentiment_model} active`
            : "Connecting to backend…"}
        </span>
      </div>
    </section>
  );
}

function ModelCard({
  role,
  name,
  desc,
  metrics,
}: {
  role: string;
  name: string;
  desc?: string;
  metrics?: { v: string; l: string }[];
}) {
  return (
    <div className="flex-1 min-w-[200px] bg-surface-card border-r border-border last:border-r-0 p-5">
      <p className="text-[0.6rem] tracking-[0.12em] uppercase text-gold mb-1">{role}</p>
      <p className="font-display text-[1rem] font-semibold text-cream mb-2">{name}</p>
      {desc && (
        <p className="text-[0.74rem] text-[#6b6455] leading-relaxed">{desc}</p>
      )}
      {metrics && (
        <div className="flex gap-4 mt-1">
          {metrics.map((m) => (
            <div key={m.l} className="text-center">
              <span className="font-display text-[1.3rem] text-gold-light block leading-none">
                {m.v}
              </span>
              <span className="text-[0.6rem] text-[#6b6455] tracking-wide">{m.l}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
