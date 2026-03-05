"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { getEDA } from "@/lib/api";
import type { EDAResponse } from "@/lib/api";

// ── Color palette for topics ──
const TOPIC_COLORS = [
  "#c8973e", "#e8b96a", "#5a9e72", "#b85252", "#6b7f9e",
  "#d4a056", "#7ab88a", "#c76b6b", "#8899b3", "#a68832",
  "#4b8860", "#9e4545", "#5a6f8e",
];

function TopicBarChart({ topics }: { topics: EDAResponse["topics"] }) {
  const maxCount = Math.max(...topics.map((t) => t.dominant_count), 1);

  return (
    <div className="flex flex-col gap-2">
      {topics.map((t, i) => (
        <div key={t.label} className="flex items-center gap-3">
          <span className="text-[0.76rem] text-cream-dim w-24 flex-shrink-0 text-right">
            {t.label}
          </span>
          <div className="flex-1 h-5 bg-border rounded-sm overflow-hidden relative">
            <div
              className="bar-fill h-full rounded-sm"
              style={{
                width: `${(t.dominant_count / maxCount) * 100}%`,
                background: TOPIC_COLORS[i % TOPIC_COLORS.length],
                opacity: 0.85,
              }}
            />
          </div>
          <span className="text-[0.7rem] text-[#6b6455] w-16 text-right tabular-nums">
            {t.dominant_count} ({t.dominant_pct}%)
          </span>
        </div>
      ))}
    </div>
  );
}

function TopicAvgScoreChart({ topics }: { topics: EDAResponse["topics"] }) {
  const maxAvg = Math.max(...topics.map((t) => t.avg_score), 0.01);

  return (
    <div className="flex flex-col gap-2">
      {topics.map((t, i) => (
        <div key={t.label} className="flex items-center gap-3">
          <span className="text-[0.76rem] text-cream-dim w-24 flex-shrink-0 text-right">
            {t.label}
          </span>
          <div className="flex-1 h-5 bg-border rounded-sm overflow-hidden">
            <div
              className="bar-fill h-full rounded-sm"
              style={{
                width: `${(t.avg_score / maxAvg) * 100}%`,
                background: TOPIC_COLORS[i % TOPIC_COLORS.length],
                opacity: 0.75,
              }}
            />
          </div>
          <span className="text-[0.7rem] text-[#6b6455] w-14 text-right tabular-nums">
            {(t.avg_score * 100).toFixed(1)}%
          </span>
        </div>
      ))}
    </div>
  );
}

function SentimentDonutEDA({
  buckets,
}: {
  buckets: Record<string, number>;
}) {
  const total = Object.values(buckets).reduce((a, b) => a + b, 0) || 1;
  const items = [
    { key: "positive", color: "#5a9e72", label: "Positive" },
    { key: "negative", color: "#b85252", label: "Negative" },
    { key: "neutral", color: "#6b7f9e", label: "Neutral" },
    { key: "mixed", color: "#9b7230", label: "Mixed" },
  ];

  // SVG donut
  const size = 140;
  const stroke = 20;
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  let offset = 0;

  return (
    <div className="flex items-center gap-6">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {items.map((item) => {
          const pct = (buckets[item.key] || 0) / total;
          const dash = pct * circumference;
          const gap = circumference - dash;
          const el = (
            <circle
              key={item.key}
              cx={size / 2}
              cy={size / 2}
              r={radius}
              fill="none"
              stroke={item.color}
              strokeWidth={stroke}
              strokeDasharray={`${dash} ${gap}`}
              strokeDashoffset={-offset}
              transform={`rotate(-90 ${size / 2} ${size / 2})`}
              opacity={0.85}
            />
          );
          offset += dash;
          return el;
        })}
        <text
          x="50%"
          y="50%"
          textAnchor="middle"
          dominantBaseline="central"
          className="fill-cream font-display text-[1.4rem] font-semibold"
        >
          {total}
        </text>
      </svg>
      <div className="flex flex-col gap-2">
        {items.map((item) => (
          <div key={item.key} className="flex items-center gap-2 text-[0.76rem] text-cream-dim">
            <span
              className="w-2.5 h-2.5 rounded-full flex-shrink-0"
              style={{ background: item.color }}
            />
            {item.label}: {buckets[item.key] || 0} ({((buckets[item.key] || 0) / total * 100).toFixed(0)}%)
          </div>
        ))}
      </div>
    </div>
  );
}

function CityChart({ cities }: { cities: EDAResponse["city_distribution"] }) {
  const maxCount = Math.max(...cities.map((c) => c.count), 1);

  return (
    <div className="flex flex-col gap-2">
      {cities.slice(0, 10).map((c) => (
        <div key={c.city} className="flex items-center gap-3">
          <span className="text-[0.76rem] text-cream-dim w-28 flex-shrink-0 text-right">
            {c.city}
          </span>
          <div className="flex-1 h-5 bg-border rounded-sm overflow-hidden relative">
            {/* Stacked: positive (green), negative (red) */}
            <div className="h-full flex">
              <div
                className="h-full"
                style={{
                  width: `${c.avg_positive * 100}%`,
                  background: "#5a9e72",
                  opacity: 0.8,
                }}
              />
              <div
                className="h-full"
                style={{
                  width: `${c.avg_negative * 100}%`,
                  background: "#b85252",
                  opacity: 0.8,
                }}
              />
            </div>
          </div>
          <span className="text-[0.7rem] text-[#6b6455] w-20 text-right tabular-nums">
            {c.count} restaurants
          </span>
        </div>
      ))}
    </div>
  );
}

function StarsChart({ distribution }: { distribution: Record<string, number> }) {
  const entries = Object.entries(distribution).sort(
    (a, b) => parseFloat(a[0]) - parseFloat(b[0])
  );
  const maxCount = Math.max(...entries.map(([, v]) => v), 1);

  return (
    <div className="flex items-end gap-2 h-32">
      {entries.map(([stars, count]) => (
        <div key={stars} className="flex flex-col items-center gap-1 flex-1">
          <div
            className="w-full rounded-t-sm"
            style={{
              height: `${(count / maxCount) * 100}%`,
              background: "linear-gradient(to top, #9b7230, #e8b96a)",
              opacity: 0.8,
              minHeight: count > 0 ? "4px" : "0",
            }}
          />
          <span className="text-[0.6rem] text-[#6b6455] tabular-nums">{stars}</span>
          <span className="text-[0.55rem] text-cream-dim tabular-nums">{count}</span>
        </div>
      ))}
    </div>
  );
}

function RestaurantTable({ restaurants }: { restaurants: EDAResponse["restaurants"] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[0.74rem]">
        <thead>
          <tr className="border-b border-border text-[0.62rem] tracking-[0.12em] uppercase text-gold">
            <th className="text-left py-2 px-2">Restaurant</th>
            <th className="text-left py-2 px-2">City</th>
            <th className="text-right py-2 px-2">Stars</th>
            <th className="text-right py-2 px-2">Reviews</th>
            <th className="text-right py-2 px-2">Pos%</th>
            <th className="text-right py-2 px-2">Neg%</th>
            <th className="text-left py-2 px-2">Top Topic</th>
          </tr>
        </thead>
        <tbody>
          {restaurants.slice(0, 30).map((r) => (
            <tr
              key={r.business_id}
              className="border-b border-border/50 hover:bg-surface-hover transition-colors"
            >
              <td className="py-2 px-2 text-cream">{r.name}</td>
              <td className="py-2 px-2 text-cream-dim">{r.city}</td>
              <td className="py-2 px-2 text-right text-gold-light tabular-nums">
                {r.stars.toFixed(1)}
              </td>
              <td className="py-2 px-2 text-right text-cream-dim tabular-nums">{r.n_reviews}</td>
              <td className="py-2 px-2 text-right text-sentiment-pos tabular-nums">
                {(r.sentiment_positive * 100).toFixed(0)}%
              </td>
              <td className="py-2 px-2 text-right text-sentiment-neg tabular-nums">
                {(r.sentiment_negative * 100).toFixed(0)}%
              </td>
              <td className="py-2 px-2">
                <span className="text-[0.64rem] px-2 py-0.5 rounded-full border border-gold-dim text-gold-light bg-[rgba(200,151,62,0.07)]">
                  {r.top_topic}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SectionCard({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="bg-surface-card border border-border rounded-sm p-6">
      <h3 className="font-display text-[1.2rem] font-semibold text-cream mb-1">{title}</h3>
      {subtitle && (
        <p className="text-[0.68rem] text-[#6b6455] mb-4">{subtitle}</p>
      )}
      {children}
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-surface-card border border-border rounded-sm px-5 py-4 text-center">
      <span className="font-display text-[2rem] text-gold-light block leading-none mb-1">
        {value}
      </span>
      <span className="text-[0.62rem] tracking-[0.12em] uppercase text-[#6b6455]">{label}</span>
    </div>
  );
}

export default function EDAPage() {
  const [data, setData] = useState<EDAResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getEDA()
      .then(setData)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="inline-block w-6 h-6 border-2 border-gold border-t-transparent rounded-full animate-spin mb-3" />
          <p className="text-[0.8rem] text-[#6b6455]">Loading EDA data...</p>
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="min-h-screen flex items-center justify-center px-6">
        <div className="text-center">
          <p className="text-sentiment-neg text-[0.85rem] mb-2">Failed to load EDA data</p>
          <p className="text-[0.75rem] text-[#6b6455]">{error}</p>
          <Link
            href="/"
            className="inline-block mt-4 text-[0.72rem] px-4 py-2 border border-border rounded-sm text-gold hover:border-gold-dim transition-colors"
          >
            Back to Search
          </Link>
        </div>
      </div>
    );
  }

  const totalSentiment = Object.values(data.sentiment_buckets).reduce((a, b) => a + b, 0);

  return (
    <main className="min-h-screen">
      {/* Header */}
      <header className="sticky top-0 z-50 flex items-center justify-between px-6 py-3 border-b border-border bg-[rgba(10,9,8,0.94)] backdrop-blur-xl">
        <Link href="/" className="flex-shrink-0">
          <span className="font-display tracking-wide text-gold text-xl">
            Taste<em className="not-italic text-cream">wise</em>
          </span>
        </Link>
        <span className="text-[0.62rem] tracking-[0.14em] uppercase text-gold">
          Exploratory Data Analysis
        </span>
        <Link
          href="/"
          className="text-[0.72rem] px-3 py-1.5 border border-border rounded-sm text-[#6b6455] hover:border-gold-dim hover:text-gold transition-all"
        >
          Back to Search
        </Link>
      </header>

      {/* Hero */}
      <div className="px-6 pt-10 pb-6 text-center">
        <h1 className="font-display text-[clamp(1.8rem,4vw,3rem)] font-semibold text-cream mb-2">
          Dataset <em className="text-gold-light">Exploration</em>
        </h1>
        <p className="text-[0.82rem] text-[#6b6455] max-w-lg mx-auto">
          Distribution analysis of BERTopic clusters, sentiment patterns, and restaurant
          characteristics across {data.n_restaurants} Arizona Yelp restaurants.
        </p>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 px-6 mb-8">
        <StatCard label="Restaurants" value={String(data.n_restaurants)} />
        <StatCard label="Topic Clusters" value={String(data.n_topics)} />
        <StatCard
          label="Avg Positive"
          value={`${(
            data.sentiment_scatter.reduce((s, r) => s + r.positive, 0) /
            data.n_restaurants *
            100
          ).toFixed(0)}%`}
        />
        <StatCard
          label="Cities"
          value={String(data.city_distribution.length)}
        />
      </div>

      {/* Charts */}
      <div className="px-6 space-y-6 pb-12">
        {/* Row 1: Topic Distribution + Sentiment */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <SectionCard
            title="Topic Dominance"
            subtitle="Number of restaurants where each topic has the highest score"
          >
            <TopicBarChart topics={data.topics} />
          </SectionCard>

          <SectionCard
            title="Sentiment Distribution"
            subtitle={`Classification of ${totalSentiment} restaurants by sentiment profile`}
          >
            <SentimentDonutEDA buckets={data.sentiment_buckets} />
          </SectionCard>
        </div>

        {/* Row 2: Avg Topic Score + Stars */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <SectionCard
            title="Average Topic Score"
            subtitle="Mean topic probability across all restaurants"
          >
            <TopicAvgScoreChart
              topics={[...data.topics].sort((a, b) => b.avg_score - a.avg_score)}
            />
          </SectionCard>

          <SectionCard
            title="Star Rating Distribution"
            subtitle="Distribution of average star ratings"
          >
            <StarsChart distribution={data.stars_distribution} />
            <div className="mt-2 text-center text-[0.65rem] text-[#6b6455]">
              Average Star Rating
            </div>
          </SectionCard>
        </div>

        {/* Row 3: City Distribution */}
        <SectionCard
          title="City Distribution"
          subtitle="Restaurant count and sentiment by city (green = positive, red = negative)"
        >
          <CityChart cities={data.city_distribution} />
        </SectionCard>

        {/* Row 4: Restaurant Table */}
        <SectionCard
          title="Restaurant Overview"
          subtitle="Top 30 restaurants sorted by positive sentiment"
        >
          <RestaurantTable restaurants={data.restaurants} />
        </SectionCard>
      </div>
    </main>
  );
}
