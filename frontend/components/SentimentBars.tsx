"use client";

import { useEffect, useRef, useState } from "react";
import clsx from "clsx";

interface Bar {
  label: string;
  value: number; // 0-1
  color: string;
}

interface Props {
  positive: number;
  negative: number;
  neutral: number;
  compact?: boolean;
}

export default function SentimentBars({
  positive,
  negative,
  neutral,
  compact = false,
}: Props) {
  const [mounted, setMounted] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Animate bars in when they enter the viewport
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setMounted(true);
          observer.disconnect();
        }
      },
      { threshold: 0.1 }
    );
    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);

  const bars: Bar[] = [
    { label: "Pos", value: positive, color: "#5a9e72" },
    { label: "Neg", value: negative, color: "#b85252" },
    { label: "Neu", value: neutral, color: "#6b7f9e" },
  ];

  return (
    <div ref={ref} className="flex flex-col gap-1.5">
      {bars.map((b) => (
        <div key={b.label} className="flex items-center gap-2">
          <span
            className={clsx(
              "text-[#6b6455] uppercase tracking-wider flex-shrink-0",
              compact ? "text-[0.6rem] w-6" : "text-[0.65rem] w-7"
            )}
          >
            {b.label}
          </span>
          <div className="flex-1 h-[3px] bg-border rounded-full overflow-hidden">
            <div
              className="bar-fill h-full rounded-full"
              style={{
                width: mounted ? `${b.value * 100}%` : "0%",
                background: b.color,
              }}
            />
          </div>
          <span className="text-[0.65rem] text-[#6b6455] w-7 text-right tabular-nums">
            {Math.round(b.value * 100)}%
          </span>
        </div>
      ))}
    </div>
  );
}
