"use client";

import { useEffect, useState } from "react";
import clsx from "clsx";

const PHRASES = [
  "Analyzing 3,900 reviews…",
  "Running sentiment models…",
  "Mapping topic clusters…",
  "Ranking restaurants…",
];

const DURATION = 2800; // total loading time in ms

interface Props {
  children: React.ReactNode;
}

export default function LoadingScreen({ children }: Props) {
  const [mounted, setMounted] = useState(false);
  const [skip, setSkip] = useState(false);
  const [progress, setProgress] = useState(0);
  const [phraseIdx, setPhraseIdx] = useState(0);
  const [done, setDone] = useState(false);
  const [exiting, setExiting] = useState(false);

  // Only show on first visit to "/"
  useEffect(() => {
    setMounted(true);
    if (sessionStorage.getItem("tastewise-loaded")) {
      setSkip(true);
    } else {
      sessionStorage.setItem("tastewise-loaded", "true");
    }
  }, []);

  // Animate progress bar
  useEffect(() => {
    if (!mounted || skip) return;
    const t0 = performance.now();
    let raf: number;
    const tick = (now: number) => {
      const pct = Math.min(((now - t0) / DURATION) * 100, 100);
      setProgress(pct);
      if (pct < 100) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [mounted, skip]);

  // Cycle through phrases
  useEffect(() => {
    if (!mounted || skip) return;
    const interval = DURATION / PHRASES.length;
    const id = setInterval(() => {
      setPhraseIdx((i) => Math.min(i + 1, PHRASES.length - 1));
    }, interval);
    return () => clearInterval(id);
  }, [mounted, skip]);

  // Exit sequence
  useEffect(() => {
    if (!mounted || skip) return;
    const t1 = setTimeout(() => setExiting(true), DURATION);
    const t2 = setTimeout(() => setDone(true), DURATION + 800);
    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
    };
  }, [mounted, skip]);

  if (!mounted) return null;

  return (
    <div className="relative">
      {/* Loader overlay */}
      {!skip && !done && (
        <div
          className={clsx(
            "fixed inset-0 z-50 flex flex-col items-center justify-center bg-[#0a0908]",
            "transition-opacity duration-700",
            exiting ? "opacity-0" : "opacity-100"
          )}
        >
          {/* Progress bar */}
          <div className="absolute top-0 left-0 w-full h-[2px] bg-border">
            <div
              className="h-full bg-gold transition-[width] duration-100 ease-linear"
              style={{ width: `${progress}%` }}
            />
          </div>

          {/* Brand mark */}
          <div className="mb-10 text-center">
            <span className="font-display text-4xl tracking-wide text-gold">
              Taste<em className="not-italic text-cream">wise</em>
            </span>
            <p className="mt-2 text-[0.62rem] tracking-[0.18em] uppercase text-[#6b6455]">
              Yelp Review Intelligence · Team&nbsp;18 · CIS&nbsp;509
            </p>
          </div>

          {/* Animated dots */}
          <div className="flex gap-2 mb-6">
            {[0, 1, 2].map((i) => (
              <span
                key={i}
                className="block w-1.5 h-1.5 rounded-full bg-gold animate-pulse"
                style={{ animationDelay: `${i * 200}ms` }}
              />
            ))}
          </div>

          {/* Rotating phrase */}
          <p className="text-[0.82rem] text-cream-dim font-body tracking-wide h-5">
            {PHRASES[phraseIdx]}
          </p>

          {/* Percentage */}
          <p className="mt-3 text-[0.68rem] text-[#6b6455] tabular-nums">
            {Math.floor(progress)}%
          </p>
        </div>
      )}

      {/* Main content */}
      <div
        className={clsx(
          "transition-opacity duration-700",
          done || skip ? "opacity-100" : "opacity-0 pointer-events-none"
        )}
      >
        {children}
      </div>
    </div>
  );
}
