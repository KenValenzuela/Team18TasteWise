"use client";

import { useState, useEffect, useRef } from "react";
import clsx from "clsx";

interface Citation {
  restaurant: string;
  snippet: string;
}

interface Props {
  /** The full LLM-generated response text */
  text: string;
  /** Optional inline citations the agent extracted */
  citations?: Citation[];
  /** Whether the response is still streaming in */
  streaming?: boolean;
  /** Controls mount animation */
  visible: boolean;
}

export default function AgentResponse({
  text,
  citations = [],
  streaming = false,
  visible,
}: Props) {
  const [expanded, setExpanded] = useState(false);
  const contentRef = useRef<HTMLDivElement>(null);
  const [needsTruncation, setNeedsTruncation] = useState(false);

  // Check if text is long enough to warrant a "show more" toggle
  useEffect(() => {
    if (contentRef.current) {
      setNeedsTruncation(contentRef.current.scrollHeight > 160);
    }
  }, [text]);

  if (!text) return null;

  return (
    <div
      className={clsx(
        "mx-6 mt-4 mb-2 transition-all duration-500 ease-out",
        visible
          ? "opacity-100 translate-y-0"
          : "opacity-0 translate-y-3 pointer-events-none"
      )}
    >
      {/* Outer container */}
      <div className="relative border border-border bg-[rgba(17,16,9,0.6)] rounded-sm overflow-hidden">
        {/* Gold left accent */}
        <div className="absolute left-0 top-0 bottom-0 w-[2px] bg-gradient-to-b from-gold via-gold-dim to-transparent" />

        <div className="pl-5 pr-5 py-4">
          {/* Label */}
          <div className="flex items-center gap-2 mb-3">
            <span className="text-[0.58rem] tracking-[0.14em] uppercase text-gold">
              ✦ Agent Summary
            </span>
            {streaming && (
              <span className="flex items-center gap-1.5 text-[0.56rem] tracking-wider uppercase text-gold-dim">
                <span className="inline-block w-1.5 h-1.5 rounded-full bg-gold animate-pulse" />
                Generating
              </span>
            )}
          </div>

          {/* Response text */}
          <div
            ref={contentRef}
            className={clsx(
              "text-[0.82rem] text-cream-dim leading-[1.7] font-body font-light transition-all duration-300",
              !expanded && needsTruncation && "max-h-[160px] overflow-hidden",
              // Fade mask when truncated
              !expanded && needsTruncation && "mask-fade"
            )}
            style={
              !expanded && needsTruncation
                ? {
                    WebkitMaskImage:
                      "linear-gradient(to bottom, black 60%, transparent 100%)",
                    maskImage:
                      "linear-gradient(to bottom, black 60%, transparent 100%)",
                  }
                : undefined
            }
          >
            {text}
          </div>

          {/* Show more / less toggle */}
          {needsTruncation && (
            <button
              onClick={() => setExpanded((e) => !e)}
              className="mt-2 text-[0.7rem] tracking-wide text-gold-dim hover:text-gold transition-colors duration-200"
            >
              {expanded ? "Show less ↑" : "Read full summary ↓"}
            </button>
          )}

          {/* Citations (evidence snippets) */}
          {citations.length > 0 && (
            <div className="mt-4 pt-3 border-t border-border">
              <span className="text-[0.56rem] tracking-[0.12em] uppercase text-[#6b6455] block mb-2">
                Evidence
              </span>
              <div className="flex flex-col gap-1.5">
                {citations.map((c, i) => (
                  <div
                    key={i}
                    className="flex items-start gap-2 text-[0.72rem]"
                  >
                    <span className="text-gold-dim flex-shrink-0 mt-0.5">›</span>
                    <span className="text-[#6b6455]">
                      <strong className="text-cream-dim font-medium">
                        {c.restaurant}
                      </strong>
                      {" — "}
                      <em className="italic">&ldquo;{c.snippet}&rdquo;</em>
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}