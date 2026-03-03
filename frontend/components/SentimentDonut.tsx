"use client";

interface Props {
  positive: number; // 0-1
  negative: number;
  neutral: number;
  size?: number;
}

export default function SentimentDonut({
  positive,
  negative,
  neutral,
  size = 100,
}: Props) {
  const cx = size / 2;
  const cy = size / 2;
  const R = size * 0.4;

  const slices = [
    { value: positive, color: "#5a9e72" },
    { value: negative, color: "#b85252" },
    { value: neutral, color: "#6b7f9e" },
  ];

  let cumul = -90;
  const paths = slices.map((s) => {
    const deg = s.value * 360;
    const startRad = (cumul * Math.PI) / 180;
    const endRad = ((cumul + deg) * Math.PI) / 180;
    const x1 = cx + R * Math.cos(startRad);
    const y1 = cy + R * Math.sin(startRad);
    const x2 = cx + R * Math.cos(endRad);
    const y2 = cy + R * Math.sin(endRad);
    const large = deg > 180 ? 1 : 0;
    const path = `M${cx},${cy} L${x1},${y1} A${R},${R} 0 ${large},1 ${x2},${y2} Z`;
    cumul += deg;
    return { path, color: s.color };
  });

  const innerR = R * 0.55;

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      {paths.map((p, i) => (
        <path key={i} d={p.path} fill={p.color} opacity="0.82" />
      ))}
      {/* Inner hole */}
      <circle cx={cx} cy={cy} r={innerR} fill="#111009" />
      {/* Center text */}
      <text
        x={cx}
        y={cy + 5}
        textAnchor="middle"
        fontFamily="Cormorant Garamond, serif"
        fontSize={size * 0.14}
        fill="#e8b96a"
        fontWeight="600"
      >
        {Math.round(positive * 100)}%
      </text>
    </svg>
  );
}
