import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        display: ["var(--font-cormorant)", "Georgia", "serif"],
        body: ["var(--font-syne)", "system-ui", "sans-serif"],
      },
      colors: {
        gold: {
          DEFAULT: "#c8973e",
          light: "#e8b96a",
          dim: "#9b7230",
        },
        cream: {
          DEFAULT: "#ede5d0",
          dim: "#b5ab95",
        },
        surface: {
          DEFAULT: "#111009",
          card: "#161410",
          hover: "#1d1a14",
        },
        border: {
          DEFAULT: "#272420",
          light: "#302c24",
        },
        sentiment: {
          pos: "#5a9e72",
          neg: "#b85252",
          neu: "#6b7f9e",
        },
      },
      keyframes: {
        fadeUp: {
          "0%": { opacity: "0", transform: "translateY(20px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        cardIn: {
          "0%": { opacity: "0", transform: "translateY(16px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        fillBar: {
          "0%": { width: "0%" },
          "100%": { width: "var(--bar-width)" },
        },
        slideIn: {
          "0%": { transform: "translateX(105%)" },
          "100%": { transform: "translateX(0)" },
        },
      },
      animation: {
        fadeUp: "fadeUp 0.6s ease both",
        cardIn: "cardIn 0.5s ease both",
        slideIn: "slideIn 0.45s cubic-bezier(0.16,1,0.3,1) both",
      },
    },
  },
  plugins: [],
};

export default config;
