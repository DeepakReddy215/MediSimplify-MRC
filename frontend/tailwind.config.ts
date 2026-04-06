import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        background: "#060b16",
        card: "#111a2b",
        accent: "#3b82f6",
        mint: "#10b981"
      },
      boxShadow: {
        soft: "0 10px 40px rgba(59, 130, 246, 0.15)",
        card: "0 12px 28px rgba(15, 23, 42, 0.35)"
      },
      backgroundImage: {
        "hero-grid": "radial-gradient(circle at 10% 20%, rgba(59,130,246,0.24), transparent 30%), radial-gradient(circle at 90% 30%, rgba(16,185,129,0.18), transparent 30%)"
      }
    }
  },
  plugins: []
};

export default config;
