const defaultTheme = require("tailwindcss/defaultTheme");

/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "Satoshi", "system-ui", ...defaultTheme.fontFamily.sans],
      },
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        surface: "var(--surface)",
        card: "var(--card)",
        muted: "var(--muted)",
        border: "var(--border)",
        accent: "var(--accent)",
        bubbleAi: "var(--bubble-ai)",
        bubbleUser: "var(--bubble-user)",
        quote: "var(--quote-border)",
        input: "var(--input)",
        skeleton: "var(--skeleton-base)",
        skeletonHighlight: "var(--skeleton-highlight)",
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
};
