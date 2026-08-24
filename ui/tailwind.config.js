/** @type {import('tailwindcss').Config} */
// Design tokens from rasa-product-spec.md §4.1: "calm dusk, not midnight —
// warm charcoal / deep plum backgrounds rather than pure black." Fonts fall
// back to system stacks until real font files are vendored locally — this
// stays a fully offline, local-first app, so no remote font CDN.
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        dusk: '#2b2233',
        charcoal: '#241c2b',
        surface: '#372c42',
        ink: '#f3ede9',
        'ink-soft': '#b8aec2',
        gold: '#c9a35c',
      },
      fontFamily: {
        display: ['Fraunces', 'Source Serif 4', 'Georgia', 'serif'],
        body: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      },
      borderRadius: {
        card: '14px',
      },
    },
  },
  plugins: [],
};
