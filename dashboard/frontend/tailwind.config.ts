/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#07090D',
        surface: '#0D1117',
        surfaceElevated: '#131920',
        accentCyan: '#00E5CC',
        accentBlue: '#4D9FFF',
        accentAmber: '#F5A623',
        accentRed: '#F05050',
        accentGreen: '#4ADE80',
        textPrimary: '#CDD5E0',
        textMuted: '#58687A',
        textDisabled: '#2E3D50',
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', 'monospace'],
        display: ['Syne', 'sans-serif'],
      },
      animation: {
        'pulse-glow': 'pulse-glow 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'shimmer': 'shimmer 1.5s linear infinite',
        'flow-dot': 'flow-dot 2s linear infinite',
      },
      keyframes: {
        'pulse-glow': {
          '0%, 100%': { opacity: '1', transform: 'scale(1)' },
          '50%': { opacity: '0', transform: 'scale(1.5)' },
        },
        'shimmer': {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        'flow-dot': {
          '0%': { offset: '0%' },
          '100%': { offset: '100%' },
        },
      },
    },
  },
  plugins: [],
}
