/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Primary Brand Colors - Anandhaas Theme
        primary: {
          50: '#1b2623ffff',
          100: '#CCEFE7',
          200: '#99DFCF',
          300: '#66CFB7',
          400: '#33BF9F',
          500: '#009972', // Main Anandhaas green
          600: '#007A5B',
          700: '#005B44',
          800: '#003D2E',
          900: '#001E17'
        },
        secondary: {
          50: '#FDF6F5',
          100: '#FBEDEB',
          200: '#F7DBD7',
          300: '#F3C9C3',
          400: '#EFB7AF',
          500: '#ED6D5F', // Anandhaas coral/orange
          600: '#E94A38',
          700: '#C73E31',
          800: '#A5322A',
          900: '#832623'
        },
        // Legacy colors updated to Anandhaas theme
        gold: {
          DEFAULT: '#009972',
        },
        cream: {
          DEFAULT: '#FFFFFF',
        },
        green: {
          accent: '#009972',
        },
        anandhaasGreen: '#009972',
        anandhaasOrange: '#ED6D5F'
      }
    },
  },
  plugins: [],
}