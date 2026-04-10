module.exports = {
  darkMode: 'class',
  content: [
    __dirname.replace(/\\/g, '/') + '/app/**/*.{js,ts,jsx,tsx,mdx}',
    __dirname.replace(/\\/g, '/') + '/components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {},
  },
  plugins: [],
};
