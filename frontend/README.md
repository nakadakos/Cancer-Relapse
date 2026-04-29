# C.A.R.E. — Frontend

React + Vite web application for the C.A.R.E. cancer relapse prediction tool.

---

## Getting Started

```bash
npm install
npm run dev
```

The app opens at **http://localhost:5173**.  
The backend must be running at **http://localhost:8000** for predictions and the data dashboard to work.

---

## Available Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start the Vite dev server with hot reload |
| `npm run build` | Build for production (output in `dist/`) |
| `npm run preview` | Preview the production build locally |
| `npm run lint` | Run ESLint |

---

## Project Structure

```
frontend/
├── public/
│   ├── favicon.svg
│   ├── icons.svg
│   └── manifest.json          # PWA manifest
├── src/
│   ├── App.jsx                # Main app — multi-step prediction form + results
│   ├── Dashboard.jsx          # Data insights charts (Recharts)
│   ├── index.css              # Global styles (dark mode, glassmorphism)
│   └── main.jsx               # React entry point
├── index.html
├── vite.config.js
└── package.json
```

---

## Tech Stack

- **React 19** — UI framework
- **Vite** — build tool and dev server
- **Recharts** — charting library for the data dashboard
- **Vanilla CSS** — custom dark-mode design system (no Tailwind)

---

## Features

- **Multi-step prediction form** — 5 steps covering Demography, Tumor, Biomarkers, Treatment, and Follow-up
- **Risk gauge** — animated SVG gauge showing relapse probability percentage
- **Data Insights dashboard** — bar charts, pie chart, and radar chart powered by real dataset statistics
- **Responsive / PWA** — bottom tab bar on mobile, installable as a home-screen app via `manifest.json`

---

## Cancer Types

Breast · Lung · Colon · Prostate · Liver · Mouth · Thyroid
