# Urban Change Detection

A minimal full-stack app for detecting urban/built-up change between two
satellite images. Upload an earlier and a later image of the same area,
and it returns a change map, NDVI/NDWI vegetation & water index maps,
growth statistics, a projected-growth chart, and a downloadable PDF report.

## Project structure

```
backend/
  model.py        # SatelliteChangeNet architecture
  inference.py     # image loading, QA masking, model inference
  report.py         # output images + PDF report generation
  db.py              # SQLite history store (SQLAlchemy)
  main.py            # FastAPI app (/analyze, /history endpoints)
  requirements.txt
  checkpoint_final.pth   # <- you add this (trained model weights, not included)
  history.db              # <- auto-created SQLite DB on first run

frontend/
  src/
    App.jsx        # upload form + results UI
    main.jsx
    index.css
  vite.config.js     # dev proxy config (backend URL lives here)
  package.json

README.md
```

## Requirements

- Python 3.10+
- Node.js 18+
- A trained model checkpoint (`checkpoint_final.pth`, with an embedded
  `model_config`, or a `config.json` next to it) — not included in this repo

## Setup & run (local development)

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
```

Place your trained checkpoint at `backend/checkpoint_final.pth`.

```bash
uvicorn main:app --reload --port 8000
```

Check it's up: `http://localhost:8000/health` should return
`{"status": "ok", "checkpoint_found": true}`.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. In dev, Vite proxies `/api` and `/files`
requests to `http://localhost:8000` (see `frontend/vite.config.js`).

## How it works

1. You upload an earlier image, a later image, and the year of each.
2. The backend runs `SatelliteChangeNet` on the pair and produces:
   - a change probability heatmap and binary change bitmap
   - the change bitmap overlaid on the newer image
   - NDVI (vegetation) maps for both years, their change map, and an overlay
   - NDWI (water) maps for both years, their change map, and an overlay
   - growth statistics: % of area changed, estimated annual growth rate,
     and a simple linear growth projection chart
   - classical image-comparison metrics: MSE, PSNR, SSIM (+ a per-pixel
     SSIM map), and a color-histogram distance flag for illumination/
     season/sensor mismatches between the two images
3. All of the above is bundled into a downloadable PDF report
   (`report.pdf`), alongside inline results in the UI.
4. Every run is saved to a local SQLite database (`backend/history.db`).
   The frontend's **History** tab lists past runs (with a thumbnail,
   years, and % changed) — click one to reload its full results, or
   delete it.

## Deploying

This is compute-heavier than a typical CRUD app (PyTorch inference +
matplotlib/PDF generation per request), so a free-tier PaaS like Render's
free plan is likely to be too slow/memory-limited for real use. Better
fits:

- **Backend**: Railway, Fly.io, a paid Render instance, or Hugging Face
  Spaces (Docker SDK) if you want a free GPU-capable option.
- **Frontend**: Vercel or Netlify (static Vite build).

To point the frontend at a deployed backend, update the `target` URLs in
`frontend/vite.config.js` for both the `/api` and `/files` proxies (or, for
a production build, replace the proxy with a build-time `VITE_API_URL` env
var and use it directly in `App.jsx`).

## Notes / next steps

- Intentionally minimal right now (single-page frontend, no auth or
  database) — meant as a base to iterate the UI/UX on next.
- `report.py` builds the PDF with `reportlab`; swap in a different
  template/library if you want a different look.
- The growth-rate projection is a simple linear extrapolation from a
  single observed change fraction between two dates — swap in a real
  time-series model if you start comparing more than 2 dates.
