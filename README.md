# PPE Detection — Monorepo

Two-folder layout:

- `backend/` — FastAPI + YOLOv8 detection service (deploys to **Render**)
- `frontend/` — React + Vite single-page UI (deploys to **Vercel**)

---

## Local development

### Backend
```bash
cd backend
source venv/bin/activate          # or create one: python -m venv venv && pip install -r requirements/base.txt
uvicorn app.main:app --reload --reload-dir app --host 127.0.0.1 --port 8000
```
Backend will be at http://127.0.0.1:8000 (docs at `/docs`).

### Frontend
```bash
cd frontend
npm install
npm run dev
```
Open the URL Vite prints (usually http://localhost:5173). The dev server proxies `/api/*` to the backend on port 8000, so no CORS or env config needed locally.

---

## Production deployment

### Backend on Render
1. Push this repo to GitHub.
2. In Render, **New + → Blueprint** → point to repo.
3. Render reads `backend/render.yaml` and creates a Docker web service rooted at `backend/`.
4. Set the `CORS_ORIGINS` env var in the Render dashboard to your Vercel URL, e.g. `https://your-app.vercel.app`.
5. Wait for build (PyTorch makes the image large; build is slow on free tier).

### Frontend on Vercel
1. In Vercel, **Add New → Project** → import the repo.
2. Set **Root Directory** to `frontend`.
3. Add env var `VITE_API_BASE_URL` = your Render backend URL (no trailing slash), e.g. `https://ppe-detection-api.onrender.com`.
4. Deploy.

---

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Returns `{status, model_loaded}` |
| POST | `/api/v1/detect-image?format=jpeg\|json` | Upload an image; returns annotated JPEG or JSON detections |
| POST | `/api/v1/detect-frame` | Upload a webcam frame; returns JSON detections + counts |

All endpoints accept `multipart/form-data` with field name `file`.
