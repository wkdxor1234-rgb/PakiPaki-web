# PakiPaki Backend (Flask API)

## Endpoints
- `GET /health` → `{ ok: true, model_loaded: bool, allowed_ext: [...] }`
- `POST /predict` (form-data, field name **file**) → `{ ok, diagnosis, confidence, features }`

## Quick Start (local)
```bash
pip install -r requirements.txt
gunicorn app:app --bind 0.0.0.0:5000 --timeout 300 --threads 2
```

### Public HTTPS (three options)

**1) ngrok (fastest)**
```bash
ngrok http 5000
# copy the https URL and use it in frontend app.js
```

**2) Render / Railway (managed)**
- Create a new web service from this folder
- Start command: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300 --threads 2`
- Add `model.pkl` to the same directory (if using a trained model)

**3) Own domain (Nginx + Let’s Encrypt)**
- Proxy to `127.0.0.1:5000`
- Increase `client_max_body_size` and allow OPTIONS for CORS
- Issue TLS with Certbot

## Config
- `FRONTEND_ORIGINS` (comma-separated): allowed origins for CORS
- `MAX_MB` (default 30)
- `ALLOWED_EXT` (default "wav"; to allow browser recordings use "wav,webm,mp4,m4a,mp3")

## Model
Place `model.pkl` next to `app.py`. If absent or prediction fails, a heuristic fallback (ZCR) is used.
