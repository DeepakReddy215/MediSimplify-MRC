# Medical Report Simplifier

Medical Report Simplifier is a full-stack app that lets users upload medical reports, run OCR, and get patient-friendly plus caregiver-focused explanations.

## Tech Stack

- Frontend: Next.js 14 (App Router), TypeScript, Tailwind CSS
- Backend: FastAPI
- Database: MongoDB (Motor + Beanie ODM)
- Auth: JWT (signup/login)
- OCR: PyMuPDF + pytesseract
- Simplification: transformer-based summarization with safe fallback logic

## Project Structure

- frontend: Next.js web app
- backend: FastAPI API app
- training: model training/evaluation scripts
- utils: shared training utilities
- data: datasets used for training/evaluation

Important backend folders:

- backend/app/core: app config, database init, security
- backend/app/models: Beanie document models
- backend/app/routers: API routes
- backend/app/services: OCR and simplification services

## Current Backend API

Auth routes:

- POST /auth/signup
- POST /auth/login

Report routes:

- POST /reports/upload
- POST /reports/simplify-text
- GET /reports
- GET /reports/{report_id}
- POST /reports/{report_id}/feedback

Utility route:

- GET /health

## Environment Variables

Backend (.env):

- SECRET_KEY (required)
- ALGORITHM (default HS256)
- ACCESS_TOKEN_EXPIRE_MINUTES (default 1440)
- MONGODB_URL or MONGODB_URI (either is accepted)
- DATABASE_NAME (default medical_report_simplifier)
- CORS_ORIGINS (comma-separated origins)
- TESSERACT_CMD (optional, needed when OCR binary is not in PATH)
- SIMPLIFIER_BASE_MODEL (default google/flan-t5-small)
- SIMPLIFIER_MODEL_PATH (optional local model path)

Frontend (.env.local):

- NEXT_PUBLIC_API_URL (default /api)
- BACKEND_ORIGIN (default http://localhost:8000)

## Local Setup

Prerequisites:

- Python 3.11
- Node.js 18+
- MongoDB running
- Tesseract OCR installed (optional if only text input is used)

Backend:

1. Open terminal in backend
2. Create and activate virtual environment
3. Install dependencies: pip install -r requirements.txt
4. Create .env from .env.example and set values
5. Run API: uvicorn app.main:app --reload

Frontend:

1. Open terminal in frontend
2. Install dependencies: npm install
3. Create .env.local from .env.local.example
4. Run app: npm run dev

By default:

- Backend runs on http://localhost:8000
- Frontend runs on http://localhost:3000

## Render Deployment Notes

Backend service:

- Root Directory: backend
- Build Command: pip install -r requirements.txt
- Start Command: uvicorn app.main:app --host 0.0.0.0 --port $PORT

Frontend service:

- Root Directory: frontend
- Build Command: npm install && npm run build
- Start Command: npm run start

Make sure backend CORS_ORIGINS includes your frontend Render URL.

## Notes

- Uploaded files are stored in backend/uploads.
- OCR for scanned image/PDF files requires Tesseract.
- If the model is unavailable, simplification falls back to a deterministic safe output path.
