# Medical Report Simplifier

A clean, startup-style MVP that helps users upload medical reports/prescriptions and receive simple, patient-friendly explanations.

## 1) Chosen stack

- Frontend: Next.js (App Router) + TypeScript + Tailwind CSS
- Backend: FastAPI + SQLAlchemy
- Database: PostgreSQL
- Authentication: JWT (login/signup/logout)
- OCR: Tesseract OCR (`pytesseract`) + PDF rendering (`PyMuPDF`)
- AI Simplification: lightweight transformer pipeline (`flan-t5-small`) with medical-term replacement fallback

## 2) Folder structure

- frontend/: Next.js UI app
- backend/: FastAPI API app
- database/: SQL schema

Key paths:
- frontend/app/: landing, auth, dashboard, report detail pages
- frontend/components/: reusable UI sections
- frontend/lib/: API client, auth store, types
- backend/app/routers/: auth + report APIs
- backend/app/services/: OCR + simplification logic
- backend/app/models/: SQLAlchemy models
- backend/app/core/: config, DB, security

## 3) Database schema

Tables:
- users (`id`, `email`, `full_name`, `hashed_password`, `created_at`)
- reports (`id`, `user_id`, `file_name`, `extracted_text`, `simplified_text`, `important_terms`, `created_at`)

SQL file: [database/schema.sql](database/schema.sql)

## 4) Backend API plan

Auth:
- `POST /auth/signup`
- `POST /auth/login`

Reports:
- `POST /reports/upload` (file upload → OCR → simplify → optional save)
- `GET /reports` (user history)
- `GET /reports/{id}` (single report detail)

Utility:
- `GET /health`

## 5) UI plan

Landing:
- Premium hero + gradient style
- Feature cards
- How it works
- Benefits + CTA + footer

Auth:
- Clean login/signup cards
- Input validation
- Clear loading/error states

Dashboard:
- Welcome + quick stats cards
- Polished upload card
- Result card (original text + simplified + key terms)
- History list with detail pages

---

## Setup instructions

## Prerequisites

- Node.js 18+
- Python 3.11+
- PostgreSQL running locally
- Tesseract installed
  - macOS: `brew install tesseract`

## Backend setup

1. Go to backend folder.
2. Create and activate virtual environment.
3. Install dependencies:
   - `pip install -r requirements.txt`
4. Copy env:
   - `cp .env.example .env`
5. Update `DATABASE_URL`, `SECRET_KEY`, and `TESSERACT_CMD` in `.env`.
6. Run API:
   - `uvicorn app.main:app --reload`

API will run on `http://localhost:8000`.

## Frontend setup

1. Go to frontend folder.
2. Install dependencies:
   - `npm install`
3. Copy env:
   - `cp .env.local.example .env.local`
4. Run app:
   - `npm run dev`

Frontend will run on `http://localhost:3000`.

Set `BACKEND_ORIGIN` in `frontend/.env.local` if your API runs on a different host/port. All frontend calls go through `/api` and are proxied to this backend, avoiding CORS issues.

---

## Notes

- The simplification pipeline uses `flan-t5-small`; if loading fails, the app falls back to a rule-based simplifier.
- Uploaded files are stored in `backend/uploads`.
- This is an MVP for demo/final-year project use.

## Training the simplifier model

Use Python 3.11 for training dependencies, then run:

1. `cd backend`
2. `py -3.11 -m venv .venv311`
3. `.venv311\\Scripts\\activate`
4. `pip install -r requirements.txt -r requirements-training.txt`
5. Shared training pairs are automatically saved to MongoDB (`training_samples`) when users save a report.
6. Train from shared MongoDB data (curated mode, default):
   - `python -m app.services.train_simplifier --source mongodb --output-dir ./model_cache/simplifier`
   - Curated mode prioritizes corrected feedback and rating-threshold samples.
7. If you want to include generated (non-corrected) samples too:
   - `python -m app.services.train_simplifier --source mongodb --include-generated --output-dir ./model_cache/simplifier`
8. Optional: disable caregiver-style pairs during training:
   - `python -m app.services.train_simplifier --source mongodb --no-include-caregiver --output-dir ./model_cache/simplifier`
9. Optional: tune quality thresholds for corrected feedback:
   - `python -m app.services.train_simplifier --source mongodb --min-clarity-rating 4 --min-accuracy-rating 4 --output-dir ./model_cache/simplifier`
10. Optional: combine MongoDB pairs + local file pairs:
   - `python -m app.services.train_simplifier --source both --data path/to/simplify_pairs.jsonl --output-dir ./model_cache/simplifier`
11. Set `SIMPLIFIER_MODEL_PATH=./model_cache/simplifier` in `backend/.env`.

The API contract remains unchanged: existing report upload and simplify routes continue to work.
