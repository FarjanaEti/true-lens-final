# TrueLens local application

The repository keeps the existing Python pipeline at its root and adds a FastAPI adapter (`api.py`) plus a Next.js TypeScript frontend in `frontend/`. The four agents and `decision_engine.py` are unchanged.

## Run locally

In PowerShell from the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements-image.txt
uvicorn api:app --reload
```

The API listens at `http://127.0.0.1:8000`; check it with `GET /health`. The image endpoint is `POST /api/analyze-image` using a multipart `file` field.

In a second terminal:

```powershell
cd frontend
Copy-Item .env.local.example .env.local
npm install
npm run dev
```

Open `http://localhost:3000`. `NEXT_PUBLIC_API_URL` in `frontend/.env.local` defaults to `http://127.0.0.1:8000`.

## API response

`POST /api/analyze-image` returns the real dictionary from `image_pipeline.run_local_image_pipeline()`: `status`, `processed_at`, `media_type`, `agents` (ingestor, provenance, forensics, cross_reference), `timing_ms`, `final_decision`, and `ready_for_display`. `final_decision` includes `category`, `confidence`, `likelihood`, evidence lists, warnings, agent summaries/outputs, and decision state where available. Likelihood values are explicitly evidence distributions, not calibrated probabilities.

The upload is validated by decoded image content, limited to 50 MB, written to a temporary file, and deleted after processing. The existing Ingestor may retain its normal canonical image output in the repository, exactly as it does for the CLI workflow.

## Existing CLI

The original workflow remains:

```powershell
.\.venv\Scripts\python.exe image_pipeline.py imageprocessor.jpeg
```

The pipeline entry point is `image_pipeline.run_pipeline()`. It invokes Ingestor → Provenance → Forensics → Cross-reference, then calls `run_decision_engine(ingestor, provenance, forensics, cross_reference)` without averaging scores.
