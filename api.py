"""Local FastAPI adapter for the existing TrueLens image pipeline."""

import os
import logging
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from image_pipeline import run_local_image_pipeline


log = logging.getLogger(__name__)


MAX_UPLOAD_BYTES = 50 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/gif"}
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
DEFAULT_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"

app = FastAPI(title="TrueLens API", version="0.1.0")
allowed_origins = [
    origin.strip()
    for origin in os.getenv("TRUELENS_CORS_ORIGINS", DEFAULT_ORIGINS).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def _validate_image(path: Path) -> None:
    """Validate decoded content; MIME type and extension are only hints."""
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from exc


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "TrueLens API"}


@app.get("/")
def root() -> dict:
    return {"service": "TrueLens API", "health": "/health", "analyze_image": "/api/analyze-image"}


@app.post("/api/analyze-image")
async def analyze_image(file: Annotated[UploadFile, File(...)]) -> dict:
    """Analyze one uploaded image through the unchanged four-agent pipeline."""
    suffix = Path(file.filename or "upload").suffix.lower()
    if file.content_type not in ALLOWED_IMAGE_TYPES and suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Supported formats are JPEG, PNG, WebP, BMP, and GIF.")

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="truelens-upload-", suffix=suffix or ".img", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            uploaded_bytes = 0
            while chunk := await file.read(1024 * 1024):
                uploaded_bytes += len(chunk)
                if uploaded_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Image exceeds the 50 MB upload limit.")
                temp_file.write(chunk)

        if temp_path.stat().st_size == 0:
            raise HTTPException(status_code=400, detail="The uploaded image is empty.")
        _validate_image(temp_path)
        result = run_local_image_pipeline(str(temp_path))
        ingestor = result.get('agents', {}).get('ingestor', {})
        decision = result.get('final_decision', {})
        log.info('analyze-image filename=%s temp_path=%s source_hash=%s canonical_path=%s '
                 'category=%s baseline=%s', file.filename, temp_path,
                 ingestor.get('source_file_hash'), ingestor.get('normalized_file_path'),
                 decision.get('category'), decision.get('decision_state', {}).get('baseline_trigger'))
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Image analysis could not be completed.") from exc
    finally:
        await file.close()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
