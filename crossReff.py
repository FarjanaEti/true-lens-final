from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
import requests
from urllib.parse import urlparse
from difflib import SequenceMatcher
from datetime import datetime, timezone
import time
import logging
import string
import json
import re
import os
import shutil

# !apt install - y tesseract-ocr ffmpeg
# !pip install pytesseract Pillow requests openai-whisper imagehash
# !pip install google-api-python-client  # for Google Fact Check API
# # INSTALL (run this cell first in Colab/Kaggle):
# !apt install - y ffmpeg
# !pip install opencv-python-headless numpy Pillow scipy librosa
# !pip install torch torchvision torchaudio - -index-url https: // download.pytorch.org/whl/cu118


_LEGACY_PROVENANCE_JSON = r"""
 {
  "status": "success",
  "processed_at": "2026-08-28T17:31:33.470721",
  "agent": "provenance",
  "file_path": "/content/canonical_image.png",
  "file_hash": "9cfaa42b90fa4ca4171124b63dc2b5582f4200813c7001f7d51ec3417a695187",
  "metadata": {
    "raw": {
      "_xmp_detected": false,
      "_iptc_detected": false
    },
    "flags": [
      "NO_EXIF: Image has no EXIF data at all",
      "NO_THUMBNAIL: No embedded EXIF thumbnail found"
    ]
  },
  "c2pa": {
    "c2pa_present": false,
    "c2pa_valid": false,
    "c2pa_status": "NO_C2PA",
    "c2pa_issuer": null,
    "c2pa_timestamp": null,
    "c2pa_claims": [],
    "note": "No C2PA manifest found."
  },
  "hash_lookup": {
    "hash_match": false,
    "verdict": "NOT_CHECKED",
    "source": null,
    "detail": "External hash database lookup is not implemented in this build",
    "source_layer": "external"
  },
  "provenance_score": 45,
  "provenance_confidence": "LOW",
  "provenance_reasoning": [
    "NO_C2PA:No C2PA Content Credentials were found.This is only a weak negative signal because most online videoscurrently do not contain C2PA manifests.",
    "HASH_UNKNOWN: No prior record in internal database (external lookup not implemented)",
    "HASH_INTEGRITY_OK: Recomputed hash matches Ingestor Agent's reported hash",
    "METADATA_FLAG [LOW_CONFIDENCE]: NO_EXIF: Image has no EXIF data at all",
    "METADATA_FLAG [LOW_CONFIDENCE]: NO_THUMBNAIL: No embedded EXIF thumbnail found",
    "METADATA_PENALTY_APPLIED: -2 (capped at -25)"
  ],
  "chain_of_custody_established": false,
  "metadata_summary": {
    "media_type": "image",
    "software_or_encoder": null,
    "camera_make": null,
    "camera_model": null,
    "resolution": null,
    "codec": null,
    "container_format": null,
    "duration_seconds": null,
    "extraction_method": "exif/exifread",
    "total_flags_raised": 2
  },
  "confidence_breakdown": {
    "HIGH_CONFIDENCE": [],
    "MEDIUM_CONFIDENCE": [],
    "LOW_CONFIDENCE": [
      "NO_EXIF: Image has no EXIF data at all",
      "NO_THUMBNAIL: No embedded EXIF thumbnail found"
    ]
  },
  "integrity_check": {
    "recomputed_hash": "9cfaa42b90fa4ca4171124b63dc2b5582f4200813c7001f7d51ec3417a695187",
    "ingestor_reported_hash": "9cfaa42b90fa4ca4171124b63dc2b5582f4200813c7001f7d51ec3417a695187",
    "hash_mismatch": false
  },
  "hash_recomputed": "9cfaa42b90fa4ca4171124b63dc2b5582f4200813c7001f7d51ec3417a695187",
  "hash_matches_ingestor": true,
  "analysis_notes": [
    "External hash database lookup is a placeholder in this build \u2014 only the internal demo lookup table was actually queried."
  ],
  "ready_for_next_agent": true
}
"""

_LEGACY_FORENSICS_JSON = r"""
 {
  "status": "success",
  "processed_at": "2026-08-28T17:32:48.727380",
  "agent": "forensics",
  "file_path": "/content/canonical_image.png",
  "media_type": "image",
  "media_authenticity": "UNCERTAIN",
  "confidence": 85,
  "forensics_score": 50,
  "evidence": [
    "HIGH_ELA_MEAN: Average error level is 17.3 \u2014 suggests compression history mismatch",
    "No GAN fingerprints or frequency-domain anomalies detected",
    "No face could be located in this image by any detector in the fallback chain (YuNet DNN, Haar primary/secondary, skin-tone heuristic) \u2014 face-artifact analysis could not be performed. This does not confirm the image is clean; it means this module has no opinion.",
    "LIGHTING_DIRECTION_SEVERE: circ_var=0.98 \u2014 near-maximal directional inconsistency, well beyond normal single-light-source photography; strong indicator of synthetic/composited lighting",
    "No hand region found, or estimated finger count was plausible"
  ],
  "recommended_next_step": "SEND_TO_CROSS_REFERENCE",
  "explainability": [
    {
      "feature_checked": "Error Level Analysis (compression consistency)",
      "result": "HIGH_ELA_MEAN: Average error level is 17.3 \u2014 suggests compression history mismatch",
      "confidence": 70,
      "contribution_to_score": -10
    },
    {
      "feature_checked": "Noise & frequency-domain analysis",
      "result": "No GAN fingerprints or frequency-domain anomalies detected",
      "confidence": 85,
      "contribution_to_score": 0
    },
    {
      "feature_checked": "Face artifact / deepfake detector",
      "result": "No face could be located in this image by any detector in the fallback chain (YuNet DNN, Haar primary/secondary, skin-tone heuristic) \u2014 face-artifact analysis could not be performed. This does not confirm the image is clean; it means this module has no opinion.",
      "confidence": 15,
      "contribution_to_score": 0
    },
    {
      "feature_checked": "AI-generation heuristics (symmetry, eyes, ears, edges, texture, lighting)",
      "result": "LIGHTING_DIRECTION_SEVERE: circ_var=0.98 \u2014 near-maximal directional inconsistency, well beyond normal single-light-source photography; strong indicator of synthetic/composited lighting",
      "confidence": 90,
      "contribution_to_score": -20
    },
    {
      "feature_checked": "Hand/finger artifact heuristic",
      "result": "No hand region found, or estimated finger count was plausible",
      "confidence": 85,
      "contribution_to_score": 0
    }
  ],
  "analysis_limitations": [
    "Hand/finger artifact heuristic: NO_HAND_REGION_DETECTED: Hand-shaped region(s) found but too small/degenerate to analyze reliably \u2014 treated as informational, not evidence",
    "This agent uses heuristic and signal-based forensic techniques (compression analysis, pixel statistics, YuNet DNN + Haar-cascade face detection, classical CV symmetry/texture/lighting checks, spectral audio features) rather than trained deep-learning manipulation classifiers. Results should be treated as screening-level evidence, combined with cross-referencing, and not relied upon in isolation for high-stakes decisions."
  ],
  "score_reasoning": [
    "MODERATE_FLAG: HIGH_ELA_MEAN: Average error level is 17.3 \u2014 suggests compression history mismatch (-10 pts)",
    "CRITICAL_FLAG: LIGHTING_DIRECTION_SEVERE: circ_var=0.98 \u2014 near-maximal directional inconsistency, well beyond normal single-light-source photography; strong indicator of synthetic/composited lighting (-20 pts \u2014 high-reliability indicator)",
    "CORROBORATION_BONUS: 2 independent evidence dimensions show real findings (compression, structural_texture) (-5 pts, single bounded application)"
  ],
  "raw_findings": {
    "ela": {
      "ela_max": 250,
      "ela_mean": 17.324,
      "suspicious_region_percent": 4.17,
      "flags": [
        "HIGH_ELA_MEAN: Average error level is 17.3 \u2014 suggests compression history mismatch"
      ],
      "module_status": "ok",
      "clean_evidence_text": "Source format is PNG, which has no prior lossy-compression history for ELA to analyze against. A 'clean' ELA reading here is NOT evidence of authenticity \u2014 this module has no opinion on this file.",
      "clean_evidence_confidence": 10,
      "ela_module_applicable": false
    },
    "noise_frequency": {
      "noise_std": 29.134,
      "noise_mean": 21.431,
      "fft_freq_ratio": 1.21,
      "flags": [],
      "module_status": "ok"
    },
    "face_artifacts": {
      "faces_found": 0,
      "face_boxes": [],
      "detection_method": "none",
      "detection_confidence": "none",
      "clean_evidence_text": "No face could be located in this image by any detector in the fallback chain (YuNet DNN, Haar primary/secondary, skin-tone heuristic) \u2014 face-artifact analysis could not be performed. This does not confirm the image is clean; it means this module has no opinion.",
      "clean_evidence_confidence": 15,
      "flags": [],
      "module_status": "ok"
    },
    "ai_heuristics": {
      "metrics": {
        "edge_density_std": 0.0656,
        "texture_coefficient_of_variation": 0.53,
        "lighting_direction_circular_variance": 0.982
      },
      "flags": [
        "LIGHTING_DIRECTION_SEVERE: circ_var=0.98 \u2014 near-maximal directional inconsistency, well beyond normal single-light-source photography; strong indicator of synthetic/composited lighting"
      ],
      "module_status": "ok"
    },
    "hand_artifacts": {
      "hand_region_found": false,
      "regions_analyzed": 1,
      "estimated_finger_count": null,
      "region_results": [
        {
          "region_index": 1,
          "analyzable": false,
          "estimated_finger_count": null
        }
      ],
      "flags": [
        "NO_HAND_REGION_DETECTED: Hand-shaped region(s) found but too small/degenerate to analyze reliably \u2014 treated as informational, not evidence"
      ],
      "module_status": "ok"
    }
  },
  "all_flags": [
    "HIGH_ELA_MEAN: Average error level is 17.3 \u2014 suggests compression history mismatch",
    "LIGHTING_DIRECTION_SEVERE: circ_var=0.98 \u2014 near-maximal directional inconsistency, well beyond normal single-light-source photography; strong indicator of synthetic/composited lighting",
    "NO_HAND_REGION_DETECTED: Hand-shaped region(s) found but too small/degenerate to analyze reliably \u2014 treated as informational, not evidence"
  ],
  "total_flags_found": 3,
  "ready_for_next_agent": true
}
"""


# ============================================================
# OPTIONAL DEPENDENCIES
# ============================================================

try:
    from bs4 import BeautifulSoup

    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("WARNING: beautifulsoup4 not installed. Trusted web search unavailable.")


try:
    import pytesseract

    if shutil.which("tesseract") is None:
        windows_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(windows_tesseract):
            pytesseract.pytesseract.tesseract_cmd = windows_tesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("WARNING: pytesseract not installed. OCR unavailable.")


try:
    import whisper

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("WARNING: whisper not installed. Audio transcription unavailable.")


try:
    import spacy

    try:
        _NLP = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except OSError:
        _NLP = None
        SPACY_AVAILABLE = False
        print(
            "WARNING: spaCy installed but en_core_web_sm is missing. "
            "Falling back to regex entity extraction."
        )

except ImportError:
    spacy = None
    _NLP = None
    SPACY_AVAILABLE = False
    print("WARNING: spaCy not installed — " "falling back to regex entity extraction.")


# PATCH: perceptual hashing for visual similarity module (see below)
try:
    import imagehash

    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False
    print("WARNING: imagehash not installed. Visual similarity check unavailable.")


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [CROSSREF] %(message)s", datefmt="%H:%M:%S"
)

log = logging.getLogger(__name__)


def _coerce_agent_dict(value, boundary_name: str) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{boundary_name} must be a dict or valid JSON object: {exc}") from exc
        if isinstance(decoded, dict):
            return decoded
    raise TypeError(f"{boundary_name} must be a dictionary")


# ============================================================
# CONFIGURATION
# ============================================================

GOOGLE_FACTCHECK_API_KEY = ""
NEWS_API_KEY = ""

# Explicit trusted domains.
# Avoid using generic "gov" because it can match unrelated domains.
TRUSTED_DOMAINS = [
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "afp.com",
    "factcheck.afp.com",
    "politifact.com",
    "snopes.com",
    "factcheck.org",
    "who.int",
    "un.org",
    "nasa.gov",
    "cdc.gov",
    "fema.gov",
    "gov.uk",
    "gov.bd",
]

MAX_CLAIMS = 5
MAX_SEARCH_QUERIES_PER_CLAIM = 2
MAX_RESULTS_PER_QUERY = 3

# PATCH (DDG CIRCUIT BREAKER): timeout split into (connect, read) instead
# of one flat 10s value — a dead/unreachable host now fails on the connect
# phase in ~4s instead of waiting the full 10s every single time.
REQUEST_TIMEOUT = (2, 4)

# Query trusted domains in small groups instead of issuing one sequential
# request per domain. All domains remain eligible, but an evidence lookup now
# needs at most two DDG requests rather than fourteen.
TRUSTED_DOMAIN_GROUP_SIZE = 7

# PATCH: known-image hash database for visual similarity lookup.
# {phash_hex: {"label": str, "source": str, "context": str}}
# Empty until you populate it — see check_visual_similarity() below for how
# it's used and how to start building this up over time (e.g. by logging
# every processed image's pHash regardless of match/no-match).
KNOWN_IMAGE_HASH_DB = {}
VISUAL_SIMILARITY_MAX_DISTANCE = 8  # 64-bit pHash; ~32 is random-chance distance


# ============================================================
# PATCH (DDG CIRCUIT BREAKER)
#
# WHY: with no circuit breaker, a single blocked/unreachable network turns
# into a near-total hang: search_trusted_sources() tries all 14
# TRUSTED_DOMAINS, and verify_claims() calls that once per query (up to 5)
# per claim (up to 5) — worst case ~350 sequential timeout requests before
# the run ever finishes. This is exactly what happened on the video test:
# html.duckduckgo.com was unreachable and the run had to be killed manually.
#
# This adds a session-level breaker: after a small number of consecutive
# CONNECTION-level failures (timeouts, connection refused/reset — NOT
# normal "zero results" responses), it stops attempting further DDG calls
# entirely for a cooldown period, logs ONE clear message instead of
# repeating the same warning dozens of times, and lets the rest of the run
# finish fast using whatever other evidence sources are configured.
# ============================================================

# trip after this many connection-level failures in a row
_DDG_MAX_CONSECUTIVE_FAILURES = 2
# after tripping, don't retry DDG at all for this long
_DDG_CIRCUIT_COOLDOWN_SECONDS = 300

_ddg_consecutive_failures = 0
_ddg_circuit_open = False
_ddg_circuit_opened_at = None


def _ddg_circuit_is_open() -> bool:
    """True if the circuit breaker has tripped and DDG calls should be
    skipped entirely. Auto-resets after the cooldown, in case the network
    issue was transient (e.g. a temporary rate limit)."""
    global _ddg_circuit_open, _ddg_circuit_opened_at, _ddg_consecutive_failures

    if not _ddg_circuit_open:
        return False

    if (
        _ddg_circuit_opened_at
        and (time.time() - _ddg_circuit_opened_at) > _DDG_CIRCUIT_COOLDOWN_SECONDS
    ):
        log.info("DDG circuit breaker cooldown elapsed — will retry DuckDuckGo search.")
        _ddg_circuit_open = False
        _ddg_consecutive_failures = 0
        _ddg_circuit_opened_at = None
        return False

    return True


def _ddg_record_failure(reason: str):
    global _ddg_consecutive_failures, _ddg_circuit_open, _ddg_circuit_opened_at

    _ddg_consecutive_failures += 1

    if _ddg_consecutive_failures >= _DDG_MAX_CONSECUTIVE_FAILURES and not _ddg_circuit_open:
        _ddg_circuit_open = True
        _ddg_circuit_opened_at = time.time()
        log.warning(
            f"DDG_CIRCUIT_BREAKER_TRIPPED: {_ddg_consecutive_failures} consecutive connection failures "
            f"to html.duckduckgo.com (last: {reason}). This looks like DuckDuckGo is unreachable from this "
            f"environment (common on Colab IP ranges), not an intermittent issue. Skipping all further "
            f"DuckDuckGo searches for the next {_DDG_CIRCUIT_COOLDOWN_SECONDS}s rather than retrying "
            f"per-domain, per-query, per-claim. Trusted-source web search will report no results for the "
            f"rest of this run unless Google Fact Check / News API keys are configured as alternate sources."
        )


def _ddg_record_success():
    global _ddg_consecutive_failures
    _ddg_consecutive_failures = 0


# ============================================================
# STEP 0 — FORENSIC CIRCUIT BREAKER
# ============================================================


def should_skip_cross_reference(forensics_output):
    """
    Cross-Reference is skipped ONLY when Forensics has
    sufficiently high-confidence evidence of manipulation.

    Routing policy:

        HIGH-CONFIDENCE MANIPULATED / AI_GENERATED
            -> skip Cross-Reference

        AUTHENTIC
            -> Cross-Reference

        UNCERTAIN
            -> Cross-Reference

        LOW-CONFIDENCE MANIPULATED
            -> Cross-Reference
    """

    if not isinstance(forensics_output, dict):
        return False, "FORENSICS_INVALID_INPUT"

    classification = (
        forensics_output.get("classification")
        or forensics_output.get("media_authenticity")
        or "UNCERTAIN"
    )

    classification = str(classification).upper().strip()

    raw_confidence = forensics_output.get("confidence", 0)

    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        confidence = 0.0

    # Support both 0-1 and 0-100 confidence formats.
    if 0.0 <= confidence <= 1.0:
        confidence *= 100.0

    manipulated_states = {
        "MANIPULATED",
        "AI_GENERATED",
        "FAKE",
        "AI-GENERATED",
        "MANIPULATED_MEDIA",
    }

    if classification in manipulated_states and confidence >= 85:
        return (
            True,
            ("FORENSICS_HIGH_CONFIDENCE_MANIPULATION: " f"{classification} ({confidence:.1f}%)"),
        )

    return (False, ("FORENSICS_REQUIRES_CROSSREF: " f"{classification} ({confidence:.1f}%)"))


# ============================================================
# MEDIA TYPE
# ============================================================


def detect_media_type(file_path: str) -> str:

    if not file_path:
        return "unknown"

    ext = os.path.splitext(file_path)[1].lower()

    if ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"):
        return "image"

    if ext in (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"):
        return "video"

    return "unknown"


# ============================================================
# PATCH: VISUAL SIMILARITY MODULE
#
# WHY: every other module in this agent is gated on text (OCR overlay text
# or a spoken transcript). A bare photo with no caption and no on-screen
# text — like the flood-water test image — makes every downstream step a
# no-op: extract_claims() returns [], verify_claims() has nothing to check,
# and the agent's only output is a single NO_FACTUAL_CLAIM_EXTRACTED flag.
# That's a real coverage gap, not a correct "nothing to report" — reverse-
# image/perceptual-hash matching is a completely independent evidence type
# that doesn't need any text at all: "has this exact or near-identical
# image appeared online before, and under what story/date/context?" is
# often the single most useful cross-reference check for photo-only content
# (misattributed old disaster photos being the classic case).
#
# This ships as a real, callable module using a local perceptual-hash
# database. It returns NOT_CHECKED / NO_MATCH honestly rather than faking
# a result, and is written so wiring in a real external reverse-image API
# (Google Vision, TinEye, PimEyes, etc.) later is a drop-in replacement of
# the lookup step, not a redesign.
# ============================================================


def compute_perceptual_hashes(image_path: str) -> dict:
    """
    Compute perceptual hashes for one image.

    A valid hash may legitimately be all-zero for a perfectly uniform
    image, but an all-zero hash on the first video frame is commonly
    caused by a black/blank opening frame. The caller therefore validates
    the hashes before using them for visual matching.
    """
    if not IMAGEHASH_AVAILABLE:
        return {"error": "imagehash not installed"}

    with Image.open(image_path) as im:
        im = im.convert("RGB")

        hashes = {
            "phash": str(imagehash.phash(im)),
            "dhash": str(imagehash.dhash(im)),
            "average_hash": str(imagehash.average_hash(im)),
        }

    return hashes


def _hashes_are_degenerate(hashes: dict) -> bool:
    """Return True when every computed hash is all zeros."""
    if not hashes:
        return True

    return all(
        str(hashes.get(name, "")).lower() in {"0000000000000000", "0" * 16}
        for name in ("phash", "dhash", "average_hash")
    )


def _extract_hash_candidate_frames(video_path: str, max_candidates: int = 8) -> list:
    """
    Extract several evenly spaced frames instead of blindly hashing frame 0.
    This avoids using a black/blank opening frame as the representative hash.
    """
    frames = []

    try:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return frames

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            return frames

        positions = sorted(
            set(int(total_frames * i / max_candidates) for i in range(max_candidates))
        )

        for position in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(position, total_frames - 1))

            ret, frame = cap.read()

            if not ret or frame is None:
                continue

            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
            except Exception:
                continue

        cap.release()

    except Exception as e:
        log.warning(f"Hash candidate frame extraction failed: {e}")

    return frames


def check_visual_similarity(
    file_path: str,
    media_type: str,
    known_hash_db: dict = None,
    max_distance: int = VISUAL_SIMILARITY_MAX_DISTANCE,
) -> dict:
    """
    Independent visual-similarity check.

    For images, hash the supplied image.
    For videos, sample several frames and choose the first non-degenerate
    frame hash. A blank first frame must never be silently reported as the
    meaningful visual hash.

    IMPORTANT:
      NOT_CHECKED means no database exists.
      NO_MATCH means a populated database was actually checked.
      MODULE_UNAVAILABLE / error means the module could not complete.
    """
    log.info("Running visual similarity (perceptual hash) check...")

    if not known_hash_db:
        return {
            "checked": False,
            "verdict": "NOT_CHECKED",
            "flags": [
                "VISUAL_SIMILARITY_NO_DATABASE: No known-image database is populated; "
                "hashing was skipped and this result has no evidentiary weight."
            ],
            "module_status": "not_applicable",
        }

    if not IMAGEHASH_AVAILABLE:
        return {
            "checked": False,
            "verdict": "MODULE_UNAVAILABLE",
            "flags": ["IMAGEHASH_NOT_INSTALLED: " "Visual similarity check skipped"],
            "module_status": "unavailable",
        }

    try:
        candidate_hashes = []

        if media_type == "image":
            hashes = compute_perceptual_hashes(file_path)

            if "error" in hashes:
                return {
                    "checked": False,
                    "verdict": "MODULE_UNAVAILABLE",
                    "flags": [f"VISUAL_SIMILARITY_ERROR: {hashes['error']}"],
                    "module_status": "error",
                }

            candidate_hashes.append(
                {"hashes": hashes, "frame_index": None, "selection_reason": "SOURCE_IMAGE"}
            )

        elif media_type == "video":
            frames = _extract_hash_candidate_frames(file_path, max_candidates=8)

            if not frames:
                return {
                    "checked": False,
                    "verdict": "NOT_CHECKED",
                    "flags": [
                        "VISUAL_SIMILARITY_NO_FRAME: "
                        "Could not extract a usable frame from video for hashing"
                    ],
                    "module_status": "error",
                }

            for index, frame in enumerate(frames):
                temp_path = f"{file_path}._hash_frame_{index}.jpg"

                try:
                    frame.save(temp_path, "JPEG", quality=90)

                    hashes = compute_perceptual_hashes(temp_path)

                    candidate_hashes.append(
                        {
                            "hashes": hashes,
                            "frame_index": index,
                            "selection_reason": (
                                "FIRST_NON_DEGENERATE_FRAME"
                                if not _hashes_are_degenerate(hashes)
                                else "DEGENERATE_FRAME"
                            ),
                        }
                    )

                finally:
                    if os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except OSError:
                            pass

            non_degenerate = [
                item for item in candidate_hashes if not _hashes_are_degenerate(item["hashes"])
            ]

            if non_degenerate:
                selected = non_degenerate[0]
            else:
                return {
                    "checked": False,
                    "verdict": "NOT_CHECKED",
                    "computed_hashes": (candidate_hashes[0]["hashes"] if candidate_hashes else {}),
                    "flags": [
                        "VISUAL_SIMILARITY_DEGENERATE_FRAMES: "
                        "All sampled video frames produced "
                        "degenerate all-zero perceptual hashes; "
                        "visual similarity was not scored."
                    ],
                    "module_status": "ok",
                }

            hashes = selected["hashes"]
            selected_frame_index = selected["frame_index"]
            selection_reason = selected["selection_reason"]

        else:
            return {
                "checked": False,
                "verdict": "MODULE_UNAVAILABLE",
                "flags": ["VISUAL_SIMILARITY_UNSUPPORTED_MEDIA_TYPE"],
                "module_status": "unavailable",
            }

        if not known_hash_db:
            return {
                "checked": True,
                "verdict": "NOT_CHECKED",
                "computed_hashes": hashes,
                "sampled_frame_index": (selected_frame_index if media_type == "video" else None),
                "selection_reason": (selection_reason if media_type == "video" else "SOURCE_IMAGE"),
                "flags": [
                    "VISUAL_SIMILARITY_NO_DATABASE: "
                    "Perceptual hash was computed but no known-image "
                    "database is populated yet. This is NOT_CHECKED, "
                    "not a clean/no-match result."
                ],
                "module_status": "ok",
            }

        query = imagehash.hex_to_hash(hashes["phash"])

        best_match = None
        best_dist = 999

        for known_hex, meta in known_hash_db.items():
            try:
                known_hash = imagehash.hex_to_hash(known_hex)
                dist = query - known_hash

                if dist < best_dist:
                    best_dist = dist
                    best_match = meta

            except Exception:
                continue

        if best_match is not None and best_dist <= max_distance:
            return {
                "checked": True,
                "verdict": "NEAR_DUPLICATE_FOUND",
                "match_distance": int(best_dist),
                "match_info": best_match,
                "computed_hashes": hashes,
                "sampled_frame_index": (selected_frame_index if media_type == "video" else None),
                "selection_reason": (selection_reason if media_type == "video" else "SOURCE_IMAGE"),
                "flags": [
                    "VISUAL_SIMILARITY_MATCH: "
                    "Near-duplicate found in known-image database "
                    f"(hash distance={best_dist}, "
                    f"threshold={max_distance})."
                ],
                "module_status": "ok",
            }

        return {
            "checked": True,
            "verdict": "NO_MATCH",
            "computed_hashes": hashes,
            "sampled_frame_index": (selected_frame_index if media_type == "video" else None),
            "selection_reason": (selection_reason if media_type == "video" else "SOURCE_IMAGE"),
            "flags": [],
            "module_status": "ok",
        }

    except Exception as e:
        log.warning(f"Visual similarity check failed: {e}")

        return {
            "checked": False,
            "verdict": "MODULE_UNAVAILABLE",
            "flags": [f"VISUAL_SIMILARITY_ERROR: {str(e)}"],
            "module_status": "error",
        }


# ============================================================
# FRAME EXTRACTION
# ============================================================


def extract_key_frames(video_path: str, num_frames: int = 8) -> list:

    log.info(f"Extracting up to {num_frames} key frames for OCR...")

    frames = []

    try:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            log.warning("Could not open video for frame extraction.")
            return frames

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            return frames

        interval = max(1, total_frames // max(1, num_frames))

        for i in range(num_frames):

            cap.set(cv2.CAP_PROP_POS_FRAMES, min(i * interval, total_frames - 1))

            ret, frame = cap.read()

            if not ret or frame is None:
                continue

            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frames.append(Image.fromarray(rgb))

            except Exception as frame_error:
                log.warning(f"Frame conversion failed: {frame_error}")

        cap.release()

    except Exception as e:
        log.warning(f"Frame extraction failed: {e}")

    return frames


# ============================================================
# OCR
# ============================================================


def _prepare_image_ocr_variants(file_path: str) -> list:
    """Build conservative OCR variants without modifying the source image."""
    with Image.open(file_path) as source:
        base = ImageOps.exif_transpose(source).convert("RGB")
        longest = max(base.size)
        if longest < 2200:
            scale = min(3.0, 2200 / max(longest, 1))
            base = base.resize(
                (max(1, int(base.width * scale)), max(1, int(base.height * scale))),
                Image.Resampling.LANCZOS,
            )

    gray = ImageOps.grayscale(base)
    contrast = ImageEnhance.Contrast(ImageOps.autocontrast(gray)).enhance(1.5)
    denoised = contrast.filter(ImageFilter.MedianFilter(size=3))
    cv_gray = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2GRAY)
    if max(cv_gray.shape) < 2200:
        scale = min(3.0, 2200 / max(cv_gray.shape))
        cv_gray = cv2.resize(cv_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    adaptive = cv2.adaptiveThreshold(
        cv_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    return [
        ("orientation_corrected", base, "--oem 3 --psm 3"),
        ("document_grayscale", denoised, "--oem 3 --psm 4"),
        ("document_threshold", Image.fromarray(adaptive), "--oem 3 --psm 6"),
    ]


def _merge_ocr_passes(texts: list) -> str:
    """Keep the strongest pass and append only genuinely new OCR lines."""
    texts = [text.strip() for text in texts if text and text.strip()]
    if not texts:
        return ""
    texts.sort(key=lambda value: sum(ch.isalnum() for ch in value), reverse=True)
    output_lines = []
    normalized_seen = []
    for text in texts:
        for line in text.splitlines():
            line = re.sub(r"\s+", " ", line).strip()
            normalized = re.sub(r"[^a-z0-9]+", "", line.lower())
            if len(normalized) < 2:
                continue
            if any(
                SequenceMatcher(None, normalized, prior).ratio() >= 0.88
                for prior in normalized_seen
            ):
                continue
            normalized_seen.append(normalized)
            output_lines.append(line)
    return "\n".join(output_lines)


def extract_text_via_ocr(file_path: str, media_type: str) -> dict:

    log.info("Running OCR text extraction...")

    if not TESSERACT_AVAILABLE:
        return {
            "ocr_text": "",
            "ocr_text_length": 0,
            "flags": ["TESSERACT_NOT_INSTALLED: OCR skipped"],
            "module_status": "unavailable",
        }

    extracted_texts = []
    flags = []

    try:

        if media_type == "image":
            ocr_attempts = []
            for name, image, config in _prepare_image_ocr_variants(file_path):
                try:
                    text = pytesseract.image_to_string(image, config=config).strip()
                    ocr_attempts.append({"name": name, "characters": len(text)})
                    if text:
                        extracted_texts.append(text)
                except Exception as pass_error:
                    log.warning(f"OCR pass {name} failed: {pass_error}")

        elif media_type == "video":

            frames = extract_key_frames(file_path, num_frames=8)

            for i, frame in enumerate(frames):

                try:

                    text = pytesseract.image_to_string(frame, config=r"--oem 3 --psm 3").strip()

                    if text:
                        extracted_texts.append(text)

                except Exception as e:
                    log.warning(f"OCR failed for frame {i + 1}: {e}")

        else:

            return {
                "ocr_text": "",
                "ocr_text_length": 0,
                "flags": ["UNSUPPORTED_MEDIA_TYPE_FOR_OCR"],
                "module_status": "unavailable",
            }

        # Deduplicate while preserving order.
        combined_text = (
            _merge_ocr_passes(extracted_texts)
            if media_type == "image"
            else "\n".join(dict.fromkeys(t.strip() for t in extracted_texts if t.strip()))
        )

        # Basic sensational-overlay detection.
        for pattern in [
            r"\bBREAKING\b",
            r"\bLEAKED\b",
            r"\bEXCLUSIVE\b",
            r"\bSHOCKING\b",
            r"\bWATCH BEFORE DELETED\b",
        ]:

            if re.search(pattern, combined_text, re.IGNORECASE):
                flags.append("SENSATIONAL_OVERLAY_TEXT: " f"Found '{pattern}' in media")

        if not combined_text:
            flags.append(
                "OCR_NO_TEXT_DETECTED: " "OCR completed but no readable text was extracted."
            )

        return {
            "ocr_text": combined_text,
            "ocr_text_length": len(combined_text),
            "flags": flags,
            "module_status": "ok",
            "ocr_attempts": ocr_attempts if media_type == "image" else [],
        }

    except Exception as e:

        log.warning(f"OCR extraction failed: {e}")

        return {
            "ocr_text": "",
            "ocr_text_length": 0,
            "flags": [f"OCR_ERROR: {str(e)}"],
            "module_status": "error",
        }


# ============================================================
# AUDIO TRANSCRIPTION
# ============================================================


def transcribe_audio(file_path: str, media_type: str) -> dict:

    if media_type != "video":
        return {
            "transcript": "",
            "language": None,
            "transcript_length": 0,
            "flags": [],
            "module_status": "not_applicable",
        }

    log.info("Transcribing audio with Whisper...")

    if not WHISPER_AVAILABLE:
        return {
            "transcript": "",
            "language": None,
            "transcript_length": 0,
            "flags": ["WHISPER_NOT_INSTALLED: " "Transcription skipped"],
            "module_status": "unavailable",
        }

    try:

        model = whisper.load_model("base")

        result = model.transcribe(file_path, fp16=False)

        transcript = result.get("text", "").strip()

        language = result.get("language", "unknown")

        flags = []

        if not transcript:
            flags.append(
                "TRANSCRIPTION_NO_SPEECH_DETECTED: "
                "Whisper completed but no speech text was returned."
            )

        return {
            "transcript": transcript,
            "language": language,
            "transcript_length": len(transcript),
            "flags": flags,
            "module_status": "ok",
        }

    except Exception as e:

        log.warning(f"Transcription failed: {e}")

        return {
            "transcript": "",
            "language": None,
            "transcript_length": 0,
            "flags": [f"TRANSCRIPTION_ERROR: {str(e)}"],
            "module_status": "error",
        }


# ============================================================
# ENTITY KEYWORDS
# ============================================================

_EVENT_KEYWORDS_LEGACY = [
    "fire",
    "flood",
    "earthquake",
    "explosion",
    "attack",
    "protest",
    "election",
    "crash",
    "shooting",
    "storm",
    "coup",
    "bombing",
    "riot",
    "outbreak",
    "epidemic",
    "collapse",
    "cyclone",
    "landslide",
    "strike",
]

_DATE_PATTERNS = [
    r"\byesterday\b",
    r"\btoday\b",
    r"\btomorrow\b",
    r"\blast night\b",
    r"\bthis (morning|week|month|year)\b",
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(st|nd|rd|th)?\b",
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    r"\b20\d{2}\b",
]

_DEATH_CONTEXT = ["died", "dead", "killed", "casualties", "fatalities"]

_INJURY_CONTEXT = ["injured", "wounded", "hurt"]

_ORG_SUFFIXES = [
    "Inc",
    "Corp",
    "Ltd",
    "Police",
    "Government",
    "Ministry",
    "Department",
    "Agency",
    "University",
    "Company",
]

_PERSON_TITLES = [
    "Mr",
    "Mrs",
    "Dr",
    "President",
    "Minister",
    "Prime Minister",
    "Officer",
    "General",
    "Sheikh",
    "Senator",
    "Governor",
]


# ============================================================
# SEMANTIC CATEGORIES
# ============================================================

_DISASTER_KEYWORDS = [
    "flood",
    "flooding",
    "hurricane",
    "wildfire",
    "fire",
    "tornado",
    "earthquake",
    "tsunami",
    "landslide",
    "drought",
    "famine",
    "cyclone",
    "storm",
    "mudslide",
    "avalanche",
    "heatwave",
]

_ELECTION_KEYWORDS = [
    "election",
    "vote",
    "voting",
    "ballot",
    "poll",
    "candidate",
    "referendum",
    "parliament",
    "president-elect",
    "campaign",
    "inauguration",
]

_CONFLICT_KEYWORDS = [
    "war",
    "invasion",
    "ceasefire",
    "airstrike",
    "missile",
    "troops",
    "militant",
    "insurgent",
    "attack",
    "bombing",
    "shelling",
    "conflict",
]

_ACCIDENT_KEYWORDS = [
    "crash",
    "collision",
    "derailment",
    "capsized",
    "collapse",
    "explosion",
    "shooting",
]

_POLITICAL_KEYWORDS = [
    "announced",
    "declared",
    "statement",
    "policy",
    "sanction",
    "resign",
    "resigned",
    "impeach",
    "summit",
    "treaty",
]

_HEALTH_KEYWORDS = [
    "outbreak",
    "virus",
    "vaccine",
    "disease",
    "pandemic",
    "epidemic",
    "hospital",
    "infected",
    "symptoms",
    "cure",
]

_ECONOMIC_KEYWORDS = [
    "inflation",
    "recession",
    "market",
    "stocks",
    "gdp",
    "unemployment",
    "currency",
    "interest rate",
    "economy",
]

_SCIENTIFIC_KEYWORDS = [
    "study",
    "research",
    "scientists",
    "discovery",
    "nasa",
    "climate",
    "experiment",
    "peer-reviewed",
]

_ANNOUNCEMENT_KEYWORDS = [
    "launch",
    "launched",
    "release",
    "released",
    "unveil",
    "unveiled",
    "confirm",
    "confirmed",
]

_DEEPFAKE_KEYWORDS = [
    "deepfake",
    "deep fake",
    "ai generated",
    "ai-generated",
    "synthetic media",
    "face swap",
    "voice clone",
    "generated using ai",
    "digitally altered",
    "manipulated video",
]

_EVENT_KEYWORDS = list(
    dict.fromkeys(
        _EVENT_KEYWORDS_LEGACY
        + _DISASTER_KEYWORDS
        + _ELECTION_KEYWORDS
        + _CONFLICT_KEYWORDS
        + _ACCIDENT_KEYWORDS
    )
)

_CATEGORY_MAP = {
    "disaster": _DISASTER_KEYWORDS,
    "election": _ELECTION_KEYWORDS,
    "conflict": _CONFLICT_KEYWORDS,
    "accident": _ACCIDENT_KEYWORDS,
    "political_statement": _POLITICAL_KEYWORDS,
    "health": _HEALTH_KEYWORDS,
    "economic": _ECONOMIC_KEYWORDS,
    "scientific": _SCIENTIFIC_KEYWORDS,
    "announcement": _ANNOUNCEMENT_KEYWORDS,
    "deepfake_technology": _DEEPFAKE_KEYWORDS,
}


# ============================================================
# RELIGIOUS / EMOTIONAL TEXT DETECTION
# ============================================================

_BIBLE_BOOKS = (
    r"(Genesis|Exodus|Leviticus|Numbers|Deuteronomy|Joshua|Judges|Ruth|Samuel|"
    r"Kings|Chronicles|Ezra|Nehemiah|Esther|Job|Psalms?|Proverbs|Ecclesiastes|"
    r"Song of Solomon|Isaiah|Jeremiah|Lamentations|Ezekiel|Daniel|Hosea|Joel|"
    r"Amos|Obadiah|Jonah|Micah|Nahum|Habakkuk|Zephaniah|Haggai|Zechariah|Malachi|"
    r"Matthew|Mark|Luke|John|Acts|Romans|Corinthians|Galatians|Ephesians|"
    r"Philippians|Colossians|Thessalonians|Timothy|Titus|Philemon|Hebrews|"
    r"James|Peter|Jude|Revelation)"
)

_BIBLE_VERSE_PATTERN = re.compile(
    r"\b([1-3]\s?)?" + _BIBLE_BOOKS + r".?\s\d{1,3}(:\d{1,3}(-\d{1,3})?)?\b", re.IGNORECASE
)

_RELIGIOUS_APPEAL_WORDS = [
    "pray for",
    "prayers for",
    "god bless",
    "thoughts and prayers",
    "keep them in your prayers",
    "praying for",
]


def _contains_religious_appeal(text: str) -> bool:

    if not text:
        return False

    t = text.lower()

    return any(phrase in t for phrase in _RELIGIOUS_APPEAL_WORDS) or bool(
        _BIBLE_VERSE_PATTERN.search(text)
    )


# ============================================================
# REGEX ENTITY EXTRACTION
# ============================================================


def _regex_entities(text: str) -> dict:

    entities = {
        "events": [],
        "locations": [],
        "dates": [],
        "persons": [],
        "organizations": [],
        "numbers": [],
        "time_references": [],
        "religious_references": [],
    }

    if not text:
        return entities

    # Events
    for kw in _EVENT_KEYWORDS:

        pattern = r"\b\w+\s+" + re.escape(kw) + r"\b|" r"\b" + re.escape(kw) + r"\b"

        for m in re.finditer(pattern, text, re.IGNORECASE):
            entities["events"].append(m.group(0).strip())

    # Dates/time references
    for pattern in _DATE_PATTERNS:

        for m in re.finditer(pattern, text, re.IGNORECASE):

            entities["dates"].append(m.group(0))

            entities["time_references"].append(m.group(0))

    # Numbers
    for m in re.finditer(r"\b(\d+)\b", text):

        try:
            value = int(m.group(1))
        except ValueError:
            continue

        window = text[max(0, m.start() - 25) : m.end() + 25].lower()

        if any(w in window for w in _DEATH_CONTEXT):
            context = "death_count"

        elif any(w in window for w in _INJURY_CONTEXT):
            context = "injury_count"

        else:
            context = "generic"

        entities["numbers"].append({"value": value, "context": context})

    # Bible references
    verse_spans = []

    for m in _BIBLE_VERSE_PATTERN.finditer(text):

        entities["religious_references"].append(m.group(0).strip())

        verse_spans.append((m.start(), m.end()))

    def _overlaps_verse(start, end):

        return any(not (end <= vs or start >= ve) for vs, ve in verse_spans)

    # Persons
    for title in _PERSON_TITLES:

        for m in re.finditer(re.escape(title) + r"\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", text):

            if not _overlaps_verse(m.start(), m.end()):
                entities["persons"].append(m.group(0).strip())

    # Organizations
    for suffix in _ORG_SUFFIXES:

        for m in re.finditer(
            r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+" + re.escape(suffix), text
        ):

            if not _overlaps_verse(m.start(), m.end()):
                entities["organizations"].append(m.group(0).strip())

    # Locations
    for m in re.finditer(
        r"\b(?:in|at|near|from)\s+" r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)", text
    ):

        if not _overlaps_verse(m.start(), m.end()):
            entities["locations"].append(m.group(1).strip())

    # Deduplicate
    for key in entities:

        if key == "numbers":

            seen = set()
            deduped = []

            for n in entities[key]:

                k = (n["value"], n["context"])

                if k not in seen:
                    seen.add(k)
                    deduped.append(n)

            entities[key] = deduped

        else:

            entities[key] = list(dict.fromkeys(entities[key]))

    return entities


# ============================================================
# STRUCTURED ENTITY EXTRACTION
# ============================================================


def extract_entities(text: str) -> dict:

    empty = {
        "events": [],
        "locations": [],
        "dates": [],
        "persons": [],
        "organizations": [],
        "numbers": [],
        "time_references": [],
        "religious_references": [],
    }

    if not text or not text.strip():
        return empty

    heuristic = _regex_entities(text)

    if not SPACY_AVAILABLE:
        return heuristic

    try:

        doc = _NLP(text)

        verse_spans = [(m.start(), m.end()) for m in _BIBLE_VERSE_PATTERN.finditer(text)]

        def _overlaps_verse(start, end):

            return any(not (end <= vs or start >= ve) for vs, ve in verse_spans)

        persons = []
        orgs = []
        locations = []
        dates = []

        for ent in doc.ents:

            if _overlaps_verse(ent.start_char, ent.end_char):
                continue

            if ent.label_ == "PERSON":
                persons.append(ent.text)

            elif ent.label_ == "ORG":
                orgs.append(ent.text)

            elif ent.label_ in ("GPE", "LOC"):
                locations.append(ent.text)

            elif ent.label_ == "DATE":
                dates.append(ent.text)

        return {
            "events": heuristic["events"],
            "locations": list(dict.fromkeys(locations + heuristic["locations"])),
            "dates": list(dict.fromkeys(dates + heuristic["dates"])),
            "persons": list(dict.fromkeys(persons + heuristic["persons"])),
            "organizations": list(dict.fromkeys(orgs + heuristic["organizations"])),
            "numbers": heuristic["numbers"],
            "time_references": heuristic["time_references"],
            "religious_references": heuristic["religious_references"],
        }

    except Exception as e:

        log.warning(f"spaCy entity extraction failed; " f"using regex fallback: {e}")

        return heuristic


# ============================================================
# CLAIM CATEGORY CLASSIFICATION
# ============================================================


def _classify_sentence(sentence: str) -> list:

    s = sentence.lower()

    categories = []

    for category, keywords in _CATEGORY_MAP.items():

        if any(kw in s for kw in keywords):
            categories.append(category)

    return categories


_SOCIAL_UI_PATTERNS = [
    r"^@\w+$",
    r"^\d+(?:\.\d+)?[km]?\s+(?:views?|likes?|shares?|comments?|reposts?)$",
    r"^\d{1,2}:\d{2}\s*(?:am|pm)?(?:\s*[|,.-]?\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2}(?:\s*,?\s*\d{2,4})?)?$",
    r"^(?:like|reply|share|follow|following|subscribe|home|search|notifications?)$",
]


def _normalize_claim_text(text: str) -> str:
    """Remove OCR punctuation artifacts without rewriting meaningful words."""
    value = text.replace("\u2018", "'").replace("\u2019", "'").replace("\ufffd", " ")
    value = re.sub(r"[|\\]{2,}", " ", value)
    value = re.sub(r"\s+", " ", value).strip(" \t\r\n,;:|_-.")
    return value


def _claim_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _is_social_ui_or_ocr_garbage(text: str) -> bool:
    value = _normalize_claim_text(text)
    low = value.lower()
    if not value or any(re.fullmatch(pattern, low, re.IGNORECASE) for pattern in _SOCIAL_UI_PATTERNS):
        return True
    if re.search(r"(?:^|\s)@\w+", value) and len(value.split()) <= 5:
        return True
    if re.search(r"\b\d+(?:\.\d+)?[km]?\s+views?\b", low):
        return True
    alnum = sum(ch.isalnum() for ch in value)
    alpha = sum(ch.isalpha() for ch in value)
    if alnum and (alpha / alnum) < 0.55:
        return True
    words = re.findall(r"[A-Za-z]{2,}", value)
    if len(words) < 4:
        return True
    # Random all-caps OCR headers are common in social screenshots. Preserve
    # known acronyms by applying this only to short, mostly uppercase fragments.
    letters = "".join(words)
    if len(words) <= 7 and letters and sum(ch.isupper() for ch in letters) / len(letters) > 0.85:
        return True
    return False


def _claims_are_duplicates(left: str, right: str) -> bool:
    a, b = _claim_key(left), _claim_key(right)
    if not a or not b:
        return False
    if a == b or (min(len(a), len(b)) >= 24 and (a in b or b in a)):
        return True
    a_words, b_words = set(a.split()), set(b.split())
    overlap = len(a_words & b_words) / max(1, min(len(a_words), len(b_words)))
    return overlap >= 0.82 or SequenceMatcher(None, a, b).ratio() >= 0.84


def _claim_candidates(ocr_text: str, transcript: str) -> list:
    """Split OCR by lines as well as punctuation so UI text cannot absorb a caption."""
    combined = f"{ocr_text}\n{transcript}"
    candidates = []
    for line in combined.splitlines():
        for fragment in re.split(r"(?<=[.!?])\s+", line):
            cleaned = _normalize_claim_text(fragment)
            if len(cleaned) >= 20:
                candidates.append(cleaned)
    return candidates


# ============================================================
# PRIMARY CLAIM DETECTION
# ============================================================


def _is_fact_checkable_sentence(sentence: str) -> tuple[bool, str]:
    """
    Decide whether a sentence makes a factual assertion that can reasonably
    be checked against external evidence.

    This deliberately excludes generic opinions, advice, predictions,
    rhetorical statements, and value judgments.
    """
    s = _normalize_claim_text(sentence)
    low = s.lower()

    if len(s) < 20:
        return False, "TOO_SHORT"

    if _is_social_ui_or_ocr_garbage(s):
        return False, "SOCIAL_UI_OR_OCR_GARBAGE"

    opinion_patterns = [
        r"\bwe need to\b",
        r"\bwe should\b",
        r"\bwe must\b",
        r"\bi think\b",
        r"\bi believe\b",
        r"\bi feel\b",
        r"\bin my opinion\b",
        r"\bit seems\b",
        r"\bit's important\b",
        r"\bwe need to be\b",
        r"\bhow we move forward\b",
        r"\bthe difference between\b",
        r"\bdangerous time\b",
        r"\bwould never say\b",
        r"\bthank you\b",
        r"\bstay woke\b",
        r"\bplease keep\b.*\bprayers?\b",
        r"\bthe world needs god\b",
        r"\bamen\b",
    ]

    if any(re.search(pattern, low) for pattern in opinion_patterns):
        return False, "OPINION_OR_ADVICE"

    factual_action_patterns = [
        r"\bwas\b",
        r"\bwere\b",
        r"\bis\b",
        r"\bare\b",
        r"\bhappened\b",
        r"\boccurred\b",
        r"\bkilled\b",
        r"\bdied\b",
        r"\barrested\b",
        r"\bannounced\b",
        r"\bconfirmed\b",
        r"\bdenied\b",
        r"\breported\b",
        r"\bwon\b",
        r"\blost\b",
        r"\bsigned\b",
        r"\blaunched\b",
        r"\battacked\b",
        r"\bstarted\b",
        r"\bended\b",
        r"\bcaused\b",
        r"\bcontains\b",
        r"\bshows\b",
        r"\boccurred\b",
    ]

    has_temporal_or_numeric_detail = bool(
        re.search(
            r"\b(19|20)\d{2}\b"
            r"|\b(yesterday|today|tomorrow|last night|this morning)\b"
            r"|\b\d+\b",
            low,
        )
    )

    has_event_category = bool(_classify_sentence(s))

    has_factual_action = any(re.search(pattern, low) for pattern in factual_action_patterns)

    has_named_entity = bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", s))

    # A name, place, date, or number alone is context/metadata, not an
    # assertion. Require an event/action plus some independently searchable
    # detail. This keeps real claims while rejecting captions such as prayer
    # requests and timestamp/view-count UI.
    if has_factual_action and (
        has_event_category or has_temporal_or_numeric_detail or has_named_entity
    ):
        return True, "FACTUAL_ASSERTION"

    if has_event_category and (has_temporal_or_numeric_detail or has_named_entity):
        return True, "FACTUAL_EVENT_ASSERTION"

    return False, "NO_FACTUAL_SIGNAL"


def detect_primary_claim(ocr_text: str, transcript: str, combined_entities: dict) -> dict | None:
    combined = (f"{ocr_text}\n{transcript}").strip()

    if not combined:
        return None

    combined_lower = combined.lower()

    # Explicit AI/deepfake language is a factual/media claim.
    if any(kw in combined_lower for kw in _DEEPFAKE_KEYWORDS):
        media_word = "video" if transcript else "media"

        return {
            "text": (
                f"This {media_word} contains an explicit "
                "claim about deepfake/AI-generated technology."
            ),
            "entities": combined_entities,
            "categories": ["deepfake_technology"],
            "derivation": "EXPLICIT_DEEPFAKE_LANGUAGE_DETECTED",
        }

    event_hit = None
    event_category = None

    for category_name, keyword_list in [
        ("disaster", _DISASTER_KEYWORDS),
        ("conflict", _CONFLICT_KEYWORDS),
        ("accident", _ACCIDENT_KEYWORDS),
        ("election", _ELECTION_KEYWORDS),
    ]:
        for kw in keyword_list:
            if re.search(r"\b" + re.escape(kw) + r"\b", combined_lower):
                event_hit = kw
                event_category = category_name
                break

        if event_hit:
            break

    location = None

    if combined_entities.get("locations"):
        location = combined_entities["locations"][0]

    if event_hit and location:
        media_word = "video" if transcript else "image"

        return {
            "text": (f"This {media_word} shows " f"{event_hit} in {location}."),
            "entities": combined_entities,
            "categories": [event_category],
            "derivation": "EVENT_KEYWORD_PLUS_LOCATION_INFERRED",
        }

    candidates = []

    for sentence in _claim_candidates(ocr_text, transcript):

        if not sentence:
            continue

        fact_checkable, _ = _is_fact_checkable_sentence(sentence)

        if not fact_checkable:
            continue

        cats = _classify_sentence(sentence)

        has_proper_noun = bool(re.search(r"\b[A-Z][a-z]+" r"(?:\s+[A-Z][a-z]+)*\b", sentence))

        has_action = bool(
            re.search(
                r"\b("
                r"said|announced|confirmed|denied|"
                r"killed|arrested|signed|launched|"
                r"attacked|died|won|lost|declared|"
                r"reported|happened|occurred"
                r")\b",
                sentence,
                re.IGNORECASE,
            )
        )

        score = len(cats) * 2 + int(has_proper_noun) + int(has_action)

        if score > 0:
            candidates.append((score, sentence, cats))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)

        _, best_sentence, best_cats = candidates[0]

        return {
            "text": best_sentence,
            "entities": extract_entities(best_sentence),
            "categories": (best_cats or ["uncategorized"]),
            "derivation": "BEST_SCORED_FACTUAL_SENTENCE",
        }

    return None


def extract_claims(ocr_text: str, transcript: str, return_stats: bool = False):
    """
    Extract only externally verifiable claims.

    Opinion/advice/rhetorical sentences are retained in a separate
    excluded_claims list by verify_claims(), but are never sent to the
    external verification layer.
    """
    log.info("Extracting structured fact-checkable claims...")

    all_text = (f"{ocr_text}\n{transcript}").strip()

    if not all_text:
        empty_stats = {"ocr_claim_candidates": 0, "reliable_claims": 0, "filtered_claims": 0}
        return ([], empty_stats) if return_stats else []

    combined_entities = extract_entities(all_text)

    claims = []

    primary = detect_primary_claim(ocr_text, transcript, combined_entities)

    if primary:
        claims.append(
            {
                "text": primary["text"],
                "entities": primary["entities"],
                "categories": primary["categories"],
                "derivation": primary["derivation"],
                "is_primary": True,
                "claim_type": "FACTUAL_CLAIM",
                "is_fact_checkable": True,
            }
        )

    raw_candidates = _claim_candidates(ocr_text, transcript)

    for sentence in raw_candidates:

        if primary and sentence == primary["text"]:
            continue

        fact_checkable, reason = _is_fact_checkable_sentence(sentence)

        if not fact_checkable:
            continue

        cats = _classify_sentence(sentence)

        has_proper_noun = bool(re.search(r"\b[A-Z][a-z]+" r"(?:\s+[A-Z][a-z]+)*\b", sentence))

        has_action = bool(
            re.search(
                r"\b("
                r"said|announced|confirmed|denied|"
                r"killed|arrested|signed|launched|"
                r"attacked|died|won|lost|declared|"
                r"reported|happened|occurred"
                r")\b",
                sentence,
                re.IGNORECASE,
            )
        )

        # For a generic sentence, require a stronger factual signal.
        if not cats and not has_action:
            continue

        if (
            not has_proper_noun
            and not cats
            and not re.search(
                r"\b(yesterday|today|tomorrow|" r"last night|this morning|20\d{2})\b",
                sentence,
                re.IGNORECASE,
            )
        ):
            continue

        claims.append(
            {
                "text": sentence,
                "entities": extract_entities(sentence),
                "categories": cats,
                "derivation": "SECONDARY_FACTUAL_SENTENCE",
                "is_primary": False,
                "claim_type": "FACTUAL_CLAIM",
                "is_fact_checkable": True,
            }
        )

    seen = set()
    deduped = []

    for claim in claims:
        claim["text"] = _normalize_claim_text(claim["text"])
        normalized = _claim_key(claim["text"])

        if normalized in seen or any(
            _claims_are_duplicates(claim["text"], prior["text"]) for prior in deduped
        ):
            continue
        seen.add(normalized)
        deduped.append(claim)

    deduped = deduped[:MAX_CLAIMS]

    log.info(
        f"Extracted {len(deduped)} fact-checkable claims " f"(primary={'yes' if primary else 'no'})"
    )

    stats = {
        "ocr_claim_candidates": len(raw_candidates),
        "reliable_claims": len(deduped),
        "filtered_claims": max(0, len(raw_candidates) - len(deduped)),
    }
    return (deduped, stats) if return_stats else deduped


# ============================================================
# SEARCH QUERY GENERATION
# ============================================================

_STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "was",
    "were",
    "are",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "and",
    "or",
    "with",
    "that",
    "this",
    "it",
    "as",
    "by",
    "from",
    "has",
    "have",
    "had",
    "will",
    "would",
    "said",
}


def _keywords(text: str) -> set:

    text = text.lower().translate(str.maketrans("", "", string.punctuation))

    return {word for word in text.split() if (word not in _STOPWORDS and len(word) > 2)}


def build_search_queries(
    claim_text: str, entities: dict, max_queries: int = MAX_SEARCH_QUERIES_PER_CLAIM
) -> list:

    event = entities["events"][0] if entities.get("events") else None

    location = entities["locations"][0] if entities.get("locations") else None

    date = entities["dates"][0] if entities.get("dates") else None

    person = entities["persons"][0] if entities.get("persons") else None

    org = entities["organizations"][0] if entities.get("organizations") else None

    combos = []

    if event and location:
        combos.append(f"{event} {location}")

    if event and date:
        combos.append(f"{event} {date}")

    if event and person:
        combos.append(f"{event} {person}")

    if event and org:
        combos.append(f"{event} {org}")

    if location and date:
        combos.append(f"{location} {date}")

    if person and event:
        combos.append(f"{person} {event}")

    if org and event:
        combos.append(f"{org} {event}")

    # If structured combinations are weak,
    # use meaningful keywords from the claim.
    if not combos:

        keywords = sorted(_keywords(claim_text), key=len, reverse=True)[:8]

        if keywords:
            combos.append(" ".join(keywords))

    if not combos:
        combos.append(claim_text[:150])

    return list(dict.fromkeys(combos))[:max_queries]


# ============================================================
# URL / DOMAIN HELPERS
# ============================================================


def normalize_domain(url: str) -> str:

    if not url:
        return ""

    value = url.strip()

    if not value.startswith(("http://", "https://")):
        value = "https://" + value

    try:

        hostname = (urlparse(value).hostname or "").lower()

        if hostname.startswith("www."):
            hostname = hostname[4:]

        return hostname

    except Exception:
        return ""


def is_trusted_domain(domain: str) -> bool:

    normalized = normalize_domain(domain)

    if not normalized:
        return False

    for trusted in TRUSTED_DOMAINS:

        trusted = trusted.lower()

        if normalized == trusted or normalized.endswith("." + trusted):
            return True

    return False


# ============================================================
# DUCKDUCKGO SEARCH
# ============================================================


def _run_ddg_search(query: str, max_results: int) -> list:

    if not BS4_AVAILABLE:
        return []

    # PATCH (DDG CIRCUIT BREAKER): short-circuit immediately if the breaker
    # has already tripped — this is what turns "350 sequential timeout
    # hangs" into "1 warning message, then instant no-op for the rest of
    # the run."
    if _ddg_circuit_is_open():
        return []

    try:

        response = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers={"User-Agent": "Mozilla/5.0 " "(TrueLens fact-check bot)"},
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()
        _ddg_record_success()

        soup = BeautifulSoup(response.text, "html.parser")

        results = []

        for result in soup.select(".result"):

            title_tag = result.select_one(".result__title")

            if not title_tag:
                continue

            link_tag = title_tag.find("a") or result.select_one(".result__url")

            url = ""

            if link_tag:
                url = link_tag.get("href") or ""

            if not url:
                url_tag = result.select_one(".result__url")

                if url_tag:
                    url = url_tag.get("href") or url_tag.get_text(strip=True)

            snippet_tag = result.select_one(".result__snippet")

            title = title_tag.get_text(" ", strip=True)

            snippet = snippet_tag.get_text(" ", strip=True) if snippet_tag else ""

            domain = normalize_domain(url)

            if not domain:
                continue

            results.append(
                {
                    "title": title,
                    "snippet": snippet,
                    "url": url,
                    "domain": domain,
                    "trusted_domain": is_trusted_domain(domain),
                }
            )

            if len(results) >= max_results:
                break

        return results

    except requests.exceptions.HTTPError as e:
        status_code = getattr(e.response, "status_code", None)
        if status_code in {403, 429}:
            _ddg_record_failure(f"HTTP {status_code}")
        log.warning(f"DuckDuckGo search failed for '{query}': {e}")
        return []

    except (
        requests.exceptions.ConnectTimeout,
        requests.exceptions.ConnectionError,
        requests.exceptions.ReadTimeout,
    ) as e:
        # PATCH (DDG CIRCUIT BREAKER): these are connection-level failures
        # — the kind that mean "this host is unreachable from here," not
        # "no results for this query." Only these count toward tripping
        # the breaker.
        _ddg_record_failure(str(e))
        log.warning(f"DuckDuckGo connection failed for '{query}': {e}")
        return []

    except Exception as e:

        log.warning(f"DuckDuckGo search failed " f"for '{query}': {e}")

        return []


# ============================================================
# TRUSTED SOURCE SEARCH
# ============================================================


def search_trusted_sources(query: str, max_results: int = MAX_RESULTS_PER_QUERY) -> dict:

    if not BS4_AVAILABLE:

        return {"found": False, "results": [], "error": "beautifulsoup4 not installed"}

    # Search small groups of trusted domains. Results are still domain-filtered
    # below, but latency is bounded to two requests with the default domain set.
    aggregated = []
    seen_urls = set()
    requests_made = 0

    domain_groups = [
        TRUSTED_DOMAINS[index : index + TRUSTED_DOMAIN_GROUP_SIZE]
        for index in range(0, len(TRUSTED_DOMAINS), TRUSTED_DOMAIN_GROUP_SIZE)
    ]
    for domains in domain_groups:

        if len(aggregated) >= max_results:
            break

        # PATCH (DDG CIRCUIT BREAKER): stop looping remaining domains the
        # moment the breaker trips mid-loop, instead of finishing out all
        # 14 domains for this query even though every one will fail the
        # same way.
        if _ddg_circuit_is_open():
            break

        domain_filter = " OR ".join(f"site:{domain}" for domain in domains)
        per_domain_query = f"{query} ({domain_filter})"

        requests_made += 1
        results = _run_ddg_search(per_domain_query, max_results)

        for result in results:

            url = result.get("url", "")

            canonical = url.lower().strip()

            if canonical in seen_urls:
                continue

            if not is_trusted_domain(result.get("domain", "")):
                continue

            seen_urls.add(canonical)

            result["query_used"] = per_domain_query

            aggregated.append(result)

            if len(aggregated) >= max_results:
                break

    return {
        "found": bool(aggregated),
        "results": aggregated,
        "requests_made": requests_made,
        "status": "SEARCH_FAILED" if _ddg_circuit_is_open() and not aggregated else (
            "VERIFIED_RESULTS_FOUND" if aggregated else "INSUFFICIENT_EVIDENCE"
        ),
    }


# ============================================================
# GOOGLE FACT CHECK API
# ============================================================


def query_google_factcheck(claim_text: str, api_key: str) -> dict:

    if not api_key:
        return {"found": False, "results": [], "skipped": True}

    try:

        response = requests.get(
            "https://factchecktools.googleapis.com/v1alpha1/claims:search",
            params={"key": api_key, "query": claim_text, "languageCode": "en", "pageSize": 3},
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        data = response.json()

        results = []

        for item in data.get("claims", [])[:3]:

            review = item.get("claimReview", [{}])[0]

            publisher = review.get("publisher", {}).get("name", "Unknown")

            results.append(
                {
                    "claim_text": item.get("text", ""),
                    "verdict": review.get("textualRating", "Unknown"),
                    "source": publisher,
                    "url": review.get("url", ""),
                    "source_type": "fact_check_database",
                }
            )

        return {"found": bool(results), "results": results}

    except Exception as e:

        return {"found": False, "results": [], "error": str(e)}


# ============================================================
# NEWS API
# ============================================================


def query_news_api(claim_text: str, api_key: str) -> dict:

    if not api_key:
        return {"found": False, "articles": [], "skipped": True}

    try:

        response = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "apiKey": api_key,
                "q": " ".join(claim_text.split()[:8]),
                "sortBy": "relevancy",
                "pageSize": 3,
                "language": "en",
            },
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        articles = response.json().get("articles", [])

        clean_articles = []

        for article in articles[:3]:

            clean_articles.append(
                {
                    "title": article.get("title", ""),
                    "source": article.get("source", {}).get("name", "Unknown"),
                    "url": article.get("url", ""),
                    "description": article.get("description", ""),
                }
            )

        return {"found": bool(clean_articles), "articles": clean_articles}

    except Exception as e:

        return {"found": False, "articles": [], "error": str(e)}


# ============================================================
# EVIDENCE COMPARISON
# ============================================================

_NEGATION_PHRASES = [
    "false",
    "fake",
    "hoax",
    "debunked",
    "denies",
    "denied",
    "no evidence",
    "incorrect",
    "misleading",
    "fabricated",
    "untrue",
    "disputed",
    "disproven",
    "did not",
    "never happened",
    "out of context",
    "not true",
]

_SUPPORT_PHRASES = [
    "confirmed",
    "verified",
    "officially confirmed",
    "official statement",
    "according to police",
    "according to authorities",
    "officials said",
]


def _contains_phrase(text: str, phrases: list) -> bool:

    lowered = text.lower()

    return any(phrase in lowered for phrase in phrases)


def _text_similarity(claim_text: str, evidence_text: str) -> dict:

    if not evidence_text.strip():

        return {
            "verdict": "UNRELATED",
            "overlap_score": 0.0,
            "keyword_overlap": 0.0,
            "sequence_similarity": 0.0,
        }

    claim_kw = _keywords(claim_text)

    evidence_kw = _keywords(evidence_text)

    if claim_kw and evidence_kw:

        keyword_overlap = len(claim_kw & evidence_kw) / len(claim_kw | evidence_kw)

    else:
        keyword_overlap = 0.0

    seq_ratio = SequenceMatcher(None, claim_text.lower(), evidence_text.lower()).ratio()

    score = round(keyword_overlap * 0.7 + seq_ratio * 0.3, 3)

    evidence_lower = evidence_text.lower()

    has_negation = _contains_phrase(evidence_lower, _NEGATION_PHRASES)

    # IMPORTANT:
    # Words such as "confirmed" alone do NOT
    # prove the extracted claim is true.
    #
    # Therefore support language only becomes
    # useful when similarity is already strong.

    has_support = _contains_phrase(evidence_lower, _SUPPORT_PHRASES)

    if score < 0.15:

        verdict = "UNRELATED"

    elif has_negation and score >= 0.25:

        verdict = "CONTRADICTED"

    elif score >= 0.50 and has_support:

        verdict = "CORROBORATED"

    elif score >= 0.60:

        verdict = "CORROBORATED"

    elif score >= 0.25:

        verdict = "RELATED"

    else:

        verdict = "UNRELATED"

    return {
        "verdict": verdict,
        "overlap_score": score,
        "keyword_overlap": round(keyword_overlap, 3),
        "sequence_similarity": round(seq_ratio, 3),
    }


# ============================================================
# ENTITY COMPARISON
# ============================================================


def compare_entities(claim_entities: dict, evidence_entities: dict) -> dict:

    matches = []
    mismatches = []

    # Only call an entity a mismatch if BOTH
    # sides actually contain that entity type.
    #
    # Missing evidence is not the same thing
    # as contradictory evidence.

    for key in ["locations", "dates", "persons", "organizations"]:

        c_vals = {str(v).lower().strip() for v in claim_entities.get(key, []) if str(v).strip()}

        e_vals = {str(v).lower().strip() for v in evidence_entities.get(key, []) if str(v).strip()}

        if not c_vals or not e_vals:
            continue

        if c_vals & e_vals:
            matches.append(key)
        else:
            mismatches.append(key)

    # Numbers require more caution.
    c_nums = {n.get("value") for n in claim_entities.get("numbers", []) if isinstance(n, dict)}

    e_nums = {n.get("value") for n in evidence_entities.get("numbers", []) if isinstance(n, dict)}

    if c_nums and e_nums:

        if c_nums & e_nums:
            matches.append("numbers")
        else:
            mismatches.append("numbers")

    return {"matches": matches, "mismatches": mismatches}


# ============================================================
# CLAIM → EVIDENCE COMPARISON
# ============================================================


def compare_claim_to_evidence(
    claim: dict,
    evidence_text: str,
    evidence_source: str,
    evidence_url: str,
    source_type: str = "trusted_search",
) -> dict:

    text_result = _text_similarity(claim["text"], evidence_text)

    evidence_entities = extract_entities(evidence_text)

    entity_result = compare_entities(claim["entities"], evidence_entities)

    verdict = text_result["verdict"]

    # Entity mismatches are important,
    # but ONLY downgrade strong corroboration.
    if entity_result["mismatches"] and verdict == "CORROBORATED":
        verdict = "CONTRADICTED"

    return {
        "source": evidence_source,
        "source_type": source_type,
        "url": evidence_url,
        "trusted_domain": is_trusted_domain(evidence_source),
        "text_verdict": text_result["verdict"],
        "overlap_score": text_result["overlap_score"],
        "keyword_overlap": text_result["keyword_overlap"],
        "sequence_similarity": text_result["sequence_similarity"],
        "entity_matches": entity_result["matches"],
        "entity_mismatches": entity_result["mismatches"],
        "final_verdict": verdict,
    }


# ============================================================
# CLAIM VERIFICATION
# ============================================================


def verify_claims(claims: list) -> dict:

    if not claims:

        return {
            "claims_checked": 0,
            "verified_count": 0,
            "debunked_count": 0,
            "unverified_count": 0,
            "claim_results": [],
            "flags": [],
            "sources_used": [],
            "external_searches_performed": 0,
        }

    flags = []
    claim_results = []

    verified_count = 0
    debunked_count = 0

    all_sources_used = set()
    external_searches_performed = 0
    search_cache = {}

    for claim in claims:

        evidence_comparisons = []
        search_statuses = []

        # ----------------------------------------------------
        # Trusted web sources
        # ----------------------------------------------------

        queries = build_search_queries(claim["text"], claim["entities"])

        for query in queries:

            cache_key = _claim_key(query)
            if cache_key in search_cache:
                trusted = search_cache[cache_key]
            else:
                trusted = search_trusted_sources(query, max_results=MAX_RESULTS_PER_QUERY)
                search_cache[cache_key] = trusted
                external_searches_performed += int(trusted.get("requests_made", 0) or 0)
            search_statuses.append(trusted.get("status", "INSUFFICIENT_EVIDENCE"))

            for result in trusted.get("results", []):

                evidence_text = f"{result.get('title', '')} " f"{result.get('snippet', '')}"

                comp = compare_claim_to_evidence(
                    claim=claim,
                    evidence_text=evidence_text,
                    evidence_source=result.get("domain", "unknown"),
                    evidence_url=result.get("url", ""),
                    source_type="trusted_web_search",
                )

                evidence_comparisons.append(comp)

                if result.get("domain"):
                    all_sources_used.add(result["domain"])

        # ----------------------------------------------------
        # Google Fact Check
        # ----------------------------------------------------

        if GOOGLE_FACTCHECK_API_KEY:

            external_searches_performed += 1
            factcheck = query_google_factcheck(claim["text"], GOOGLE_FACTCHECK_API_KEY)

            for item in factcheck.get("results", []):

                evidence_text = f"{item.get('claim_text', '')} " f"{item.get('verdict', '')}"

                comp = compare_claim_to_evidence(
                    claim=claim,
                    evidence_text=evidence_text,
                    evidence_source=item.get("source", "unknown"),
                    evidence_url=item.get("url", ""),
                    source_type="fact_check_database",
                )

                evidence_comparisons.append(comp)

                all_sources_used.add(item.get("source", "unknown"))

        # ----------------------------------------------------
        # News API
        # ----------------------------------------------------

        if NEWS_API_KEY:

            external_searches_performed += 1
            news_result = query_news_api(claim["text"], NEWS_API_KEY)

            for article in news_result.get("articles", []):

                evidence_text = f"{article.get('title', '')} " f"{article.get('description', '')}"

                comp = compare_claim_to_evidence(
                    claim=claim,
                    evidence_text=evidence_text,
                    evidence_source=article.get("source", "unknown"),
                    evidence_url=article.get("url", ""),
                    source_type="news_api",
                )

                evidence_comparisons.append(comp)

                all_sources_used.add(article.get("source", "unknown"))

        # ----------------------------------------------------
        # Aggregate evidence
        # ----------------------------------------------------

        corroborated = [c for c in evidence_comparisons if c["final_verdict"] == "CORROBORATED"]

        contradicted = [c for c in evidence_comparisons if c["final_verdict"] == "CONTRADICTED"]

        related = [c for c in evidence_comparisons if c["final_verdict"] == "RELATED"]

        # ----------------------------------------------------
        # Stronger aggregation logic
        # ----------------------------------------------------

        # Multiple independent corroborating sources
        # are stronger than a single matching result.
        unique_corroborating_sources = {c["source"] for c in corroborated if c.get("source")}

        unique_contradicting_sources = {c["source"] for c in contradicted if c.get("source")}

        if unique_corroborating_sources and unique_contradicting_sources:

            verdict = "CONFLICTING_EVIDENCE"

        elif len(unique_contradicting_sources) >= 1 and not unique_corroborating_sources:

            verdict = "DEBUNKED"
            debunked_count += 1

        elif len(unique_corroborating_sources) >= 2:

            verdict = "VERIFIED"
            verified_count += 1

        elif len(unique_corroborating_sources) == 1 and not contradicted:

            # One source is not enough for a strong
            # VERIFIED classification.
            verdict = "PARTIALLY_SUPPORTED"

        elif related:

            verdict = "PARTIALLY_SUPPORTED"

        elif search_statuses and all(status == "SEARCH_FAILED" for status in search_statuses):

            verdict = "SEARCH_FAILED"

        else:

            verdict = "INSUFFICIENT_EVIDENCE"

        # ----------------------------------------------------
        # Evidence confidence
        # ----------------------------------------------------

        source_count = len({c.get("source") for c in evidence_comparisons if c.get("source")})

        if verdict == "VERIFIED":

            confidence = min(0.95, 0.55 + 0.10 * min(source_count, 4))

        elif verdict == "DEBUNKED":

            confidence = min(0.95, 0.60 + 0.08 * min(source_count, 4))

        elif verdict == "CONFLICTING_EVIDENCE":

            confidence = 0.50

        elif verdict == "PARTIALLY_SUPPORTED":

            confidence = 0.40

        else:

            confidence = 0.0

        confidence = round(confidence, 2)

        # ----------------------------------------------------
        # Reasoning
        # ----------------------------------------------------

        reasoning = []

        if contradicted:

            best = max(contradicted, key=lambda c: c["overlap_score"])

            reasoning.append("Contradictory evidence found from " f"{best['source']}.")

            flags.append(
                "CLAIM_DEBUNKED: "
                f"'{claim['text'][:100]}...' — "
                f"contradicted by {best['source']}"
            )

        if corroborated:

            best = max(corroborated, key=lambda c: c["overlap_score"])

            reasoning.append("Corroborating evidence found from " f"{best['source']}.")

        if len(unique_corroborating_sources) == 1:

            reasoning.append(
                "Only one corroborating source "
                "was found; treated as partial support "
                "rather than definitive verification."
            )

        if verdict == "CONFLICTING_EVIDENCE":

            reasoning.append(
                "Trusted-source evidence contains " "both supporting and contradictory signals."
            )

            flags.append(
                "CONFLICTING_EVIDENCE: "
                f"Claim '{claim['text'][:100]}...' "
                "has both corroborating and contradictory evidence."
            )

        if verdict == "SEARCH_FAILED":

            reasoning.append("External search failed; no authenticity inference was made.")
            flags.append(f"SEARCH_FAILED: '{claim['text'][:100]}...'")

        elif not evidence_comparisons:

            reasoning.append(
                "No matching evidence was found " "from the configured external sources."
            )

            flags.append(
                "INSUFFICIENT_EVIDENCE: "
                f"'{claim['text'][:100]}...' — "
                "no trusted-source matches found."
            )

        elif verdict == "INSUFFICIENT_EVIDENCE":

            reasoning.append(
                "Search results were insufficient " "to establish or contradict the claim."
            )

        # ----------------------------------------------------
        # Store claim result
        # ----------------------------------------------------

        claim_results.append(
            {
                "claim": claim["text"],
                "claim_type": claim.get("claim_type", "FACTUAL_CLAIM"),
                "is_fact_checkable": claim.get("is_fact_checkable", True),
                "is_primary_claim": claim.get("is_primary", False),
                "categories": claim.get("categories", []),
                "matched_entities": claim.get("entities", {}),
                "evidence_sources": list(
                    dict.fromkeys(c["source"] for c in evidence_comparisons if c.get("source"))
                ),
                "evidence_comparisons": evidence_comparisons,
                "verdict": verdict,
                "search_status": (
                    "SEARCH_FAILED" if verdict == "SEARCH_FAILED" else (
                        "EVIDENCE_FOUND" if evidence_comparisons else "INSUFFICIENT_EVIDENCE"
                    )
                ),
                "structured_errors": (
                    [{
                        "status": "SEARCH_FAILED",
                        "reason": "External search provider unavailable",
                        "claim": claim["text"],
                    }]
                    if verdict == "SEARCH_FAILED" else []
                ),
                "confidence": confidence,
                "reasoning": reasoning,
            }
        )

        log.info(
            f"Claim verdict: {verdict} "
            f"(confidence={confidence}) — "
            f"'{claim['text'][:80]}...'"
        )

    return {
        "claims_checked": len(claims),
        "verified_count": verified_count,
        "debunked_count": debunked_count,
        "unverified_count": len(claims) - verified_count - debunked_count,
        "claim_results": claim_results,
        "flags": flags,
        "sources_used": sorted(all_sources_used),
        "external_searches_performed": external_searches_performed,
    }


# ============================================================
# CHEAPFAKE / MISCONTEXTUALIZATION DETECTION
# ============================================================

_STALE_REUSE_PHRASES = [
    "throwback",
    "file photo",
    "stock photo",
    "archive footage",
    "archival footage",
    "this happened in",
    "flashback to",
    "old footage",
]

_URGENCY_PHRASES = [
    "happening now",
    "just happened",
    "breaking news",
    "watch before",
    "they don't want you to see",
    "share before deleted",
    "leaked footage",
    "exclusive footage",
    "unedited",
    "real footage",
    "right now",
    "today",
]


def detect_cheapfake(ocr_text: str, transcript: str) -> dict:
    log.info("Running cheapfake / " "miscontextualization detection...")

    flags = []

    all_text = (f"{ocr_text} {transcript}").strip().lower()

    found_phrases = [phrase for phrase in _URGENCY_PHRASES if phrase in all_text]

    if found_phrases:
        flags.append(
            "CHEAPFAKE_LANGUAGE: " "Urgency/manipulation phrases found: " f"{found_phrases}"
        )

    # An "extraordinary claim" must be an actual event assertion.
    # Merely mentioning words such as "attack" or "war" in a hypothetical,
    # discussion, quotation, or general speech must not trigger it.
    extraordinary_event_patterns = [
        r"\b(?:an?|the)\s+(?:explosion|attack|assassination|"
        r"disaster|war|coup)\s+(?:happened|occurred|"
        r"took place|is happening|has happened)\b",
        r"\b\d+\s+(?:people|persons)\s+" r"(?:were\s+)?(?:killed|injured)\b",
        r"\b(?:was|were|has been|have been)\s+" r"(?:killed|assassinated|bombed|attacked)\b",
    ]

    has_extraordinary = any(
        re.search(pattern, all_text, re.IGNORECASE) for pattern in extraordinary_event_patterns
    )

    current_year = datetime.now(timezone.utc).year

    past_years = [
        int(year) for year in re.findall(r"\b(20\d{2})\b", all_text) if int(year) < current_year - 1
    ]

    if past_years:
        flags.append(
            "OLD_DATE_IN_TEXT: "
            f"Text mentions past year(s) "
            f"{past_years} — media may be older "
            "than the current context."
        )

    reuse_hits = [phrase for phrase in _STALE_REUSE_PHRASES if phrase in all_text]

    if reuse_hits:
        flags.append(
            "STALE_MEDIA_LANGUAGE: " "Text references old/archive footage: " f"{reuse_hits}"
        )

    if past_years and found_phrases:
        flags.append(
            "STALE_MEDIA_REPOSTED_AS_CURRENT: "
            f"Old date(s) {past_years} appear alongside "
            f"current/urgent framing {found_phrases}."
        )

    ocr_entities = extract_entities(ocr_text)

    transcript_entities = extract_entities(transcript)

    contradiction = compare_entities(ocr_entities, transcript_entities)

    if contradiction["mismatches"]:
        flags.append(
            "OCR_TRANSCRIPT_CONTRADICTION: "
            "On-screen text and spoken audio disagree "
            f"on {contradiction['mismatches']}."
        )

    caption_only_appeal = _contains_religious_appeal(all_text) and not any(
        re.search(r"\b" + re.escape(kw) + r"\b", all_text) for kw in _EVENT_KEYWORDS
    )

    if caption_only_appeal:
        flags.append(
            "CAPTION_LACKS_CONTEXT: "
            "Text appears primarily emotional/religious "
            "and contains no identifiable factual event."
        )

    return {
        "cheapfake_phrases_found": found_phrases,
        "extraordinary_claim_detected": has_extraordinary,
        "old_dates_in_text": past_years,
        "stale_media_language_found": reuse_hits,
        "caption_lacks_context": caption_only_appeal,
        "ocr_transcript_mismatches": contradiction["mismatches"],
        "flags": flags,
    }


# ============================================================
# IMPORTANT:
# NO CROSSREF TRUST SCORE
# ============================================================


def compute_crossref_score(
    claim_verification: dict, cheapfake_result: dict, visual_similarity_result: dict | None = None
) -> tuple[int, list]:
    """
    Compute the Cross-Reference evidence score.

    This is NOT the final media-trust score. The Decision Engine should
    still fuse Cross-Reference with Forensics and Provenance.

    Scoring policy:
      - 0 fact-checkable claims checked -> 50 (neutral)
      - claims checked but unresolved -> 50 (neutral)
      - each debunked claim -> -25
      - each verified claim -> +15
      - bounded to 0..100

    NOT_CHECKED visual similarity does not change the score because it has
    no evidentiary weight.
    """
    claims_checked = int(claim_verification.get("claims_checked", 0) or 0)

    verified_count = int(claim_verification.get("verified_count", 0) or 0)

    debunked_count = int(claim_verification.get("debunked_count", 0) or 0)

    score_reasoning = []

    if claims_checked == 0:
        score = 50
        score_reasoning.append(
            "CROSSREF_NEUTRAL: No fact-checkable claims were " "checked — scored as neutral (50)."
        )
    else:
        score = 50
        score_reasoning.append(
            "CROSSREF_ACTIVE_NEUTRAL: Fact-checkable claims were checked, but checking alone "
            "is not supporting evidence — baseline remains neutral (50)."
        )

        if debunked_count:
            penalty = debunked_count * 25
            score -= penalty
            score_reasoning.append(f"DEBUNKED_PENALTY: -{penalty} " f"({debunked_count} × 25).")

        if verified_count:
            bonus = verified_count * 15
            score += bonus
            score_reasoning.append(f"VERIFIED_BONUS: +{bonus} " f"({verified_count} × 15).")

    score = max(0, min(100, int(score)))

    if visual_similarity_result:
        visual_verdict = visual_similarity_result.get("verdict")

        if visual_verdict == "NOT_CHECKED":
            score_reasoning.append(
                "VISUAL_SIMILARITY_NEUTRAL: Visual similarity "
                "was not checked against a populated database; "
                "no score impact."
            )
        elif visual_verdict == "NEAR_DUPLICATE_FOUND":
            score_reasoning.append(
                "VISUAL_SIMILARITY_MATCH: Near-duplicate evidence "
                "was found; surfaced for Decision Engine review."
            )

    return score, score_reasoning


# ============================================================
# MASTER CROSS-REFERENCE AGENT
# ============================================================


def run_crossref_agent(provenance_output: dict, forensics_output: dict) -> dict:

    log.info("=" * 60)

    try:
        provenance_output = _coerce_agent_dict(provenance_output, "provenance_output")
        forensics_output = _coerce_agent_dict(forensics_output, "forensics_output")
    except (TypeError, ValueError) as exc:
        return {
            "status": "error",
            "agent": "cross_reference",
            "error_message": str(exc),
            "ready_for_next_agent": False,
        }
    log.info("CROSS-REFERENCE AGENT STARTING")
    log.info("=" * 60)

    # --------------------------------------------------------
    # Validate Forensics input
    # --------------------------------------------------------

    if not isinstance(forensics_output, dict):

        return {
            "status": "error",
            "error_message": "Invalid forensics_output: " "expected dictionary.",
            "ready_for_next_agent": False,
        }

    # --------------------------------------------------------
    # Forensic routing
    # --------------------------------------------------------

    skip_crossref, routing_reason = should_skip_cross_reference(forensics_output)

    if skip_crossref:

        log.info("Cross-Reference skipped: " + routing_reason)

        return {
            "status": "skipped",
            "reason": routing_reason,
            "routing": "DECISION_ENGINE",
            "ready_for_next_agent": True,
            # Crossref does NOT score skipped media.
            "crossref_score": None,
            "claims_checked": 0,
            "claim_results": [],
            "sources_used": [],
            "reasoning": [routing_reason],
            "flags": [routing_reason],
            "cheapfake_analysis": None,
            "visual_similarity": None,
            "ocr_text": "",
            "transcript": "",
            "language_detected": None,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }

    # --------------------------------------------------------
    # File path
    # --------------------------------------------------------

    file_path = forensics_output.get("file_path") or provenance_output.get("file_path")

    if not file_path:

        return {
            "status": "error",
            "error_message": "No file_path available " "from Forensics or Provenance output.",
            "ready_for_next_agent": False,
        }

    if not os.path.exists(file_path):

        return {
            "status": "error",
            "error_message": f"Media file does not exist: " f"{file_path}",
            "file_path": file_path,
            "ready_for_next_agent": False,
        }

    media_type = (
        forensics_output.get("media_type")
        or provenance_output.get("media_type")
        or detect_media_type(file_path)
    )

    # --------------------------------------------------------
    # Main processing
    # --------------------------------------------------------

    try:

        all_flags = []

        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        ocr_result = extract_text_via_ocr(file_path, media_type)

        all_flags.extend(ocr_result.get("flags", []))

        ocr_text = ocr_result.get("ocr_text", "")

        # ----------------------------------------------------
        # Transcription
        # ----------------------------------------------------

        transcript_result = {
            "transcript": "",
            "language": None,
            "flags": [],
            "module_status": "not_applicable",
        }

        if media_type == "video":

            transcript_result = transcribe_audio(file_path, media_type)

            all_flags.extend(transcript_result.get("flags", []))

        transcript = transcript_result.get("transcript", "")

        # ----------------------------------------------------
        # PATCH: Visual similarity — runs independently of
        # OCR/transcript, so it still produces a result when
        # there is zero text on the media.
        # ----------------------------------------------------

        visual_source_path = (
            provenance_output.get("source_file_path") if media_type == "image" else file_path
        )
        if not visual_source_path or not os.path.exists(visual_source_path):
            visual_source_path = file_path
        visual_similarity_result = check_visual_similarity(
            visual_source_path, media_type, known_hash_db=KNOWN_IMAGE_HASH_DB
        )

        all_flags.extend(visual_similarity_result.get("flags", []))

        # ----------------------------------------------------
        # Claims
        # ----------------------------------------------------

        claims, claim_filter_stats = extract_claims(ocr_text, transcript, return_stats=True)

        if not claims:

            claim_verification = {
                "claims_checked": 0,
                "verified_count": 0,
                "debunked_count": 0,
                "unverified_count": 0,
                "claim_results": [],
                "flags": [],
                "sources_used": [],
                "external_searches_performed": 0,
            }

            # PATCH: clarified that this is specifically about
            # text-based claim checking having nothing to work
            # with — visual_similarity above is a separate,
            # independently-run module and is NOT covered by
            # this limitation.
            all_flags.append(
                "NO_FACTUAL_CLAIM_EXTRACTED: "
                "No fact-checkable factual claim was extracted "
                "from available OCR/transcript text. Opinion, "
                "advice, and rhetorical statements are excluded "
                "from external verification. See visual_similarity "
                "for the independent perceptual-hash check."
            )

        else:

            claim_verification = verify_claims(claims)

            all_flags.extend(claim_verification.get("flags", []))

        # ----------------------------------------------------
        # Cheapfake analysis
        # ----------------------------------------------------

        cheapfake_result = detect_cheapfake(ocr_text, transcript)

        all_flags.extend(cheapfake_result.get("flags", []))

        # ----------------------------------------------------
        # Build final evidence output
        # ----------------------------------------------------

        crossref_score, crossref_score_reasoning = compute_crossref_score(
            claim_verification, cheapfake_result, visual_similarity_result
        )

        result = {
            "status": "completed",
            "agent": "cross_reference",
            "file_path": file_path,
            "media_type": media_type,
            # Evidence score only; Decision Engine owns final fusion.
            "crossref_score": crossref_score,
            "crossref_note": "CROSSREF_NEUTRAL" if crossref_score == 50 else "CROSSREF_EVIDENCE",
            "crossref_score_reasoning": crossref_score_reasoning,
            "claims_checked": claim_verification["claims_checked"],
            "claim_results": claim_verification["claim_results"],
            "verified_count": claim_verification["verified_count"],
            "debunked_count": claim_verification["debunked_count"],
            "unverified_count": claim_verification["unverified_count"],
            "sources_used": claim_verification.get("sources_used", []),
            "external_searches_performed": claim_verification.get(
                "external_searches_performed", 0
            ),
            "reasoning": all_flags,
            "flags": all_flags,
            "cheapfake_analysis": cheapfake_result,
            # PATCH: new independent field
            "visual_similarity": visual_similarity_result,
            "ocr_text": ocr_text,
            "transcript": transcript,
            "language_detected": transcript_result.get("language"),
            "routing_reason": routing_reason,
            "modules": {
                "ocr": ocr_result.get("module_status"),
                "transcription": transcript_result.get("module_status"),
                "claim_extraction": "ok" if claims else "no_fact_checkable_claims",
                "external_verification": "ok" if claims else "not_applicable",
                "cheapfake_analysis": "ok",
                # PATCH: new module status entry
                "visual_similarity": visual_similarity_result.get("module_status", "unknown"),
            },
            "analysis_summary": {
                "claims_found": len(claims),
                "ocr_claim_candidates_before_filtering": claim_filter_stats[
                    "ocr_claim_candidates"
                ],
                "reliable_claims_after_filtering": claim_filter_stats["reliable_claims"],
                "claims_filtered_as_noise_or_duplicates": claim_filter_stats["filtered_claims"],
                "fact_checkable_claims_checked": claim_verification.get("claims_checked", 0),
                "external_searches_performed": claim_verification.get(
                    "external_searches_performed", 0
                ),
                "primary_claim_found": any(c.get("is_primary", False) for c in claims),
                "external_evidence_found": bool(claim_verification.get("sources_used")),
                "cheapfake_flags_found": len(cheapfake_result.get("flags", [])),
                # PATCH: new summary field
                "visual_similarity_verdict": visual_similarity_result.get("verdict", "NOT_CHECKED"),
            },
            "ready_for_next_agent": True,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }

        log.info(
            "Cross-Reference complete — "
            f"claims_checked="
            f"{result['claims_checked']}, "
            f"visual_similarity="
            f"{visual_similarity_result.get('verdict')}"
        )

        return result

    except Exception as e:

        log.error("Cross-Reference Agent FAILED: " f"{e}")

        import traceback

        traceback.print_exc()

        return {
            "status": "error",
            "agent": "cross_reference",
            "error_message": str(e),
            "file_path": file_path,
            "crossref_score": None,
            "claims_checked": 0,
            "claim_results": [],
            "verified_count": 0,
            "debunked_count": 0,
            "unverified_count": 0,
            "sources_used": [],
            "flags": ["CROSSREF_AGENT_ERROR: " + str(e)],
            "cheapfake_analysis": None,
            "visual_similarity": None,
            "ready_for_next_agent": False,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }


# ============================================================
# Convert JSON strings → Python dictionaries
if __name__ == "__main__":
    import sys

    from forensic import run_forensics_agent
    from ingestor import run_ingestor_agent
    from provinence import run_provenance_agent

    local_image = sys.argv[1] if len(sys.argv) > 1 else "imageprocessor.jpeg"
    ingestor_result = run_ingestor_agent(local_image)
    provenance_result = run_provenance_agent(ingestor_result)
    forensics_result = run_forensics_agent(provenance_result)
    crossref_result = run_crossref_agent(provenance_result, forensics_result)
    print(json.dumps(crossref_result, indent=2, default=str))
