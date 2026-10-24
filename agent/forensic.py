# !pip install exifread mutagen imagehash
# ============================================================
# THE TRUE LENS — FORENSICS AGENT (agent3) — PATCHED VERSION
# future work
# eyes_detected: 0 in the face analysis — that's still the old Haar eye cascade, and it found nothing (no eyes visible/open at this angle, or just a miss). Because eye-catchlight and eye-color-consistency checks require 2 detected eyes to run, they never fired here. That's not a bug, just a coverage gap: you're not getting a false signal, you're getting no signal from that specific check. If you want to push further, the same YuNet-style upgrade could apply to eye/landmark detection (YuNet's DNN cousins or a proper 5-point/68-point landmark model would find eyes far more reliably than Haar on a partially-submerged, tilted face).

import os
import json
import logging
import tempfile
import subprocess
import urllib.request
import numpy as np
from datetime import datetime
from pathlib import Path

import cv2  # OpenCV for image/video analysis
from PIL import Image, ImageChops, ImageEnhance
import scipy.stats as stats  # For statistical analysis

# Audio analysis
try:
    import librosa
    import librosa.display

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("WARNING: librosa not installed. Audio analysis will be skipped.")
    print("Run: pip install librosa")

# Perceptual hashing (PATCH 4)
try:
    import imagehash

    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False
    print("WARNING: imagehash not installed. Perceptual hash lookup will be skipped.")
    print("Run: pip install imagehash")

# ── LOGGING SETUP ──
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [FORENSICS] %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


def _coerce_agent_dict(value, boundary_name: str) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{boundary_name} must be a dict or valid JSON object: {exc}") from exc
        if isinstance(decoded, dict):
            return decoded
    raise TypeError(f"{boundary_name} must be a dictionary")


# ============================================================
# PATCH 1-fix: YuNet model download was pulling a Git LFS *pointer* file.
# ============================================================

_YUNET_MODEL_PATH = "face_detection_yunet_2023mar.onnx"

# Primary: GitHub's LFS media-resolving host (serves the real binary).
# Fallback: a non-LFS mirror, in case media.githubusercontent.com is
# blocked in your network environment.
_YUNET_MODEL_URLS = [
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
]

# A real YuNet ONNX file is ~232KB. An LFS pointer text file is ~130 bytes.
# Anything under this is almost certainly a pointer/error page, not a model.
_YUNET_MIN_VALID_SIZE_BYTES = 50_000

_yunet_detector = None
_yunet_download_attempted = False


def _validate_onnx_download(path: str) -> bool:
    """Returns True only if the file looks like a real binary ONNX model
    rather than a Git LFS pointer or an HTML error page."""
    if not os.path.exists(path):
        return False
    size = os.path.getsize(path)
    if size < _YUNET_MIN_VALID_SIZE_BYTES:
        return False
    # LFS pointer files and HTML error pages are both plain text starting
    # with ASCII; a real ONNX file starts with protobuf binary bytes.
    try:
        with open(path, "rb") as f:
            head = f.read(20)
        if (
            head.startswith(b"version https://git-lfs")
            or head.startswith(b"<!DOCTYPE")
            or head.startswith(b"<html")
        ):
            return False
    except Exception:
        return False
    return True


def _ensure_yunet_model():
    global _yunet_download_attempted
    if os.path.exists(_YUNET_MODEL_PATH) and _validate_onnx_download(_YUNET_MODEL_PATH):
        return
    if _yunet_download_attempted:
        return
    _yunet_download_attempted = True

    # Clear out any previously-cached bad download (e.g. an LFS pointer
    # saved under this filename by a prior run) so we don't skip re-fetching.
    if os.path.exists(_YUNET_MODEL_PATH) and not _validate_onnx_download(_YUNET_MODEL_PATH):
        log.warning(
            f"Existing {_YUNET_MODEL_PATH} failed validation (likely a Git LFS pointer, not the real model) — removing and retrying download."
        )
        try:
            os.remove(_YUNET_MODEL_PATH)
        except Exception:
            pass

    for url in _YUNET_MODEL_URLS:
        try:
            log.info(f"Downloading YuNet face detection model from {url} ...")
            tmp_path = _YUNET_MODEL_PATH + ".tmp"
            urllib.request.urlretrieve(url, tmp_path)
            if _validate_onnx_download(tmp_path):
                os.replace(tmp_path, _YUNET_MODEL_PATH)
                log.info(
                    f"YuNet model downloaded successfully ({os.path.getsize(_YUNET_MODEL_PATH)} bytes)."
                )
                return
            else:
                log.warning(
                    f"Downloaded file from {url} failed validation (likely LFS pointer/HTML error page) — trying next source."
                )
                os.remove(tmp_path)
        except Exception as e:
            log.warning(f"Download from {url} failed: {e}")

    log.warning(
        "All YuNet download sources failed — will fall back to Haar cascade chain for this session."
    )


def _get_yunet(input_size=(320, 320)):
    global _yunet_detector
    if _yunet_detector is not None:
        return _yunet_detector
    _ensure_yunet_model()
    if not os.path.exists(_YUNET_MODEL_PATH) or not _validate_onnx_download(_YUNET_MODEL_PATH):
        return None
    try:
        _yunet_detector = cv2.FaceDetectorYN.create(
            _YUNET_MODEL_PATH, "", input_size, score_threshold=0.6, nms_threshold=0.3, top_k=50
        )
    except Exception as e:
        log.warning(
            f"YuNet init failed even after validated download, will fall back to Haar chain: {e}"
        )
        _yunet_detector = None
    return _yunet_detector


# ============================================================
# CASCADE LOADER — self-healing + graceful fallback (UNCHANGED)
# ============================================================

_CASCADE_CACHE = {}
_SELF_HEAL_ATTEMPTED = False


def _get_cascade(cascade_filename: str):
    """Lazily loads and caches a Haar cascade. Returns None (not an
    exception) if no working cascade could be obtained, so callers can
    fall back cleanly instead of crashing the module."""
    global cv2, _SELF_HEAL_ATTEMPTED

    if cascade_filename in _CASCADE_CACHE:
        return _CASCADE_CACHE[cascade_filename]

    if not hasattr(cv2, "CascadeClassifier") and not _SELF_HEAL_ATTEMPTED:
        _SELF_HEAL_ATTEMPTED = True
        log.warning(
            "cv2.CascadeClassifier is unavailable; continuing with YuNet and the "
            "non-Haar fallback. Install opencv-python-headless<5 to enable Haar cascades."
        )

    clf = None
    if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data"):
        try:
            cascade_path = os.path.join(
                cv2.data.haarcascades, cascade_filename)
            candidate = cv2.CascadeClassifier(cascade_path)
            if candidate is not None and not candidate.empty():
                clf = candidate
        except Exception as e:
            log.warning(
                f"CascadeClassifier load failed for {cascade_filename}: {e}")

    _CASCADE_CACHE[cascade_filename] = clf
    return clf


def _fallback_face_regions(color_img) -> list:
    """
    Haar-cascade-free face-region estimator, used ONLY when no working
    CascadeClassifier could be obtained. Unions two independent skin-tone
    models (HSV + YCrCb) and texture-gates candidates so recall gains
    don't inflate false positives.
    """
    hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(
        hsv, np.array([0, 30, 60], dtype=np.uint8), np.array(
            [25, 180, 255], dtype=np.uint8)
    )

    ycrcb = cv2.cvtColor(color_img, cv2.COLOR_BGR2YCrCb)
    ycrcb_mask = cv2.inRange(
        ycrcb, np.array([0, 133, 77], dtype=np.uint8), np.array(
            [255, 173, 127], dtype=np.uint8)
    )

    mask = cv2.bitwise_or(hsv_mask, ycrcb_mask)
    mask = cv2.medianBlur(mask, 7)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    img_area = color_img.shape[0] * color_img.shape[1]
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        aspect = h / (w + 1e-6)
        if area < 0.006 * img_area or not (0.8 <= aspect <= 2.0):
            continue
        roi = gray[y: y + h, x: x + w]
        if roi.size == 0:
            continue
        if float(cv2.Laplacian(roi, cv2.CV_64F).var()) < 15:
            continue
        boxes.append((int(x), int(y), int(w), int(h)))

    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes


def _detect_faces_haar_chain(color_img) -> tuple:
    """Original Haar-primary chain, kept intact as the fallback tier
    under the new YuNet-primary detector (see _detect_faces below)."""
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    # Small faces are common in news and disaster imagery.  Try the original
    # resolution first, then one moderate upscale; this improves recall without
    # weakening the cascade's neighbour threshold (which would add many false
    # positives).  Boxes are mapped back to source-image coordinates.
    attempts = [(gray, 1.0)]
    shortest_side = min(gray.shape[:2])
    if shortest_side < 900:
        upscale = min(2.0, 900.0 / max(shortest_side, 1))
        attempts.append(
            (cv2.resize(gray, None, fx=upscale, fy=upscale,
                        interpolation=cv2.INTER_CUBIC), upscale)
        )

    primary = _get_cascade("haarcascade_frontalface_default.xml")
    if primary is not None:
        for attempt, scale in attempts:
            faces = primary.detectMultiScale(
                attempt, scaleFactor=1.1, minNeighbors=5,
                minSize=(max(30, int(50 * scale)),) * 2)
            if len(faces) > 0:
                boxes = [tuple(int(round(v / scale)) for v in f) for f in faces]
                method = "haar_default" if scale == 1.0 else "haar_default_upscaled"
                return boxes, method, "high"

    secondary = _get_cascade("haarcascade_frontalface_alt2.xml")
    if secondary is not None:
        for attempt, scale in attempts:
            faces = secondary.detectMultiScale(
                attempt, scaleFactor=1.1, minNeighbors=5,
                minSize=(max(30, int(50 * scale)),) * 2)
            if len(faces) > 0:
                boxes = [tuple(int(round(v / scale)) for v in f) for f in faces]
                method = "haar_alt2" if scale == 1.0 else "haar_alt2_upscaled"
                return boxes, method, "high"

    fallback_boxes = _fallback_face_regions(color_img)
    if fallback_boxes:
        return fallback_boxes, "fallback_skin_color", "low"

    return [], "none", "none"


def _detect_faces(color_img) -> tuple:
    """
    PATCH 1: Detection chain reordered by actual reliability:

        YuNet DNN (handles occlusion/pose/harsh lighting well)
              -> Haar frontal_default
              -> Haar frontal_alt2
              -> skin-tone/contour fallback
              -> nothing found

    Returns (boxes, method, confidence). Boxes stay in the same
    (x, y, w, h) int format the rest of the pipeline already expects, so
    no other module needs to change.
    """
    h_img, w_img = color_img.shape[:2]
    detector = _get_yunet((w_img, h_img))
    if detector is not None:
        try:
            detector.setInputSize((w_img, h_img))
            _, faces = detector.detect(color_img)
            if faces is not None and len(faces) > 0:
                boxes = []
                for f in faces:
                    x, y, w, h = f[:4]
                    x, y = max(0, int(x)), max(0, int(y))
                    w, h = int(w), int(h)
                    boxes.append((x, y, w, h))
                return boxes, "yunet_dnn", "high"
        except Exception as e:
            log.warning(
                f"YuNet detection failed, falling back to Haar chain: {e}")

    return _detect_faces_haar_chain(color_img)


# ============================================================
# MODULE 1: ERROR LEVEL ANALYSIS (ELA)
# PATCH 2: now format-aware — see get_source_format_info() and
# run_ela_analysis_v2() below, which wrap this unchanged core function.
# ============================================================


def run_ela_analysis(image_path: str, quality: int = 90) -> dict:
    log.info("Running Error Level Analysis (ELA)...")

    try:
        original = Image.open(image_path).convert("RGB")

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        original.save(tmp_path, "JPEG", quality=quality)

        compressed = Image.open(tmp_path).convert("RGB")

        ela_image = ImageChops.difference(original, compressed)
        ela_array = np.array(ela_image)

        scale = 10
        ela_array_scaled = np.clip(ela_array * scale, 0, 255)

        ela_max = int(ela_array_scaled.max())
        ela_mean = float(ela_array_scaled.mean())

        threshold = 50
        suspicious_pixels = (ela_array_scaled > threshold).sum()
        total_pixels = ela_array_scaled.size
        suspicious_percent = (suspicious_pixels / total_pixels) * 100

        os.unlink(tmp_path)

        flags = []
        if ela_mean > 15:
            flags.append(
                f"HIGH_ELA_MEAN: Average error level is {ela_mean:.1f} — suggests compression history mismatch"
            )
        if suspicious_percent > 20:
            flags.append(
                f"LARGE_SUSPICIOUS_REGION: {suspicious_percent:.1f}% of image shows elevated error levels"
            )

        log.info(
            f"ELA complete — max: {ela_max}, mean: {ela_mean:.2f}, suspicious: {suspicious_percent:.1f}%"
        )

        result = {
            "ela_max": ela_max,
            "ela_mean": round(ela_mean, 3),
            "suspicious_region_percent": round(suspicious_percent, 2),
            "flags": flags,
            "module_status": "ok",
        }
        if not flags:
            # ELA is an indirect JPEG consistency check, so a clean result is
            # useful but never high-confidence proof of authenticity.
            result["clean_evidence_text"] = "No significant compression-history anomalies detected"
            result["clean_evidence_confidence"] = 65
        return result

    except Exception as e:
        log.warning(f"ELA analysis failed: {e}")
        return {
            "ela_max": None,
            "ela_mean": None,
            "suspicious_region_percent": None,
            "flags": [f"ELA_ERROR: {str(e)}"],
            "module_status": "error",
        }


def get_source_format_info(image_path: str) -> dict:
    """
    PATCH 2: Call ONCE per image, early in run_forensics_agent, and pass
    the result into run_ela_analysis_v2(). A native PNG (or other lossless
    format) has no prior JPEG compression history for ELA to measure
    inconsistency against — a "clean" ELA reading on such a file is not
    evidence of authenticity, just a non-result.
    """
    try:
        with Image.open(image_path) as im:
            fmt = im.format
        ela_applicable = fmt in ("JPEG", "JPG", "MPO")
        return {"source_format": fmt, "ela_applicable": ela_applicable}
    except Exception:
        # unknown: don't over- or under-claim
        return {"source_format": None, "ela_applicable": True}


def run_ela_analysis_v2(image_path: str, format_info: dict, quality: int = 90) -> dict:
    """PATCH 2: wraps run_ela_analysis with format-aware confidence."""
    if not format_info.get("ela_applicable", True):
        return {
            "flags": [],
            "module_status": "not_applicable",
            "ela_module_applicable": False,
            "source_format": format_info.get("source_format"),
            "clean_evidence_confidence": 0,
            "clean_evidence_text": (
                f"ELA is NOT_APPLICABLE to source format {format_info.get('source_format')}. "
                "No ELA evidence or score penalty was produced."
            ),
        }
    result = run_ela_analysis(image_path, quality)
    result["ela_module_applicable"] = True
    result["source_format"] = format_info.get("source_format")
    return result


# ============================================================
# MODULE 2: NOISE & FREQUENCY ANALYSIS (UNCHANGED)
# ============================================================


def analyze_noise_and_frequency(image_path: str) -> dict:
    log.info("Running noise and frequency analysis...")
    flags = []

    try:
        img = cv2.imread(image_path)
        if img is None:
            return {
                "flags": ["IMAGE_READ_ERROR: Could not read image with OpenCV"],
                "module_status": "error",
            }

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray - blurred

        noise_std = float(np.std(noise))
        noise_mean = float(np.mean(np.abs(noise)))

        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.log(np.abs(fft_shift) + 1)

        center_y, center_x = magnitude.shape[0] // 2, magnitude.shape[1] // 2
        edge_region = np.concatenate(
            [magnitude[:10, :].flatten(), magnitude[-10:, :].flatten()])
        center_region = magnitude[
            center_y - 20: center_y + 20, center_x - 20: center_x + 20
        ].flatten()

        freq_ratio = float(np.mean(center_region) /
                           (np.mean(edge_region) + 1e-10))

        if noise_std < 1.5:
            flags.append(
                f"LOW_NOISE_STD: Noise std={noise_std:.2f} — low residual variance; weak, non-diagnostic signal common in flat-color graphics and illustrations"
            )
        if noise_std > 30:
            flags.append(
                f"HIGH_NOISE: Noise std={noise_std:.2f} — heavy noise or compression artifacts"
            )
        if freq_ratio > 100:
            flags.append(
                f"FFT_ANOMALY: Unusual frequency concentration — weak indicator; can result from upscaling, sharpening filters, or AI generation"
            )

        log.info(
            f"Noise analysis — std: {noise_std:.2f}, freq_ratio: {freq_ratio:.2f}")

        result = {
            "noise_std": round(noise_std, 3),
            "noise_mean": round(noise_mean, 3),
            "fft_freq_ratio": round(freq_ratio, 3),
            "flags": flags,
            "module_status": "ok",
        }
        if not flags:
            result["clean_evidence_text"] = "No frequency-domain anomaly was detected by the completed checks"
            result["clean_evidence_confidence"] = 55
        return result

    except Exception as e:
        log.warning(f"Noise/frequency analysis failed: {e}")
        return {"flags": [f"NOISE_ANALYSIS_ERROR: {str(e)}"], "module_status": "error"}


# ============================================================
# MODULE 3: FACE ARTIFACT DETECTOR
# Uses the PATCHED _detect_faces() (YuNet-primary) via the same call
# signature as before — no changes needed to the body of this function.
# ============================================================


def detect_face_artifacts(image_path: str) -> dict:
    log.info("Running face artifact detection...")
    flags = []

    try:
        img = cv2.imread(image_path)
        if img is None:
            return {
                "faces_found": 0,
                "face_boxes": [],
                "flags": ["IMAGE_READ_ERROR"],
                "module_status": "error",
            }

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces, detection_method, detection_confidence = _detect_faces(img)
        eye_cascade = _get_cascade("haarcascade_eye.xml")

        if detection_method == "fallback_skin_color":
            flags.append(
                "FACE_DETECTION_FALLBACK_MODE: Haar cascade classifiers were unavailable in this "
                "environment; face regions were estimated using a lower-fidelity skin-tone/contour "
                "method instead. Face-based findings below carry reduced confidence."
            )

        if len(faces) == 0:
            log.info(
                f"No faces detected in image (chain exhausted, method={detection_method})")
            return {
                "faces_found": 0,
                "face_boxes": [],
                "detection_method": detection_method,
                "detection_confidence": "none",
                "coverage_status": "partial",
                "coverage_reason": "NO_RELIABLE_FACE_DETECTED",
                "inconclusive_evidence_text": (
                    "No face could be located in this image by any detector in the fallback chain "
                    "(YuNet DNN, Haar primary/secondary, skin-tone heuristic) — face-artifact analysis "
                    "could not be performed. This does not confirm the image is clean; it means this "
                    "module has no opinion."
                ),
                "flags": flags,
                "module_status": "partial",
            }

        log.info(
            f"Found {len(faces)} face(s) via {detection_method} (confidence={detection_confidence})"
        )
        face_analyses = []

        for i, (x, y, w, h) in enumerate(faces):
            face_roi_gray = gray[y: y + h, x: x + w]

            inner = face_roi_gray[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
            border_top = face_roi_gray[: h // 8, :]
            border_bot = face_roi_gray[7 * h // 8:, :]

            inner_laplacian = cv2.Laplacian(inner, cv2.CV_64F).var()
            border_laplacian = (
                cv2.Laplacian(border_top, cv2.CV_64F).var()
                + cv2.Laplacian(border_bot, cv2.CV_64F).var()
            ) / 2

            sharpness_ratio = float(
                inner_laplacian / (border_laplacian + 1e-10))
            skin_noise = float(np.std(face_roi_gray.astype(np.float64)))

            eye_count = 0
            if eye_cascade is not None:
                eyes = eye_cascade.detectMultiScale(
                    face_roi_gray, scaleFactor=1.1, minNeighbors=3)
                eye_count = len(eyes)

            face_flags = []
            if sharpness_ratio > 15:
                face_flags.append(
                    f"FACE_{i+1}_BORDER_BLUR: Face border is much blurrier than inner face — possible deepfake/compositing artifact"
                )
            if skin_noise < 8:
                face_flags.append(
                    f"FACE_{i+1}_SMOOTH_SKIN: Skin is unnaturally smooth (noise={skin_noise:.1f}) — possible AI-smoothing artifact"
                )
            if eye_cascade is None:
                pass
            elif eye_count == 0:
                face_flags.append(
                    f"FACE_{i+1}_NO_EYES: No eyes detected in face region — weak indicator; may simply be a detector miss (angle, occlusion, glasses, closed eyes)"
                )
            elif eye_count > 3:
                face_flags.append(
                    f"FACE_{i+1}_EYE_ARTIFACT: Unusual eye count ({eye_count}) detected — weak indicator; often a false detection rather than a facial artifact"
                )

            flags.extend(face_flags)
            face_analyses.append(
                {
                    "face_index": i + 1,
                    "bounding_box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                    "sharpness_ratio": round(sharpness_ratio, 2),
                    "skin_smoothness_noise": round(skin_noise, 2),
                    "eyes_detected": int(eye_count),
                    "face_flags": face_flags,
                }
            )

        log.info(
            f"Face artifact detection complete — {len(flags)} flags found")
        result = {
            "faces_found": len(faces),
            "face_boxes": [list(f) for f in faces],
            "detection_method": detection_method,
            "detection_confidence": detection_confidence,
            "face_analyses": face_analyses,
            "flags": flags,
            "module_status": "ok",
            "coverage_status": "complete",
        }
        incomplete_eye_faces = [
            item["face_index"] for item in face_analyses
            if item.get("eyes_detected", 0) < 2
        ]
        if detection_confidence == "low" or incomplete_eye_faces:
            result["module_status"] = "partial"
            result["coverage_status"] = "partial"
            reasons = []
            if detection_confidence == "low":
                reasons.append("LOW_CONFIDENCE_FACE_LOCALIZATION")
            if incomplete_eye_faces:
                reasons.append("FEWER_THAN_TWO_RELIABLE_EYES")
            result["coverage_reason"] = ",".join(reasons)
            result["inconclusive_evidence_text"] = (
                f"Face detected via low-confidence fallback method ({detection_method}); no artifacts "
                "found, but detection fidelity is limited so this check is inconclusive."
                if detection_confidence == "low" else
                "Face localization succeeded, but fewer than two reliable eyes were detected; "
                "eye consistency checks did not run and the face result is inconclusive."
            )
        else:
            result["clean_evidence_text"] = "No face-swap or deepfake artifacts were found by the checks that completed."
            result["clean_evidence_confidence"] = 70
        return result

    except Exception as e:
        log.warning(f"Face artifact detection failed: {e}")
        return {
            "faces_found": 0,
            "face_boxes": [],
            "flags": [f"FACE_DETECTION_ERROR: {str(e)}"],
            "module_status": "error",
        }


# ============================================================
# MODULE 3b: AI-GENERATION HEURISTICS (images)
# PATCH 3: lighting-direction flag is now severity-scaled (see
# score_lighting_flag below) instead of a flat threshold/deduction.
# ============================================================

LIGHTING_CIRC_VAR_THRESHOLD = 0.85
# near-random gradient field — essentially no coherent light source
LIGHTING_CIRC_VAR_SEVERE = 0.93


def score_lighting_flag(circ_var: float) -> tuple:
    """
    PATCH 3: circ_var ranges 0 (perfectly consistent single-light-source
    shading) to 1 (uniformly random gradient directions). A near-maximal
    reading (>= LIGHTING_CIRC_VAR_SEVERE) deserves more weight than a
    barely-over-threshold one, so it's emitted under a distinct flag name
    that CRITICAL_KEYWORDS can match (add "LIGHTING_DIRECTION_SEVERE" to
    that list in Module 7 — see note at the bottom of that section).
    Returns (flag_text_or_None, severity_label).
    """
    if circ_var <= LIGHTING_CIRC_VAR_THRESHOLD:
        return None, None
    if circ_var >= LIGHTING_CIRC_VAR_SEVERE:
        return (
            f"LIGHTING_DIRECTION_SEVERE: circ_var={circ_var:.2f} — near-maximal directional "
            "inconsistency, well beyond normal single-light-source photography; strong indicator "
            "of synthetic/composited lighting",
            "critical",
        )
    return (
        f"LIGHTING_DIRECTION_INCONSISTENT: circ_var={circ_var:.2f} — weak-to-moderate indicator; "
        "can suggest mismatched or synthetic lighting",
        "moderate",
    )


def classify_image_content(image_path: str) -> dict:
    """Separate structured layouts from photographic scenes conservatively.

    Texture in water, foliage, hair, and buildings creates many elongated and
    rectangular contours.  Those shapes are candidates, not proof of text or a
    page layout, so layout-only routing requires corroboration from their amount
    and foreground coverage.
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"content_type": "UNKNOWN", "document_like": False, "module_status": "error"}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, 1600.0 / max(gray.shape))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 2))
    joined = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        joined, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = gray.shape[0] * gray.shape[1]
    text_rows = 0
    rectangular_regions = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area_ratio = (w * h) / max(image_area, 1)
        if 8 <= h <= max(12, int(gray.shape[0] * 0.08)) and w >= h * 1.8:
            text_rows += 1
        if 0.01 <= area_ratio <= 0.45 and w > 20 and h > 20:
            rectangular_regions += 1
    foreground_density = float(np.mean(binary > 0))
    text_row_coverage = 0.0
    if image_area:
        text_row_coverage = sum(
            cv2.boundingRect(contour)[2] * cv2.boundingRect(contour)[3]
            for contour in contours
            if (
                8 <= cv2.boundingRect(contour)[3]
                <= max(12, int(gray.shape[0] * 0.08))
                and cv2.boundingRect(contour)[2] >= cv2.boundingRect(contour)[3] * 1.8
            )
        ) / image_area

    # Strong text-row evidence is enough for a page/news-card decision.  At
    # lower counts, require both dense thresholded foreground and substantial
    # row coverage.  In particular, rectangle counts alone never make a photo a
    # document: natural scenes routinely contain many such contours.
    strong_text_layout = text_rows >= 50
    corroborated_layout = (
        text_rows >= 20
        and rectangular_regions >= 8
        and foreground_density >= 0.18
        and text_row_coverage >= 0.18
    )
    document_like = bool(strong_text_layout or corroborated_layout)
    classification_reasons = []
    if strong_text_layout:
        classification_reasons.append("DENSE_TEXT_ROW_STRUCTURE")
    if corroborated_layout:
        classification_reasons.append("CORROBORATED_LAYOUT_STRUCTURE")
    if not document_like:
        classification_reasons.append("INSUFFICIENT_CORROBORATED_LAYOUT_EVIDENCE")
    return {
        "content_type": "DOCUMENT_OR_LAYOUT" if document_like else "PHOTO_SCENE",
        # Keep the legacy routing field above, but expose a more useful asset
        # type to downstream consumers. A layout is not one natural-camera
        # scene, so portrait/lighting heuristics are gated.
        "asset_type": "COMPOSITE_GRAPHIC" if document_like else "NATURAL_PHOTOGRAPH",
        "document_like": document_like,
        "text_row_candidates": text_rows,
        "rectangular_region_candidates": rectangular_regions,
        "threshold_foreground_density": round(foreground_density, 4),
        "text_row_coverage": round(float(text_row_coverage), 4),
        "classification_reasons": classification_reasons,
        "module_status": "ok",
    }


def analyze_ai_generation_heuristics(
    image_path: str, face_boxes: list, content_context: dict = None
) -> dict:
    log.info("Running AI-generation heuristic analysis...")
    flags = []
    metrics = {}

    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"flags": ["IMAGE_READ_ERROR"], "module_status": "error"}
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h_img, w_img = gray.shape
        eye_cascade = _get_cascade("haarcascade_eye.xml")

        if face_boxes:
            x, y, w, h = face_boxes[0]
            x, y, w, h = int(x), int(y), int(w), int(h)
            face = gray[y: y + h, x: x + w]

            if face.size > 0:
                mid = w // 2
                left = face[:, :mid]
                right = face[:, mid: mid *
                             2] if w % 2 == 0 else face[:, mid + 1: mid * 2 + 1]
                right_flipped = cv2.flip(right, 1)
                min_w = min(left.shape[1], right_flipped.shape[1])
                if min_w > 0:
                    diff = cv2.absdiff(
                        left[:, :min_w], right_flipped[:, :min_w])
                    symmetry_score = float(np.mean(diff))
                    metrics["facial_symmetry_diff"] = round(symmetry_score, 2)
                    if symmetry_score < 4.0:
                        flags.append(
                            f"FACE_OVERSYMMETRIC: Left/right face halves are unusually mirror-like (diff={symmetry_score:.1f}) — weak indicator; real faces are rarely this symmetric, sometimes seen in GAN/diffusion output"
                        )
                    elif symmetry_score > 35.0:
                        flags.append(
                            f"FACE_ASYMMETRY_ANOMALY: Left/right face halves diverge sharply (diff={symmetry_score:.1f}) — weak indicator; can also result from pose angle or lighting"
                        )

                if eye_cascade is not None:
                    eyes = eye_cascade.detectMultiScale(
                        face, scaleFactor=1.1, minNeighbors=3)
                    if len(eyes) >= 2:
                        eyes_sorted = sorted(eyes, key=lambda e: e[0])[:2]
                        catchlights, eye_colors = [], []
                        for ex, ey, ew, eh in eyes_sorted:
                            eye_roi = img[y + ey: y + ey +
                                          eh, x + ex: x + ex + ew]
                            if eye_roi.size == 0:
                                continue
                            eye_gray = cv2.cvtColor(
                                eye_roi, cv2.COLOR_BGR2GRAY)
                            _, max_val, _, max_loc = cv2.minMaxLoc(eye_gray)
                            catchlights.append(
                                (max_val > 200, max_loc, eye_gray.shape))
                            eye_colors.append(
                                np.mean(eye_roi.reshape(-1, 3), axis=0))

                        if len(catchlights) == 2:
                            (has1, loc1, shape1), (has2,
                                                   loc2, shape2) = catchlights
                            if has1 != has2:
                                flags.append(
                                    "EYE_CATCHLIGHT_MISMATCH: Reflection highlight present in one eye but not the other — weak-to-moderate indicator of compositing/AI generation"
                                )
                            elif has1 and has2:
                                norm1 = (loc1[0] / max(shape1[1], 1),
                                         loc1[1] / max(shape1[0], 1))
                                norm2 = (loc2[0] / max(shape2[1], 1),
                                         loc2[1] / max(shape2[0], 1))
                                pos_diff = float(
                                    np.hypot(norm1[0] - norm2[0], norm1[1] - norm2[1]))
                                metrics["catchlight_position_diff"] = round(
                                    pos_diff, 3)
                                if pos_diff > 0.35:
                                    flags.append(
                                        f"EYE_CATCHLIGHT_POSITION_MISMATCH: Reflection highlight sits in inconsistent positions between eyes (diff={pos_diff:.2f}) — weak indicator; light-source/reflection inconsistency is common in AI-generated faces"
                                    )

                        if len(eye_colors) == 2:
                            color_diff = float(np.linalg.norm(
                                eye_colors[0] - eye_colors[1]))
                            metrics["eye_color_diff"] = round(color_diff, 2)
                            if color_diff > 40:
                                flags.append(
                                    f"EYE_COLOR_MISMATCH: Left/right eye color differs noticeably (diff={color_diff:.1f}) — weak indicator; can also occur naturally (heterochromia) or from lighting"
                                )

                ear_margin = int(w * 0.12)
                if ear_margin > 3 and x - ear_margin >= 0 and x + w + ear_margin <= w_img:
                    left_ear = gray[y + int(h * 0.3): y +
                                    int(h * 0.75), x - ear_margin: x]
                    right_ear = gray[
                        y + int(h * 0.3): y + int(h * 0.75), x + w: x + w + ear_margin
                    ]
                    cheek = face[int(h * 0.3): int(h * 0.75),
                                 int(w * 0.1): int(w * 0.3)]
                    if left_ear.size > 0 and right_ear.size > 0 and cheek.size > 0:
                        ear_texture = float(
                            np.std(left_ear)) + float(np.std(right_ear))
                        cheek_texture = float(np.std(cheek)) * 2
                        ratio = ear_texture / (cheek_texture + 1e-6)
                        metrics["ear_vs_cheek_texture_ratio"] = round(ratio, 3)
                        if ratio < 0.5:
                            flags.append(
                                "EAR_TEXTURE_SMOOTHING: Ear regions are much smoother/less detailed than surrounding skin — weak indicator; AI generation often under-renders ears"
                            )

        edges = cv2.Canny(gray, 80, 160)
        grid = 8
        gh, gw = h_img // grid, w_img // grid
        if gh > 0 and gw > 0:
            densities = [
                float(
                    np.mean(edges[r * gh: (r + 1) * gh, c * gw: (c + 1) * gw] > 0))
                for r in range(grid)
                for c in range(grid)
            ]
            edge_density_std = float(np.std(densities))
            metrics["edge_density_std"] = round(edge_density_std, 4)
            if edge_density_std > 0.18:
                flags.append(
                    f"EDGE_DENSITY_IRREGULAR: Edge density varies sharply across image regions (std={edge_density_std:.2f}) — weak indicator of splicing/patchwork generation seams"
                )

            lap = cv2.Laplacian(gray, cv2.CV_64F)
            block_vars = [
                float(np.var(lap[r * gh: (r + 1) * gh, c * gw: (c + 1) * gw]))
                for r in range(grid)
                for c in range(grid)
            ]
            if block_vars and np.mean(block_vars) > 1e-6:
                texture_cv = float(np.std(block_vars) /
                                   (np.mean(block_vars) + 1e-6))
                metrics["texture_coefficient_of_variation"] = round(
                    texture_cv, 3)
                if texture_cv > 2.5:
                    flags.append(
                        f"TEXTURE_INCONSISTENCY: Local texture sharpness varies unusually between image regions (CV={texture_cv:.2f}) — weak indicator; consistent with AI upsampling/inpainting patchwork"
                    )

        not_applicable_checks = []
        if (content_context or {}).get("document_like"):
            not_applicable_checks.append(
                "GLOBAL_LIGHTING_DIRECTION: NOT_APPLICABLE to document, screenshot, collage, "
                "poster, or multi-panel layout content"
            )
        else:
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            magnitude = np.hypot(sobel_x, sobel_y)
            if magnitude.max() > 0:
                strong_mask = magnitude > np.percentile(magnitude, 90)
                if np.sum(strong_mask) > 100:
                    angles = np.arctan2(
                        sobel_y[strong_mask], sobel_x[strong_mask])
                    circ_var = 1 - \
                        float(np.hypot(np.mean(np.cos(angles)),
                              np.mean(np.sin(angles))))
                    metrics["lighting_direction_circular_variance"] = round(
                        circ_var, 3)
                    lighting_flag_text, _ = score_lighting_flag(circ_var)
                    if lighting_flag_text:
                        flags.append(lighting_flag_text)

        log.info(
            f"AI-generation heuristics complete — {len(flags)} flags found")
        result = {
            "metrics": metrics,
            "flags": flags,
            "not_applicable_checks": not_applicable_checks,
            "content_context": content_context or {},
            "module_status": "ok",
        }
        if not flags:
            result["clean_evidence_text"] = "No AI-generation anomaly was detected by the heuristic checks that ran"
            result["clean_evidence_confidence"] = 50
        return result

    except Exception as e:
        log.warning(f"AI-generation heuristics analysis failed: {e}")
        return {"flags": [f"AI_HEURISTICS_ERROR: {str(e)}"], "module_status": "error"}


# ============================================================
# MODULE 3c: HAND / FINGER ARTIFACT HEURISTIC (UNCHANGED)
# ============================================================

_MIN_HAND_CONTOUR_POINTS = 5
_MIN_HAND_CONTOUR_AREA_PX = 400
_MAX_HAND_REGIONS = 2


def _count_finger_defects(contour) -> int:
    if contour is None or len(contour) < _MIN_HAND_CONTOUR_POINTS:
        return -1
    if cv2.contourArea(contour) < _MIN_HAND_CONTOUR_AREA_PX:
        return -1

    try:
        hull_idx = cv2.convexHull(contour, returnPoints=False)
    except cv2.error:
        return -1
    if hull_idx is None or len(hull_idx) <= 3:
        return -1

    hull_idx = np.unique(hull_idx.flatten())
    if len(hull_idx) <= 3:
        return -1
    hull_idx = hull_idx.reshape(-1, 1).astype(np.int32)

    try:
        defects = cv2.convexityDefects(contour, hull_idx)
    except cv2.error:
        return -1
    if defects is None or defects.size == 0:
        return -1

    defects = defects.reshape(-1, 4)

    finger_count = 0
    for row in defects:
        depth = row[3]
        if depth / 256.0 > 12:
            finger_count += 1
    if finger_count > 0:
        finger_count += 1
    return finger_count


def detect_hand_artifacts(image_path: str) -> dict:
    log.info("Running hand/finger artifact heuristic...")
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {
                "hand_region_found": False,
                "flags": ["IMAGE_READ_ERROR"],
                "module_status": "error",
            }

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 30, 60], dtype=np.uint8)
        upper = np.array([25, 180, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_area = img.shape[0] * img.shape[1]
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 0.005 * img_area or area > 0.35 * img_area:
                continue
            if len(c) < _MIN_HAND_CONTOUR_POINTS:
                continue
            x, y, w, h = cv2.boundingRect(c)
            aspect = h / (w + 1e-6)
            if 0.6 <= aspect <= 2.2:
                candidates.append(c)
        candidates.sort(key=cv2.contourArea, reverse=True)
        candidates = candidates[:_MAX_HAND_REGIONS]

        if not candidates:
            log.info("No hand-shaped region found")
            return {
                "hand_region_found": False,
                "regions_analyzed": 0,
                "flags": [
                    "NO_HAND_REGION_DETECTED: No plausible hand-shaped skin-tone region was found — this is an informational result, not evidence either way"
                ],
                "module_status": "ok",
            }

        region_results = []
        flags = []
        for idx, c in enumerate(candidates):
            finger_count = _count_finger_defects(c)
            if finger_count < 0:
                region_results.append(
                    {"region_index": idx + 1, "analyzable": False,
                        "estimated_finger_count": None}
                )
                continue
            region_results.append(
                {
                    "region_index": idx + 1,
                    "analyzable": True,
                    "estimated_finger_count": int(finger_count),
                }
            )
            if finger_count > 0 and (finger_count > 6 or finger_count < 3):
                flags.append(
                    f"HAND_FINGER_COUNT_ANOMALY: Region {idx+1} — estimated {finger_count} finger-like "
                    "protrusions on a detected hand-shaped region — weak indicator; this heuristic is noisy "
                    "(occlusion, pose, jewelry commonly cause miscounts) and should not be relied on alone"
                )

        analyzable_results = [r for r in region_results if r["analyzable"]]
        if not analyzable_results:
            flags.append(
                "NO_HAND_REGION_DETECTED: Hand-shaped region(s) found but too small/degenerate to analyze reliably — treated as informational, not evidence"
            )

        log.info(
            f"Hand heuristic complete — {len(analyzable_results)}/{len(candidates)} region(s) analyzable"
        )
        return {
            "hand_region_found": bool(analyzable_results),
            "regions_analyzed": len(candidates),
            "estimated_finger_count": (
                analyzable_results[0]["estimated_finger_count"] if analyzable_results else None
            ),
            "region_results": region_results,
            "flags": flags,
            "module_status": "ok",
        }

    except Exception as e:
        log.warning(f"Hand artifact detection failed: {e}")
        return {
            "hand_region_found": False,
            "flags": [f"HAND_ARTIFACT_ERROR: {str(e)}"],
            "module_status": "error",
        }


# ============================================================
# MODULE 4: VIDEO TEMPORAL CONSISTENCY CHECKER (UNCHANGED)
# ============================================================


def analyze_video_temporal_consistency(video_path: str, max_frames: int = 60) -> dict:
    log.info(
        f"Analyzing video temporal consistency (sampling {max_frames} frames)...")
    flags = []

    try:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return {
                "flags": ["VIDEO_OPEN_ERROR: Could not open video file"],
                "module_status": "error",
            }

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        log.info(
            f"Video: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s")

        sample_interval = max(1, total_frames // max_frames)
        frame_brightnesses = []
        frame_noise_levels = []

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_brightnesses.append(float(np.mean(gray)))
                blurred = cv2.GaussianBlur(gray.astype(np.float64), (5, 5), 0)
                noise = float(np.std(gray.astype(np.float64) - blurred))
                frame_noise_levels.append(noise)

            frame_idx += 1

        cap.release()

        if len(frame_brightnesses) < 2:
            return {
                "flags": ["TOO_FEW_FRAMES: Not enough frames to analyze"],
                "module_status": "error",
            }

        brightness_std = float(np.std(frame_brightnesses))
        brightness_changes = np.diff(frame_brightnesses)
        sudden_changes = np.sum(np.abs(brightness_changes) > 30)

        noise_std = float(np.std(frame_noise_levels))

        if sudden_changes > 5:
            flags.append(
                f"BRIGHTNESS_JUMPS: {int(sudden_changes)} abrupt brightness changes detected. "
                "May indicate scene transitions, editing, or lighting changes. "
                "This alone is insufficient evidence of manipulation."
            )
        if noise_std > 5:
            flags.append(
                f"INCONSISTENT_NOISE: Frame noise varies significantly (std={noise_std:.1f}) — moderate indicator of inconsistent rendering or re-encoding"
            )
        if brightness_std > 40:
            flags.append(
                f"HIGH_BRIGHTNESS_VARIANCE: Large brightness variance (std={brightness_std:.1f}) — can occur in normal video with varied scenes/lighting"
            )

        log.info(
            f"Temporal analysis complete — brightness_std: {brightness_std:.1f}, sudden_changes: {sudden_changes}"
        )

        return {
            "total_frames": total_frames,
            "fps": round(fps, 2),
            "duration_seconds": round(duration, 2),
            "frames_sampled": len(frame_brightnesses),
            "brightness_std": round(brightness_std, 3),
            "sudden_brightness_changes": int(sudden_changes),
            "noise_std": round(noise_std, 3),
            "flags": flags,
            "module_status": "ok",
        }

    except Exception as e:
        log.warning(f"Video temporal analysis failed: {e}")
        return {"flags": [f"VIDEO_ANALYSIS_ERROR: {str(e)}"], "module_status": "error"}


# ============================================================
# MODULE 5: AUDIO SPECTROGRAM ANALYSIS (UNCHANGED)
# ============================================================


def analyze_audio_spectrogram(audio_path: str) -> dict:
    if not LIBROSA_AVAILABLE:
        return {
            "flags": ["LIBROSA_NOT_INSTALLED: Audio analysis skipped"],
            "module_status": "skipped",
        }

    log.info("Analyzing audio spectrogram...")
    flags = []

    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True, duration=60)

        if len(y) == 0:
            return {"flags": ["EMPTY_AUDIO: No audio content found"], "module_status": "error"}

        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        centroid_mean = float(np.mean(spectral_centroids))
        centroid_std = float(np.std(spectral_centroids))

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        rolloff_mean = float(np.mean(rolloff))

        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean = float(np.mean(zcr))
        zcr_std = float(np.std(zcr))

        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
        )
        f0_clean = f0[~np.isnan(f0)]

        if len(f0_clean) > 0:
            pitch_std = float(np.std(f0_clean))
            pitch_mean = float(np.mean(f0_clean))
            voiced_ratio = float(np.sum(voiced_flag) / len(voiced_flag))

            if pitch_std < 10:
                flags.append(
                    f"FLAT_PITCH: Low pitch variation (std={pitch_std:.1f}Hz) — moderate indicator; consistent with robotic/synthetic voice but also with monotone speech, whispering, or short clips"
                )
            if voiced_ratio > 0.95:
                flags.append(
                    "NO_SILENCE: Barely any silence detected — weak indicator; unusual for natural human speech but common in music, ambient recordings, or tightly-edited clips"
                )
            if voiced_ratio < 0.1:
                flags.append(
                    "MOSTLY_SILENCE: Very little voice content detected")
        else:
            pitch_std = None
            pitch_mean = None
            voiced_ratio = None
            flags.append(
                "NO_PITCH_DETECTED: No voiced segments found — may be ambient noise or music only"
            )

        clipping_threshold = 0.99
        clipped_samples = np.sum(np.abs(y) > clipping_threshold)
        clipping_ratio = clipped_samples / len(y)
        if clipping_ratio > 0.01:
            flags.append(
                f"AUDIO_CLIPPING: {clipping_ratio*100:.1f}% of samples clipped — weak indicator; usually a recording/processing artifact rather than evidence of manipulation"
            )

        flatness = librosa.feature.spectral_flatness(y=y)[0]
        flatness_mean = float(np.mean(flatness))
        if flatness_mean > 0.1:
            flags.append(
                f"HIGH_SPECTRAL_FLATNESS: Value={flatness_mean:.3f} — weak indicator. "
                "Elevated spectral flatness may result from compression, environmental noise, "
                "microphone quality, or synthesized audio. Additional evidence is required."
            )

        log.info(
            f"Audio analysis complete — pitch_std: {pitch_std}, voiced_ratio: {voiced_ratio}")

        return {
            "duration_seconds": round(len(y) / sr, 2),
            "sample_rate": sr,
            "centroid_mean": round(centroid_mean, 2),
            "centroid_std": round(centroid_std, 2),
            "rolloff_mean": round(rolloff_mean, 2),
            "zcr_mean": round(zcr_mean, 4),
            "zcr_std": round(zcr_std, 4),
            "pitch_mean_hz": round(pitch_mean, 2) if pitch_mean else None,
            "pitch_std_hz": round(pitch_std, 2) if pitch_std else None,
            "voiced_ratio": round(voiced_ratio, 3) if voiced_ratio else None,
            "spectral_flatness": round(flatness_mean, 4),
            "clipping_ratio": round(clipping_ratio, 5),
            "flags": flags,
            "module_status": "ok",
        }

    except Exception as e:
        log.warning(f"Audio spectrogram analysis failed: {e}")
        return {"flags": [f"AUDIO_ANALYSIS_ERROR: {str(e)}"], "module_status": "error"}


# ============================================================
# MODULE 6: LIP-SYNC DRIFT DETECTOR (UNCHANGED, uses patched _get_cascade)
# ============================================================

# ============================================================
# FIX A — FORENSICS AGENT: detect_lipsync_drift() face detection
#
# WHY: this function bypassed the patched _detect_faces() chain entirely
# and called _get_cascade() + raw Haar detectMultiScale() directly. So the
# YuNet DNN detector — which is working fine elsewhere in your pipeline —
# was never even tried here. Any environment where Haar's self-heal is
# flaky (as yours currently is) makes lip-sync permanently unavailable,
# even on videos with clear, well-lit faces.
#
# Replace the face-detection block inside detect_lipsync_drift() with this
# version. Only the face-finding lines change — audio loading, envelope
# extraction, and correlation math are untouched.
# ============================================================


def detect_lipsync_drift(video_path: str) -> dict:
    log.info("Running lip-sync drift detection...")
    flags = []

    if not LIBROSA_AVAILABLE:
        return {
            "flags": ["LIBROSA_NOT_INSTALLED: Lip-sync analysis skipped"],
            "module_status": "skipped",
        }

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_audio = tmp.name

        subprocess.run(
            ["ffmpeg", "-i", video_path, "-vn", "-ar",
                "16000", "-ac", "1", "-y", tmp_audio],
            capture_output=True,
        )

        audio, sr = librosa.load(tmp_audio, sr=16000)
        os.unlink(tmp_audio)

        hop_length = 512
        audio_envelope = librosa.feature.rms(y=audio, hop_length=hop_length)[0]

        cap = cv2.VideoCapture(video_path)

        # FIX: use the same YuNet-primary / Haar-fallback chain as the
        # image pipeline, instead of a raw Haar-only cascade call. This is
        # the same _detect_faces() function already defined earlier in
        # your forensics cell (PATCH 1) — no new detector code needed here,
        # just routing through the one that already works.
        detector_available = (
            _get_yunet() is not None
            or _get_cascade("haarcascade_frontalface_default.xml") is not None
        )

        if not detector_available:
            cap.release()
            return {
                "flags": [
                    "FACE_CASCADE_UNAVAILABLE: Could not obtain a working face detector (YuNet or Haar) for lip-sync analysis"
                ],
                "module_status": "error",
            }

        mouth_motion = []
        prev_mouth = None
        frame_count = 0
        last_face_box = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_count > 300:  # first ~10s at 30fps
                break

            # FIX: route through _detect_faces() (YuNet-primary, Haar/
            # skin-tone fallback) instead of calling a single Haar cascade
            # directly. Only run detection every few frames — lip-sync only
            # needs the mouth ROI to track roughly, and re-running YuNet on
            # every single frame is unnecessarily expensive; reuse the last
            # box for a few frames in between, re-detecting periodically to
            # stay correct as the subject moves.
            if frame_count % 3 == 0 or last_face_box is None:
                faces, _method, _conf = _detect_faces(frame)
            else:
                faces = [last_face_box]

            if len(faces) > 0:
                x, y, w, h = faces[0]
                last_face_box = (x, y, w, h)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mouth_y_start = y + int(h * 0.65)
                mouth_y_end = y + h
                mouth_roi = gray[mouth_y_start:mouth_y_end, x: x + w]

                if prev_mouth is not None and mouth_roi.shape == prev_mouth.shape:
                    motion = float(
                        np.mean(
                            np.abs(mouth_roi.astype(np.float64) -
                                   prev_mouth.astype(np.float64))
                        )
                    )
                    mouth_motion.append(motion)
                else:
                    mouth_motion.append(0.0)

                prev_mouth = mouth_roi.copy()
            else:
                mouth_motion.append(0.0)

            frame_count += 1

        cap.release()

        if len(mouth_motion) < 10:
            return {
                "flags": [
                    "INSUFFICIENT_FRAMES: Not enough frames with faces for lip-sync analysis"
                ],
                "module_status": "error",
            }

        audio_resampled = np.interp(
            np.linspace(0, len(audio_envelope) - 1, len(mouth_motion)),
            np.arange(len(audio_envelope)),
            audio_envelope,
        )

        if np.std(mouth_motion) > 0 and np.std(audio_resampled) > 0:
            correlation = float(np.corrcoef(
                mouth_motion, audio_resampled)[0, 1])
        else:
            correlation = 0.0

        if correlation < 0.15:
            flags.append(
                f"LIP_SYNC_DRIFT: Low correlation ({correlation:.2f}) between mouth motion and audio — strong indicator of possible audio replacement (evaluated only when sufficient spoken audio was present)"
            )
        elif correlation < 0.30:
            flags.append(
                f"WEAK_LIP_SYNC: Moderate correlation ({correlation:.2f}) — slight desync detected"
            )

        log.info(f"Lip-sync correlation: {correlation:.3f}")

        return {
            "frames_analyzed": len(mouth_motion),
            "audio_video_correlation": round(correlation, 4),
            "interpretation": (
                "Strong sync"
                if correlation >= 0.30
                else (
                    "Weak sync — investigate"
                    if correlation >= 0.15
                    else "Poor sync — likely manipulated"
                )
            ),
            "flags": flags,
            "module_status": "ok",
        }

    except Exception as e:
        log.warning(f"Lip-sync detection failed: {e}")
        return {"flags": [f"LIPSYNC_ERROR: {str(e)}"], "module_status": "error"}


# ============================================================
# PATCH 4: PERCEPTUAL HASH UTILITY (replaces hash_lookup stub)
#
# WHY: exact-byte hashing is useless against reposted/re-encoded real
# photos OR known generator outputs, since any resize/recompress changes
# the hash entirely. pHash survives that. Wire this into your provenance
# agent's hash_lookup step once you have (or start building) a hash DB —
# start logging pHashes now even with an empty DB so the lookup has
# something to check against later.
# ============================================================


def compute_perceptual_hashes(image_path: str) -> dict:
    if not IMAGEHASH_AVAILABLE:
        return {"error": "imagehash not installed"}
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        return {
            "phash": str(imagehash.phash(im)),
            "dhash": str(imagehash.dhash(im)),
            "average_hash": str(imagehash.average_hash(im)),
        }


def perceptual_hash_lookup(
    image_path: str, known_hash_db: dict = None, max_distance: int = 8
) -> dict:
    """
    known_hash_db: {phash_str: {"label": "...", "source": "..."}}.
    Pass {} until you have a real datastore populated — even a flat file
    you append pHashes to over time becomes useful once you have a corpus.
    Distance <= max_distance on a 64-bit pHash is a strong near-duplicate
    match (random-chance expected distance is ~32).
    """
    if not IMAGEHASH_AVAILABLE:
        return {"hash_match": False, "verdict": "NOT_CHECKED", "detail": "imagehash not installed"}

    hashes = compute_perceptual_hashes(image_path)
    if not known_hash_db:
        return {
            "hash_match": False,
            "verdict": "NOT_CHECKED",
            "detail": "No hash database populated yet — this call computed and can log the pHash for future lookups.",
            "computed_hashes": hashes,
        }
    query = imagehash.hex_to_hash(hashes["phash"])
    best_match, best_dist = None, 999
    for known_hex, meta in known_hash_db.items():
        dist = query - imagehash.hex_to_hash(known_hex)
        if dist < best_dist:
            best_dist, best_match = dist, meta
    if best_dist <= max_distance:
        return {
            "hash_match": True,
            "verdict": "NEAR_DUPLICATE_FOUND",
            "match_distance": int(best_dist),
            "match_info": best_match,
            "computed_hashes": hashes,
        }
    return {
        "hash_match": False,
        "verdict": "NO_MATCH",
        "detail": f"Closest known hash distance={best_dist} (threshold={max_distance})",
        "computed_hashes": hashes,
    }


# ============================================================
# MODULE 7: SCORING, EXPLAINABILITY & ROUTING
# PATCH 3 note: "LIGHTING_DIRECTION_SEVERE" added to CRITICAL_KEYWORDS
# so a near-maximal circ_var reading lands at -20 instead of -10.
# ============================================================

_MODULE_LABELS = {
    "ela": "Error Level Analysis (compression consistency)",
    "noise_frequency": "Noise & frequency-domain analysis",
    "face_artifacts": "Face artifact / deepfake detector",
    "ai_heuristics": "AI-generation heuristics (symmetry, eyes, ears, edges, texture, lighting)",
    "hand_artifacts": "Hand/finger artifact heuristic",
    "temporal_consistency": "Video frame-to-frame consistency",
    "audio_analysis": "Audio spectrogram / synthetic voice analysis",
    "lipsync": "Lip-sync drift detection",
}

_CLEAN_EVIDENCE = {
    "ela": "No significant compression-history anomalies detected",
    "noise_frequency": "No GAN fingerprints or frequency-domain anomalies detected",
    "face_artifacts": "No face-swap or deepfake artifacts detected",
    "ai_heuristics": "No AI-generation heuristic anomalies detected across symmetry, eye, ear, edge, texture, or lighting checks",
    "hand_artifacts": "No hand region found, or estimated finger count was plausible",
    "temporal_consistency": "Frame-to-frame consistency is high; no splicing indicators",
    "audio_analysis": "No synthetic-voice signatures detected in audio",
    "lipsync": "Lip movement correlates normally with audio",
}

# PATCH 3: "LIGHTING_DIRECTION_SEVERE" added here so a near-maximal circ_var
# reading is scored as critical (-20) rather than falling into MODERATE_KEYWORDS.
CRITICAL_KEYWORDS = ["LIP_SYNC_DRIFT", "AI_SOFTWARE",
                     "AI_ENCODER", "LIGHTING_DIRECTION_SEVERE"]

MODERATE_KEYWORDS = [
    "BORDER_BLUR",
    "SMOOTH_SKIN",
    "HIGH_ELA",
    "LARGE_SUSPICIOUS",
    "NO_EXIF",
    "WEAK_LIP_SYNC",
    "FFT_ANOMALY",
    "INCONSISTENT_NOISE",
    "NO_EYES",
    "EYE_ARTIFACT",
    "FLAT_PITCH",
    "EYE_CATCHLIGHT_MISMATCH",
    "EDGE_DENSITY_IRREGULAR",
    "TEXTURE_INCONSISTENCY",
    "LIGHTING_DIRECTION_INCONSISTENT",
]

MANIPULATION_CONFIDENCE_THRESHOLD = 60

MODERATE_EVIDENCE_CAP = 30
MINOR_EVIDENCE_CAP = 15

NO_CRITICAL_EVIDENCE_SCORE_FLOOR = 45

INCOMPLETE_MODULE_PENALTY_PER_MODULE = 10
INCOMPLETE_MODULE_PENALTY_CAP = 25


def _is_noise_flag(flag: str) -> bool:
    head = flag.split(":")[0]
    return (
        head.endswith("_ERROR")
        or "NOT_INSTALLED" in flag
        or "SKIPPED" in flag
        or "UNAVAILABLE" in flag
        or "NO_HAND_REGION_DETECTED" in flag
        or head.endswith("_FALLBACK_MODE")
    )


def _flag_severity(flag: str) -> str:
    if any(k in flag for k in CRITICAL_KEYWORDS):
        return "critical"
    if any(k in flag for k in MODERATE_KEYWORDS):
        return "moderate"
    return "minor"


def _tally_flag_severities(all_flags: list) -> tuple:
    critical_n = moderate_n = minor_n = 0
    for flag in all_flags:
        if _is_noise_flag(flag):
            continue
        sev = _flag_severity(flag)
        if sev == "critical":
            critical_n += 1
        elif sev == "moderate":
            moderate_n += 1
        else:
            minor_n += 1
    return critical_n, moderate_n, minor_n


_EVIDENCE_DIMENSIONS = {
    "compression": ["HIGH_ELA", "LARGE_SUSPICIOUS", "LOW_NOISE_STD", "HIGH_NOISE", "FFT_ANOMALY"],
    "face_geometry": ["BORDER_BLUR", "SMOOTH_SKIN", "NO_EYES", "EYE_ARTIFACT"],
    "facial_symmetry": [
        "FACE_OVERSYMMETRIC",
        "FACE_ASYMMETRY_ANOMALY",
        "EYE_CATCHLIGHT",
        "EYE_COLOR_MISMATCH",
        "EAR_TEXTURE_SMOOTHING",
    ],
    "structural_texture": [
        "EDGE_DENSITY_IRREGULAR",
        "TEXTURE_INCONSISTENCY",
        "LIGHTING_DIRECTION_INCONSISTENT",
        "LIGHTING_DIRECTION_SEVERE",
    ],
    "anatomical": ["HAND_FINGER_COUNT_ANOMALY"],
    "temporal": ["BRIGHTNESS_JUMPS", "INCONSISTENT_NOISE", "HIGH_BRIGHTNESS_VARIANCE"],
    "audio": [
        "FLAT_PITCH",
        "NO_SILENCE",
        "AUDIO_CLIPPING",
        "HIGH_SPECTRAL_FLATNESS",
        "MOSTLY_SILENCE",
        "NO_PITCH_DETECTED",
    ],
    "lipsync": ["LIP_SYNC_DRIFT", "WEAK_LIP_SYNC"],
}

CORROBORATION_BONUS_2_DIMENSIONS = 5
CORROBORATION_BONUS_3PLUS_DIMENSIONS = 15


def _evidence_dimension(flag: str) -> str:
    for dim, keywords in _EVIDENCE_DIMENSIONS.items():
        if any(k in flag for k in keywords):
            return dim
    return "other"


def _tally_corroboration(all_flags: list) -> dict:
    dims = {}
    rank = {"critical": 3, "moderate": 2, "minor": 1}
    for flag in all_flags:
        if _is_noise_flag(flag):
            continue
        dim = _evidence_dimension(flag)
        sev = _flag_severity(flag)
        if dim not in dims or rank[sev] > rank[dims[dim]]:
            dims[dim] = sev
    return dims


def _apply_capped_deduction(severity: str, tier_state: dict) -> tuple:
    if severity == "moderate":
        cap = MODERATE_EVIDENCE_CAP
        used = tier_state["moderate"]
        if used >= cap:
            return 0, f" (0 pts — moderate-evidence cap of -{cap} already reached)"
        deduction = min(10, cap - used)
        tier_state["moderate"] += deduction
        return deduction, ""
    else:
        cap = MINOR_EVIDENCE_CAP
        used = tier_state["minor"]
        if used >= cap:
            return (
                0,
                f" (0 pts — weak-evidence cap of -{cap} already reached; multiple weak indicators cannot substitute for strong evidence)",
            )
        deduction = min(5, cap - used)
        tier_state["minor"] += deduction
        return deduction, ""


def calculate_forensics_score(all_flags: list, media_type: str, findings: dict = None) -> tuple:
    score = 85
    reasoning = []
    tier_state = {"moderate": 0, "minor": 0}
    critical_present = False

    for flag in all_flags:
        if _is_noise_flag(flag):
            if "SKIPPED" in flag:
                reasoning.append(
                    f"NO_PENALTY: {flag} (module was intentionally skipped; no score impact)"
                )
            continue

        severity = _flag_severity(flag)
        if severity == "critical":
            critical_present = True
            score -= 20
            reasoning.append(
                f"CRITICAL_FLAG: {flag} (-20 pts — high-reliability indicator)")
        else:
            deduction, cap_note = _apply_capped_deduction(severity, tier_state)
            if deduction:
                score -= deduction
                reasoning.append(
                    f"{severity.upper()}_FLAG: {flag} (-{deduction} pts)")
            else:
                reasoning.append(f"{severity.upper()}_FLAG: {flag}{cap_note}")

    dims = _tally_corroboration(all_flags)
    dim_count = len(dims)
    if dim_count >= 3:
        score -= CORROBORATION_BONUS_3PLUS_DIMENSIONS
        reasoning.append(
            f"CORROBORATION_BONUS: {dim_count} independent evidence dimensions show real findings "
            f"({', '.join(sorted(dims))}) — multiple independent weak/moderate signals reinforcing each "
            f"other (-{CORROBORATION_BONUS_3PLUS_DIMENSIONS} pts, single bounded application)"
        )
    elif dim_count == 2:
        score -= CORROBORATION_BONUS_2_DIMENSIONS
        reasoning.append(
            f"CORROBORATION_BONUS: {dim_count} independent evidence dimensions show real findings "
            f"({', '.join(sorted(dims))}) (-{CORROBORATION_BONUS_2_DIMENSIONS} pts, single bounded application)"
        )

    if findings:
        errored_modules = [k for k, v in findings.items() if v.get(
            "module_status") == "error"]
        if errored_modules:
            penalty = min(
                INCOMPLETE_MODULE_PENALTY_PER_MODULE * len(errored_modules),
                INCOMPLETE_MODULE_PENALTY_CAP,
            )
            score -= penalty
            reasoning.append(
                f"INCOMPLETE_ANALYSIS_PENALTY: -{penalty} pts — {len(errored_modules)} module(s) failed to "
                f"complete ({', '.join(errored_modules)}); the score cannot fully reflect a completed analysis"
            )

    if not critical_present and score < NO_CRITICAL_EVIDENCE_SCORE_FLOOR:
        reasoning.append(
            f"SCORE_FLOOR_APPLIED: No critical/high-reliability evidence was found — score floored at "
            f"{NO_CRITICAL_EVIDENCE_SCORE_FLOOR} instead of {score} (weak/moderate heuristic evidence "
            "alone cannot establish manipulation)"
        )
        score = NO_CRITICAL_EVIDENCE_SCORE_FLOOR

    score = max(0, min(100, score))
    return score, reasoning


def classify_media_authenticity(
    score: int, dimension_severities: dict = None, media_type: str = ""
) -> str:
    dimension_severities = dimension_severities or {}
    if score < 40:
        if media_type == "image":
            critical_dimensions = sum(
                1 for severity in dimension_severities.values() if severity == "critical"
            )
            if critical_dimensions < 2:
                return "UNCERTAIN"
        return "MANIPULATED"
    if score >= 70:
        severities = list(dimension_severities.values())
        critical_dims = severities.count("critical")
        moderate_dims = severities.count("moderate")
        minor_dims = severities.count("minor")
        if critical_dims >= 1:
            return "UNCERTAIN"
        if moderate_dims >= 2:
            return "UNCERTAIN"
        if len(severities) >= 3:
            return "UNCERTAIN"
        if moderate_dims >= 1 and minor_dims >= 2:
            return "UNCERTAIN"
        return "AUTHENTIC"
    return "UNCERTAIN"


def compute_confidence(score: int, findings: dict) -> int:
    modules_run = len(findings)
    if modules_run == 0:
        return 0

    statuses = [f.get("module_status", "ok") for f in findings.values()]
    resolved_count = sum(1 for s in statuses if s in ("ok", "skipped"))
    coverage_ratio = resolved_count / modules_run

    real_flags = []
    for f in findings.values():
        for flag in f.get("flags", []):
            if not _is_noise_flag(flag):
                real_flags.append(flag)

    critical_n = sum(1 for f in real_flags if _flag_severity(f) == "critical")
    moderate_n = sum(1 for f in real_flags if _flag_severity(f) == "moderate")
    weak_n = sum(1 for f in real_flags if _flag_severity(f) == "minor")

    if not real_flags:
        base = 55 + coverage_ratio * 20
    elif critical_n >= 2 or (critical_n >= 1 and moderate_n >= 1):
        base = 82 + min(3 * (critical_n + moderate_n - 1), 12)
    elif critical_n == 1:
        base = 76
    elif moderate_n >= 2:
        base = 70
    elif moderate_n == 1:
        base = 66
    else:
        base = 60 + min(weak_n * 1.5, 8)

    base *= 0.5 + 0.5 * coverage_ratio

    confidence = int(round(max(0, min(100, base))))
    return confidence


def determine_evidence_strength(dimension_severities: dict) -> str:
    """Describe forensic evidence quality, not manipulation probability."""
    severities = list((dimension_severities or {}).values())
    critical_n = severities.count("critical")
    moderate_n = severities.count("moderate")
    if not severities:
        return "NONE"
    if critical_n >= 2 or (critical_n >= 1 and moderate_n >= 1):
        return "STRONG"
    if critical_n >= 1 or moderate_n >= 2 or len(severities) >= 2:
        return "MODERATE"
    return "WEAK"


def build_evidence_and_explainability(findings: dict) -> tuple:
    evidence = []
    explainability = []
    analysis_limitations = []
    tier_state = {"moderate": 0, "minor": 0}

    for module_key, module_result in findings.items():
        label = _MODULE_LABELS.get(module_key, module_key)
        status = module_result.get("module_status", "ok")
        raw_flags = module_result.get("flags", [])
        noise_flags = [f for f in raw_flags if _is_noise_flag(f)]
        module_flags = [f for f in raw_flags if not _is_noise_flag(f)]

        if status in {"not_applicable", "skipped", "unavailable", "partial"}:
            reason = (
                module_result.get("coverage_reason")
                or module_result.get("skip_reason")
                or module_result.get("inconclusive_evidence_text")
                or (noise_flags[0] if noise_flags else f"{module_key} {status}")
            )
            analysis_limitations.append(
                f"{label}: {status.upper()} — {reason}; no clean evidence was inferred."
            )
            explainability.append({
                "feature_checked": label,
                "result": str(reason),
                "confidence": 0,
                "contribution_to_score": 0,
                "execution_state": "DID_NOT_COMPLETE" if status == "partial" else "DID_NOT_RUN",
            })
            continue

        if status == "error":
            reason = noise_flags[0] if noise_flags else f"{module_key} unavailable"
            evidence.append(f"{label}: unavailable — {reason}")
            explainability.append(
                {
                    "feature_checked": label,
                    "result": reason,
                    "confidence": 0,
                    "contribution_to_score": 0,
                }
            )
            analysis_limitations.append(
                f"{label} did not complete and contributed no direct evidence (a bounded incompleteness penalty was applied to the overall score): {reason}"
            )
            continue

        if status == "skipped":
            reason = noise_flags[0] if noise_flags else f"{module_key} skipped"
            evidence.append(f"{label}: skipped — {reason}")
            explainability.append(
                {
                    "feature_checked": label,
                    "result": reason,
                    "confidence": 100,
                    "contribution_to_score": 0,
                }
            )
            analysis_limitations.append(
                f"{label} was intentionally skipped and had no effect on the score: {reason}"
            )
            continue

        info_flags = [f for f in module_flags if f.split(
            ":")[0].endswith("_FALLBACK_MODE")]
        module_flags = [f for f in module_flags if f not in info_flags]
        for info_flag in info_flags + noise_flags:
            analysis_limitations.append(f"{label}: {info_flag}")

        if not module_flags and module_result.get("clean_evidence_text"):
            clean_text = module_result["clean_evidence_text"]
            clean_confidence = module_result.get("clean_evidence_confidence", 0)
            evidence.append(clean_text)
            explainability.append(
                {
                    "feature_checked": label,
                    "result": clean_text,
                    "confidence": clean_confidence,
                    "contribution_to_score": 0,
                    "execution_state": "RAN_AND_FOUND_NOTHING",
                }
            )
        elif module_flags:
            for flag in module_flags:
                severity = _flag_severity(flag)
                if severity == "critical":
                    deduction, cap_note = 20, ""
                else:
                    deduction, cap_note = _apply_capped_deduction(
                        severity, tier_state)
                confidence_for_flag = {"critical": 90,
                                       "moderate": 70, "minor": 50}[severity]
                evidence.append(flag)
                explainability.append(
                    {
                        "feature_checked": label,
                        "result": flag + cap_note,
                        "confidence": confidence_for_flag,
                        "contribution_to_score": -deduction,
                        "execution_state": "RAN_AND_FOUND_SOMETHING",
                    }
                )

    return evidence, explainability, analysis_limitations


def determine_next_step(media_authenticity: str, confidence: int) -> str:
    if media_authenticity == "MANIPULATED" and confidence >= MANIPULATION_CONFIDENCE_THRESHOLD:
        return "STOP_AND_SEND_TO_DECISION_ENGINE"
    return "SEND_TO_CROSS_REFERENCE"


# ============================================================
# MAIN PIPELINE — RUN FORENSICS AGENT
# PATCH 2 integration: get_source_format_info() computed once and passed
# into run_ela_analysis_v2() instead of calling run_ela_analysis() directly.
# ============================================================

LIPSYNC_SPEECH_THRESHOLD = 0.10

_HEURISTIC_DISCLAIMER = (
    "This agent uses heuristic and signal-based forensic techniques (compression "
    "analysis, pixel statistics, YuNet DNN + Haar-cascade face detection, classical CV "
    "symmetry/texture/lighting checks, spectral audio features) rather than trained "
    "deep-learning manipulation classifiers. Results should be treated as "
    "screening-level evidence, combined with cross-referencing, and not relied "
    "upon in isolation for high-stakes decisions."
)


def run_forensics_agent(provenance_output: dict) -> dict:
    log.info("=" * 50)

    try:
        provenance_output = _coerce_agent_dict(
            provenance_output, "provenance_output")
    except (TypeError, ValueError) as exc:
        return {
            "status": "error",
            "agent": "forensics",
            "error_message": str(exc),
            "media_authenticity": "UNCERTAIN",
            "confidence": 0,
            "evidence_strength": "NONE",
            "ready_for_next_agent": False,
        }
    log.info("FORENSICS AGENT STARTING")
    log.info("=" * 50)

    if not provenance_output.get("ready_for_next_agent"):
        return {
            "status": "error",
            "error_message": "Previous agent did not complete successfully",
            "media_authenticity": "UNCERTAIN",
            "confidence": 0,
            "evidence_strength": "NONE",
            "forensics_score": 0,
            "evidence": ["Provenance agent did not complete — no forensic analysis was run"],
            "explainability": [],
            "analysis_limitations": [
                "No forensic analysis was performed because the provenance agent did not complete."
            ],
            "recommended_next_step": "STOP_AND_SEND_TO_DECISION_ENGINE",
            "ready_for_next_agent": False,
        }

    file_path = provenance_output["file_path"]
    media_type = provenance_output.get("media_type", "")

    if not media_type:
        ext = Path(file_path).suffix.lower()
        if ext in [".mp4", ".mov", ".avi", ".webm"]:
            media_type = "video"
        elif ext in [".jpg", ".jpeg", ".png", ".webp"]:
            media_type = "image"
        elif ext in [".mp3", ".wav", ".aac"]:
            media_type = "audio"

    if not os.path.exists(file_path):
        log.error(f"File not found: {file_path}")
        return {
            "status": "error",
            "error_message": f"File not found: {file_path}",
            "media_authenticity": "UNCERTAIN",
            "confidence": 0,
            "evidence_strength": "NONE",
            "forensics_score": 0,
            "evidence": [f"Could not locate media file at {file_path}"],
            "explainability": [],
            "analysis_limitations": [
                "No forensic analysis was performed because the media file could not be located."
            ],
            "recommended_next_step": "STOP_AND_SEND_TO_DECISION_ENGINE",
            "ready_for_next_agent": False,
        }

    try:
        all_flags = []
        findings = {}

        if media_type == "image":
            log.info("Running IMAGE forensics pipeline...")

            source_file_path = provenance_output.get(
                "source_file_path") or file_path
            format_info = get_source_format_info(source_file_path)
            ela = run_ela_analysis_v2(source_file_path, format_info)
            findings["ela"] = ela
            all_flags.extend(ela.get("flags", []))

            content_context = classify_image_content(file_path)
            findings["content_classification"] = content_context

            noise = analyze_noise_and_frequency(file_path)
            findings["noise_frequency"] = noise
            all_flags.extend(noise.get("flags", []))

            # Detect faces before applying the coarse layout gate. A portrait's
            # natural edges can otherwise be mistaken for rows of text.
            detected_faces = detect_face_artifacts(file_path)
            image = cv2.imread(file_path)
            image_area = int(image.shape[0] * image.shape[1]) if image is not None else 0
            largest_face_area = max(
                (int(box[2]) * int(box[3]) for box in detected_faces.get("face_boxes", [])),
                default=0,
            )
            dominant_face = (
                detected_faces.get("detection_confidence") in {"high", "medium"}
                and image_area > 0 and largest_face_area / image_area >= 0.04
            )
            if content_context.get("document_like") and not dominant_face:
                faces = {
                    "faces_found": 0,
                    "face_boxes": [],
                    "flags": [],
                    "module_status": "not_applicable",
                    "clean_evidence_text": (
                        "Face-artifact analysis is NOT_APPLICABLE to document or "
                        "multi-panel layout content."
                    ),
                    "clean_evidence_confidence": 0,
                    "skip_reason": "DOCUMENT_OR_MULTI_PANEL_LAYOUT_WITHOUT_DOMINANT_FACE",
                }
            else:
                faces = detected_faces
                if content_context.get("document_like") and dominant_face:
                    content_context["routing_override"] = "DOMINANT_FACE_DETECTED"
                    content_context["document_like"] = False
                    content_context["content_type"] = "PHOTO_SCENE_WITH_POSSIBLE_TEXT"
            findings["face_artifacts"] = faces
            all_flags.extend(faces.get("flags", []))

            face_boxes = faces.get("face_boxes", [])
            # PATCH 3: uses score_lighting_flag() internally
            if content_context.get("document_like"):
                ai_heuristics = {
                    "metrics": {},
                    "flags": [],
                    "module_status": "not_applicable",
                    "content_context": content_context,
                    "not_applicable_checks": [
                        "FACE_AND_SCENE_AI_HEURISTICS: NOT_APPLICABLE to document, screenshot, "
                        "collage, poster, or multi-panel layout content"
                    ],
                }
            else:
                ai_heuristics = analyze_ai_generation_heuristics(
                    file_path, face_boxes, content_context
                )
            findings["ai_heuristics"] = ai_heuristics
            all_flags.extend(ai_heuristics.get("flags", []))

            if content_context.get("document_like"):
                hand_artifacts = {
                    "flags": [],
                    "module_status": "not_applicable",
                    "clean_evidence_text": (
                        "Hand analysis is NOT_APPLICABLE to document or multi-panel layout content."
                    ),
                    "clean_evidence_confidence": 0,
                }
            else:
                hand_artifacts = detect_hand_artifacts(file_path)
            findings["hand_artifacts"] = hand_artifacts
            all_flags.extend(hand_artifacts.get("flags", []))

        elif media_type == "video":
            log.info("Running VIDEO forensics pipeline...")

            temporal = analyze_video_temporal_consistency(file_path)
            findings["temporal_consistency"] = temporal
            all_flags.extend(temporal.get("flags", []))

            audio = analyze_audio_spectrogram(file_path)
            findings["audio_analysis"] = audio
            all_flags.extend(audio.get("flags", []))

            voiced_ratio = audio.get("voiced_ratio")
            if audio.get("module_status") != "ok":
                skip_reason = (
                    "audio analysis did not complete, so speech content could not be assessed"
                )
            elif voiced_ratio is None:
                skip_reason = "no voiced/pitched speech segments were detected"
            elif voiced_ratio < LIPSYNC_SPEECH_THRESHOLD:
                skip_reason = f"voiced_ratio={voiced_ratio:.2f} is below the {LIPSYNC_SPEECH_THRESHOLD} minimum-speech threshold"
            else:
                skip_reason = None

            if skip_reason is not None:
                log.info(f"Skipping lip-sync analysis — {skip_reason}")
                lipsync = {
                    "frames_analyzed": 0,
                    "audio_video_correlation": None,
                    "interpretation": "Not evaluated — insufficient spoken audio",
                    "flags": [
                        f"LIP_SYNC_SKIPPED: Insufficient spoken audio to evaluate synchronization ({skip_reason})."
                    ],
                    "module_status": "skipped",
                }
            else:
                lipsync = detect_lipsync_drift(file_path)

            findings["lipsync"] = lipsync
            all_flags.extend(lipsync.get("flags", []))

        elif media_type == "audio":
            log.info("Running AUDIO forensics pipeline...")

            audio = analyze_audio_spectrogram(file_path)
            findings["audio_analysis"] = audio
            all_flags.extend(audio.get("flags", []))

        else:
            log.warning(f"Unsupported media_type: '{media_type}'")
            return {
                "status": "error",
                "error_message": f"Unsupported or unrecognized media type: '{media_type}'",
                "media_authenticity": "UNCERTAIN",
                "confidence": 0,
                "evidence_strength": "NONE",
                "forensics_score": 0,
                "evidence": [
                    f"Media type '{media_type}' is not supported by the forensics pipeline"
                ],
                "explainability": [],
                "analysis_limitations": [
                    f"No forensic analysis was performed because media type '{media_type}' is unsupported."
                ],
                "recommended_next_step": "STOP_AND_SEND_TO_DECISION_ENGINE",
                "ready_for_next_agent": False,
            }

        forensics_score, score_reasoning = calculate_forensics_score(
            all_flags, media_type, findings
        )
        dimension_severities = _tally_corroboration(all_flags)
        evidence_strength = determine_evidence_strength(dimension_severities)
        media_authenticity = classify_media_authenticity(
            forensics_score, dimension_severities, media_type
        )
        if (
            media_type == "image"
            and media_authenticity == "AUTHENTIC"
            and not provenance_output.get("c2pa", {}).get("c2pa_valid")
            and provenance_output.get("hash_lookup", {}).get("verdict") != "VERIFIED_AUTHENTIC"
        ):
            media_authenticity = "UNCERTAIN"
            score_reasoning.append(
                "AUTHENTICITY_DOWNGRADE: Heuristic checks alone cannot establish image authenticity; "
                "no verified positive provenance was available."
            )
        confidence = compute_confidence(forensics_score, findings)
        evidence, explainability, analysis_limitations = build_evidence_and_explainability(
            findings)
        next_step = determine_next_step(media_authenticity, confidence)

        if media_authenticity == "UNCERTAIN" and forensics_score >= 70:
            score_reasoning.append(
                f"CORROBORATION_DOWNGRADE: forensics_score={forensics_score} was in the AUTHENTIC band, but "
                f"{len(dimension_severities)} independent evidence dimension(s) showed real findings together "
                f"({', '.join(sorted(dimension_severities))}), so the verdict was downgraded to UNCERTAIN "
                "rather than reported as AUTHENTIC."
            )

        analysis_limitations.append(_HEURISTIC_DISCLAIMER)
        module_execution = {}
        for name, module in findings.items():
            state = module.get("module_status", "unknown")
            reason = module.get("skip_reason") or module.get("coverage_reason")
            module_execution[name] = {"status": state, "reason": reason}
            log.info(
                "Forensics module %s: status=%s reason=%s flags=%d",
                name, state, reason or "none", len(module.get("flags", [])),
            )

        result = {
            "status": "success",
            "processed_at": datetime.now().isoformat(),
            "agent": "forensics",
            "file_path": file_path,
            "media_type": media_type,
            "media_authenticity": media_authenticity,
            "confidence": confidence,
            "evidence_strength": evidence_strength,
            "forensics_score": forensics_score,
            "evidence": evidence,
            "recommended_next_step": next_step,
            "explainability": explainability,
            "analysis_limitations": analysis_limitations,
            "score_reasoning": score_reasoning,
            "raw_findings": findings,
            "module_execution": module_execution,
            "modules_not_run": [
                {"module": name, **details}
                for name, details in module_execution.items()
                if details["status"] in {"not_applicable", "skipped", "unavailable", "error"}
            ],
            "all_flags": all_flags,
            "total_flags_found": len(all_flags),
            "ready_for_next_agent": True,
        }

        log.info(
            f"FORENSICS AGENT COMPLETE ✓ — Verdict: {media_authenticity}, "
            f"Score: {forensics_score}/100, Confidence: {confidence}, "
            f"Next step: {next_step}"
        )
        return result

    except Exception as e:
        log.error(f"Forensics Agent FAILED: {str(e)}")
        return {
            "status": "error",
            "error_message": str(e),
            "processed_at": datetime.now().isoformat(),
            "media_authenticity": "UNCERTAIN",
            "confidence": 0,
            "evidence_strength": "NONE",
            "forensics_score": 0,
            "evidence": [f"Forensics pipeline crashed: {str(e)}"],
            "explainability": [],
            "analysis_limitations": [
                "No forensic conclusions should be drawn — the pipeline crashed before completing."
            ],
            "recommended_next_step": "STOP_AND_SEND_TO_DECISION_ENGINE",
            "ready_for_next_agent": False,
        }


# ============================================================
# ── HOW TO CHANGE FOR YOUR OWN TEST ──
#
# 1. Run Agent 2 and get its JSON output
# 2. Paste2 that JSON output into agent2_output below
# 3. Make sure "file_path" points to your actual canonical file
# 4. Run this cell Eti3
# ============================================================

# Legacy pasted-JSON fixture retained only as reference; never executed.
if False:

    # ── PASTE AGENT 2 OUTPUT HERE AS RAW JSON ──
    agent2_output = r"""
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

    # JSON text → Python dictionary
    agent2_output = json.loads(agent2_output)

    print("\n" + "=" * 60)
    print("  RUNNING FORENSICS AGENT")
    print("=" * 60)

    result = run_forensics_agent(agent2_output)

    print("\n>>> FINAL FORENSICS RESULT:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    import sys
    from agent.ingestor import run_ingestor_agent
    from agent.provinence import run_provenance_agent

    local_image = sys.argv[1] if len(sys.argv) > 1 else "imageprocessor.jpeg"
    ingestor_result = run_ingestor_agent(local_image)
    provenance_result = run_provenance_agent(ingestor_result)
    result = run_forensics_agent(provenance_result)
    print(json.dumps(result, indent=2, default=str))
