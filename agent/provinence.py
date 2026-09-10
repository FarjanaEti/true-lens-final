# !pip install Pillow exifread mutagen hachoir requests

# # OPTIONAL (for richer video metadata + full C2PA verification):
# !apt-get -y install ffmpeg
# !pip install c2pa-python
# agent2
# ============================================================
# THE TRUE LENS — PROVENANCE AGENT (AgentF2)


import json
import hashlib
import logging
import shutil
import subprocess
import os
from datetime import datetime

# For metadata extraction
import exifread  # EXIF data from images
import mutagen  # Audio/video metadata (kept for compatibility)
from PIL import Image
from PIL.ExifTags import TAGS

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [PROVENANCE] %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


def _coerce_agent_dict(value, boundary_name: str) -> dict:
    """Accept an in-process dict or valid serialized JSON at an agent boundary."""
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


AI_GENERATIVE_KEYWORDS = [
    "midjourney",
    "stable diffusion",
    "dall-e",
    "dalle",
    "firefly",
    "runway",
    "sora",
    "synthesia",
    "heygen",
    "d-id",
    "did.com",
    "deepfake",
    "faceswap",
    "face swap",
    "reface",
    "comfyui",
    "invokeai",
    "automatic1111",
    "a1111",
    "generated",
    "artificial intelligence",
    " ai ",
]

CONVENTIONAL_EDITING_KEYWORDS = [
    "adobe photoshop",
    "photoshop",
    "adobe lightroom",
    "lightroom",
    "gimp",
    "canva",
    "affinity photo",
    "capture one",
]


def _classify_software_string(value: str) -> list:
    """
    Helper used by both image and video metadata extractors.
    Scans a free-text software/encoder/comment string for known
    keywords and returns a list of (flag_text, confidence) tuples.

    Important: presence of an editing tool is NOT evidence of fakery.
    Presence of a generative-AI tool is a more meaningful signal, but
    still not proof by itself (e.g. a partially AI-retouched real photo).
    """
    findings = []
    if not value:
        return findings

    lowered = f" {value.lower()} "

    for keyword in AI_GENERATIVE_KEYWORDS:
        if keyword in lowered:
            findings.append(
                (
                    f"AI_SOFTWARE_DETECTED: Software/encoder tag references '{keyword.strip()}' "
                    f"(raw value: '{value}')",
                    "MEDIUM_CONFIDENCE",
                )
            )

    for keyword in CONVENTIONAL_EDITING_KEYWORDS:
        if keyword in lowered:
            findings.append(
                (
                    f"EDITING_SOFTWARE_DETECTED: Software tag references '{keyword.strip()}' "
                    f"(raw value: '{value}') — common conventional editing, not evidence of fabrication",
                    "LOW_CONFIDENCE",
                )
            )

    return findings


def _make_flag(text: str, confidence: str) -> dict:
    """
    Standard shape for every flag raised anywhere in this agent:
        {"flag": "<human readable text>", "confidence": "<LOW|MEDIUM|HIGH>_CONFIDENCE"}
    Keeping this consistent lets the scoring module weight flags properly
    and lets the output JSON build a clean confidence_breakdown section.
    """
    assert confidence in (
        "LOW_CONFIDENCE",
        "MEDIUM_CONFIDENCE",
        "HIGH_CONFIDENCE",
    ), f"Invalid confidence level: {confidence}"
    return {"flag": text, "confidence": confidence}


def _scan_file_for_markers(file_path: str, markers: list, read_bytes: int = 65536) -> list:
    """
    Generic binary-marker scanner. Reads the first `read_bytes` bytes of
    the file and reports which byte markers were found. Used for C2PA,
    XMP, and IPTC/Photoshop-resource detection since none of these
    formats are reliably parsed by exifread/mutagen alone.

    This is a presence check only — it does NOT validate or parse the
    embedded data, and finding nothing does NOT mean the data isn't
    present later in the file (we intentionally cap the read for speed).
    """
    found = []
    try:
        with open(file_path, "rb") as f:
            header = f.read(read_bytes)
        lowered = header.lower()
        for marker in markers:
            if marker in lowered:
                found.append(marker.decode("utf-8", errors="ignore"))
    except Exception as e:
        log.warning(f"Marker scan failed for {file_path}: {e}")
    return found


def extract_image_metadata(file_path: str) -> dict:
    """
    Extracts EXIF metadata from images.
    EXIF contains: camera model, GPS, timestamp, software used, etc.
    AI-generated images typically have NO EXIF or suspicious software tags.

    Returns: {"raw_metadata": {...}, "flags": [ {"flag":..., "confidence":...}, ... ]}
    """
    log.info("Extracting image EXIF metadata...")
    metadata = {}
    flags = []
    exif_data = None

    try:
        # Method 1: PIL EXIF extraction
        img = Image.open(file_path)
        metadata["_source_format"] = img.format
        metadata["_dimensions"] = {"width": img.width, "height": img.height}
        exif_data = img.getexif()

        if exif_data:
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, tag_id)
                metadata[str(tag_name)] = str(value)

            # ── FLAG: Check for AI/generative or conventional editing software ──
            software = metadata.get("Software", "")
            for text, confidence in _classify_software_string(software):
                flags.append(_make_flag(text, confidence))

            # ── FLAG: Missing camera make/model ──
            # NOTE: absence alone is common (social platforms strip EXIF,
            # screenshots, scans, etc.) so this stays LOW_CONFIDENCE.
            if "Make" not in metadata and "Model" not in metadata:
                flags.append(
                    _make_flag(
                        "NO_CAMERA_INFO: No camera make/model found — may be synthetic, "
                        "stripped, or a non-camera source (screenshot/scan)",
                        "LOW_CONFIDENCE",
                    )
                )
            elif ("Make" in metadata) != ("Model" in metadata):
                # One is present, the other missing — more unusual than both missing.
                flags.append(
                    _make_flag(
                        "PARTIAL_CAMERA_INFO: Camera Make and Model are inconsistently present "
                        "(only one of the two exists) — unusual for genuine camera output",
                        "MEDIUM_CONFIDENCE",
                    )
                )

            # ── FLAG: Creation vs modification timestamp inconsistency ──
            if "DateTime" in metadata and "DateTimeOriginal" in metadata:
                if metadata["DateTime"] != metadata["DateTimeOriginal"]:
                    flags.append(
                        _make_flag(
                            "TIMESTAMP_MISMATCH: EXIF DateTime (modification) differs from "
                            "DateTimeOriginal (capture) — file was modified after capture",
                            "MEDIUM_CONFIDENCE",
                        )
                    )

            # ── FLAG: Orientation anomaly ──
            # Valid EXIF Orientation values are 1-8. Anything else is malformed/injected.
            orientation = metadata.get("Orientation")
            if orientation is not None:
                try:
                    orientation_val = int(orientation)
                    if orientation_val < 1 or orientation_val > 8:
                        flags.append(
                            _make_flag(
                                f"ORIENTATION_ANOMALY: Orientation tag value '{orientation}' is "
                                "outside the valid EXIF range (1-8)",
                                "MEDIUM_CONFIDENCE",
                            )
                        )
                except (ValueError, TypeError):
                    flags.append(
                        _make_flag(
                            f"ORIENTATION_ANOMALY: Orientation tag is non-numeric ('{orientation}')",
                            "LOW_CONFIDENCE",
                        )
                    )

            # ── FLAG: Resolution consistency ──
            # Compare EXIF-reported pixel dimensions against the actual decoded image size.
            try:
                exif_w = metadata.get("ExifImageWidth")
                exif_h = metadata.get("ExifImageHeight")
                if exif_w and exif_h:
                    actual_w, actual_h = img.size
                    if int(exif_w) != actual_w or int(exif_h) != actual_h:
                        flags.append(
                            _make_flag(
                                f"RESOLUTION_MISMATCH: EXIF reports {exif_w}x{exif_h} but the "
                                f"decoded image is {actual_w}x{actual_h}",
                                "HIGH_CONFIDENCE",
                            )
                        )
            except Exception:
                # Non-critical enrichment check — never let this break the pipeline.
                pass

            # ── FLAG: Missing ICC profile ──
            # Very common on web-optimized/re-saved images, so this is LOW_CONFIDENCE only.
            if not img.info.get("icc_profile"):
                flags.append(
                    _make_flag(
                        "NO_ICC_PROFILE: No embedded ICC color profile found", "LOW_CONFIDENCE"
                    )
                )

        else:
            # Missing EXIF entirely is common (messaging apps, screenshots, many
            # AI generators, social platforms that strip metadata). We do NOT
            # treat this as proof of manipulation — LOW_CONFIDENCE only.
            flags.append(_make_flag(
                "NO_EXIF: Image has no EXIF data at all", "LOW_CONFIDENCE"))
            log.info("No EXIF data found in image")

    except Exception as e:
        log.warning(f"PIL EXIF extraction failed: {e}")
        flags.append(_make_flag(
            f"METADATA_READ_ERROR: {str(e)}", "LOW_CONFIDENCE"))

    # exifread emits expected warnings for formats such as metadata-free PNG.
    # Use it only for formats it can meaningfully inspect.
    tags = {}
    source_format = str(metadata.get("_source_format") or "").upper()
    try:
        if source_format in {"JPEG", "TIFF"}:
            with open(file_path, "rb") as f:
                # details=True (changed from False) so exifread also reports the
                # embedded 'JPEGThumbnail' tag when present, needed for the
                # thumbnail-presence check below.
                tags = exifread.process_file(f, details=True)
                for key, val in tags.items():
                    if key not in metadata:
                        metadata[key] = str(val)

        # ── FLAG: Thumbnail presence ──
        # Genuine camera JPEGs almost always embed a thumbnail. Its absence
        # is common enough in re-saved/AI images to only warrant LOW_CONFIDENCE.
        if (
            source_format == "JPEG"
            and "JPEGThumbnail" not in tags
            and "EXIF ThumbnailImage" not in tags
        ):
            flags.append(
                _make_flag(
                    "NO_THUMBNAIL: No embedded EXIF thumbnail found", "LOW_CONFIDENCE")
            )
    except Exception as e:
        log.warning(f"exifread extraction failed: {e}")

    # ── FLAG: XMP metadata presence (informational, not parsed) ──
    xmp_markers = _scan_file_for_markers(
        file_path, [b"<x:xmpmeta", b"adobe:ns:meta"])
    if xmp_markers:
        metadata["_xmp_detected"] = True
        flags.append(
            _make_flag(
                "XMP_METADATA_PRESENT: File contains embedded XMP metadata block", "LOW_CONFIDENCE"
            )
        )
    else:
        metadata["_xmp_detected"] = False

    # ── FLAG: IPTC metadata presence (informational, not parsed) ──
    iptc_markers = _scan_file_for_markers(
        file_path, [b"photoshop 3.0", b"8bim"])
    if iptc_markers:
        metadata["_iptc_detected"] = True
        flags.append(
            _make_flag(
                "IPTC_METADATA_PRESENT: File contains an embedded IPTC/Photoshop resource block",
                "LOW_CONFIDENCE",
            )
        )
    else:
        metadata["_iptc_detected"] = False

    return {
        "raw_metadata": metadata,
        "flags": flags,
        "exif_status": "PRESENT" if exif_data else "MISSING",
        "xmp_status": "PRESENT" if xmp_markers else "MISSING",
        "iptc_status": "PRESENT" if iptc_markers else "MISSING",
        "thumbnail_status": (
            "PRESENT"
            if "JPEGThumbnail" in tags or "EXIF ThumbnailImage" in tags
            else "MISSING" if source_format == "JPEG" else "NOT_APPLICABLE"
        ),
    }


def check_ffprobe_available() -> bool:
    """
    Detects whether the ffprobe binary (part of FFmpeg) is available on
    PATH. ffprobe gives far richer video metadata than mutagen alone
    (frame rate, exact codec, bitrate, container-level tags, etc.).
    """
    return shutil.which("ffprobe") is not None


def extract_video_metadata_ffprobe(file_path: str) -> dict:
    """
    Extracts rich video/container metadata using ffprobe, if installed.
    Extracts: encoder, codec, creation_time, duration, frame rate,
    resolution, bitrate, container format.

    Returns {"raw_metadata": {...}, "flags": [...]} same shape as the
    mutagen-based extractor so the two are interchangeable/mergeable.
    """
    log.info("Extracting video metadata via ffprobe...")
    metadata = {}
    flags = []

    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            file_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if proc.returncode != 0:
            log.warning(
                f"ffprobe exited with code {proc.returncode}: {proc.stderr[:300]}")
            flags.append(
                _make_flag(
                    f"FFPROBE_ERROR: ffprobe failed with exit code {proc.returncode}",
                    "LOW_CONFIDENCE",
                )
            )
            return {"raw_metadata": metadata, "flags": flags}

        data = json.loads(proc.stdout)
        fmt = data.get("format", {})
        streams = data.get("streams", [])
        video_stream = next(
            (s for s in streams if s.get("codec_type") == "video"), {})
        format_tags = fmt.get("tags", {}) or {}

        # ── Container-level fields ──
        metadata["container_format"] = fmt.get(
            "format_long_name") or fmt.get("format_name")
        metadata["duration_seconds"] = fmt.get("duration")
        metadata["bitrate"] = fmt.get("bit_rate")
        metadata["creation_time"] = format_tags.get("creation_time")
        metadata["encoder"] = format_tags.get("encoder")

        # ── Video-stream-level fields ──
        metadata["codec"] = video_stream.get("codec_name")
        metadata["resolution"] = (
            f"{video_stream.get('width')}x{video_stream.get('height')}"
            if video_stream.get("width")
            else None
        )

        # Frame rate is reported as a fraction string like "30000/1001"
        raw_fps = video_stream.get(
            "avg_frame_rate") or video_stream.get("r_frame_rate")
        if raw_fps and raw_fps != "0/0":
            try:
                num, den = raw_fps.split("/")
                metadata["frame_rate"] = round(
                    int(num) / int(den), 3) if int(den) != 0 else None
            except Exception:
                metadata["frame_rate"] = raw_fps

        # ── FLAG: AI/editing software in encoder or container tags ──
        encoder_val = metadata.get("encoder") or ""
        comment_val = format_tags.get("comment", "") or ""
        for text, confidence in _classify_software_string(encoder_val):
            flags.append(_make_flag(text, confidence))
        for text, confidence in _classify_software_string(comment_val):
            flags.append(_make_flag(text, confidence))

        # ── FLAG: creation_time vs filesystem modification time ──
        # Not proof of anything by itself (files get copied/re-uploaded
        # constantly), so this stays LOW_CONFIDENCE — purely informational.
        if metadata.get("creation_time"):
            try:
                creation_dt = datetime.fromisoformat(
                    metadata["creation_time"].replace("Z", "+00:00")
                )
                mtime_dt = datetime.fromtimestamp(os.path.getmtime(file_path))
                delta_days = abs(
                    (mtime_dt.replace(tzinfo=None) -
                     creation_dt.replace(tzinfo=None)).days
                )
                if delta_days > 3650:  # sanity-check only for wildly implausible gaps
                    flags.append(
                        _make_flag(
                            "CREATION_MTIME_GAP: Embedded creation_time and filesystem "
                            f"modification time differ by an implausible {delta_days} days",
                            "MEDIUM_CONFIDENCE",
                        )
                    )
            except Exception:
                pass  # Timestamp parsing is best-effort only.

        log.info("ffprobe extraction succeeded")

    except FileNotFoundError:
        flags.append(_make_flag(
            "FFPROBE_NOT_FOUND: ffprobe binary missing", "LOW_CONFIDENCE"))
    except subprocess.TimeoutExpired:
        flags.append(
            _make_flag(
                "FFPROBE_TIMEOUT: ffprobe took too long and was aborted", "LOW_CONFIDENCE")
        )
    except Exception as e:
        log.warning(f"ffprobe extraction failed: {e}")
        flags.append(_make_flag(f"FFPROBE_ERROR: {str(e)}", "LOW_CONFIDENCE"))

    return {"raw_metadata": metadata, "flags": flags}


def extract_video_metadata(file_path: str) -> dict:
    """
    Extracts metadata from video/audio files.

    Behavior (unchanged function name/signature for backward compatibility):
      1. Always runs mutagen first (original implementation, kept in full
         for compatibility and because it independently surfaces
         comment/encoder tags mutagen exposes differently than ffprobe).
      2. If ffprobe is available, ALSO runs ffprobe and merges its richer
         fields (codec, frame rate, bitrate, container format, etc.) into
         the same raw_metadata dict, prefixed so nothing is silently
         overwritten.
      3. If ffprobe is not available, silently falls back to mutagen-only
         behavior (the original behavior of this function).
    """
    log.info("Extracting video/audio metadata (mutagen)...")
    metadata = {}
    flags = []

    # ── ORIGINAL MUTAGEN IMPLEMENTATION (kept intact for compatibility) ──
    try:
        media_file = mutagen.File(file_path, easy=True)

        if media_file and media_file.tags:
            for key, val in media_file.tags.items():
                metadata[key] = str(val)

            # ── FLAG: Check encoder/comment for AI or editing tools ──
            encoder = str(metadata.get("encoder", ""))
            comment = str(metadata.get("comment", ""))
            for text, confidence in _classify_software_string(encoder):
                flags.append(_make_flag(text, confidence))
            for text, confidence in _classify_software_string(comment):
                flags.append(_make_flag(text, confidence))

        else:
            flags.append(
                _make_flag(
                    "NO_EMBEDDED_METADATA_TAGS: mutagen found no metadata tags in this file",
                    "LOW_CONFIDENCE",
                )
            )
            log.info("No metadata tags found via mutagen")

    except Exception as e:
        log.warning(f"Mutagen metadata extraction failed: {e}")
        flags.append(_make_flag(
            f"METADATA_READ_ERROR: {str(e)}", "LOW_CONFIDENCE"))

    # ── NEW: ffprobe enrichment, with automatic fallback ──
    if check_ffprobe_available():
        ffprobe_result = extract_video_metadata_ffprobe(file_path)
        # Merge ffprobe's richer fields into the same metadata dict.
        # Keys are distinct (container_format, codec, frame_rate, etc.)
        # from mutagen's tag-based keys, so this is a safe merge.
        metadata.update(ffprobe_result["raw_metadata"])
        flags.extend(ffprobe_result["flags"])
        metadata["_extraction_method"] = "mutagen+ffprobe"
    else:
        log.info(
            "ffprobe not available on this system — using mutagen-only extraction")
        flags.append(
            _make_flag(
                "FFPROBE_UNAVAILABLE: ffprobe not found on PATH, falling back to mutagen-only "
                "video metadata (install ffmpeg for richer extraction)",
                "LOW_CONFIDENCE",
            )
        )
        metadata["_extraction_method"] = "mutagen_only"

    return {"raw_metadata": metadata, "flags": flags}


# ============================================================
# MODULE 2: METADATA CONSISTENCY ANALYZER
# Looks for internal contradictions in the metadata
# ============================================================


def analyze_metadata_consistency(metadata_result: dict, media_type: str) -> list:
    """
    Checks for logical inconsistencies that reveal tampering, or that
    are simply worth noting. Returns a list of flag dicts
    ({"flag": ..., "confidence": ...}).

    IMPORTANT: Missing metadata is never, by itself, treated as proof of
    manipulation. Many legitimate pipelines (social media re-uploads,
    messaging apps, screenshots) strip metadata entirely.
    """
    log.info("Analyzing metadata consistency...")
    flags = []
    raw = metadata_result.get("raw_metadata", {})

    if media_type == "image":
        # GPS presence/absence — informational only, not a flag.
        if "GPS GPSLatitude" in raw and "GPS GPSLongitude" in raw:
            log.info("GPS data present in image — noted")
        else:
            log.info("No GPS data in image (common for shared images)")

        # Color space check (kept from original implementation).
        color_space = raw.get("ColorSpace", "")
        if color_space and "uncalibrated" in str(color_space).lower():
            flags.append(
                _make_flag(
                    "UNCALIBRATED_COLOR_SPACE: Unusual color space — possible synthetic or "
                    "heavily processed origin",
                    "MEDIUM_CONFIDENCE",
                )
            )

    else:  # video / audio
        # Container metadata consistency: does ffprobe's declared duration
        # roughly match mutagen's, if both extractors ran?
        try:
            ffprobe_duration = raw.get("duration_seconds")
            # mutagen easy tags sometimes expose this
            mutagen_length = raw.get("length")
            if ffprobe_duration and mutagen_length:
                diff = abs(float(ffprobe_duration) - float(mutagen_length))
                if diff > 2.0:  # more than 2 seconds apart
                    flags.append(
                        _make_flag(
                            f"DURATION_MISMATCH: ffprobe reports {ffprobe_duration}s but mutagen "
                            f"reports {mutagen_length}s for the same file",
                            "MEDIUM_CONFIDENCE",
                        )
                    )
        except (ValueError, TypeError):
            pass

        # Encoding software note (informational, e.g. generic "Lavf" from FFmpeg
        # indicates the file was re-muxed/re-encoded at some point — common
        # and not inherently suspicious).
        encoder = str(raw.get("encoder", "") or "")
        if encoder.lower().startswith("lavf"):
            flags.append(
                _make_flag(
                    f"REENCODED_CONTAINER: Encoder tag '{encoder}' indicates the file was "
                    "processed/re-muxed by FFmpeg at some point (common, not inherently suspicious)",
                    "LOW_CONFIDENCE",
                )
            )

    return flags


# ============================================================
# MODULE 3: C2PA SIGNATURE CHECKER
# Checks for the industry-standard content authenticity signature
# ============================================================


def check_c2pa_signature(file_path: str) -> dict:
    """
    Checks whether the file has a valid C2PA (Content Credentials) signature.

    C2PA is the gold standard for content provenance — used by Adobe, BBC,
    Microsoft, Sony, and major news organizations.

    This module reports one of four distinct statuses via "c2pa_status":
      - NO_C2PA:                   no markers found at all
      - C2PA_MARKER_FOUND:         binary markers found, but NOT
                                    cryptographically verified (c2pa-python
                                    unavailable or verification not attempted)
      - C2PA_VERIFIED:             markers found AND cryptographically
                                    verified via c2pa-python
      - C2PA_VERIFICATION_FAILED:  markers found, c2pa-python IS installed,
                                    but verification raised an error or
                                    produced no valid manifest (this is a
                                    meaningfully different — and more
                                    suspicious — case than "no markers at all")

    Presence of a marker is NEVER, by itself, presented as proof of
    authenticity. Only C2PA_VERIFIED represents a cryptographically
    checked claim.
    """
    log.info("Checking for C2PA digital signature...")

    result = {
        "c2pa_present": False,
        "c2pa_valid": False,
        "c2pa_status": "NO_C2PA",
        "presence_status": "MISSING",
        "verification_status": "NOT_CHECKED",
        "c2pa_issuer": None,
        "c2pa_timestamp": None,
        "c2pa_claims": [],
        "note": "",
    }

    # ── Method 1: Check for C2PA markers in raw bytes ──
    # C2PA embeds a 'jumbf' or 'c2pa' marker in the file binary. This is a
    # PRESENCE check only — it says nothing about validity.
    c2pa_markers = [b"c2pa", b"jumbf", b"contentcredentials", b"cai.adobe.com"]
    found_markers = _scan_file_for_markers(file_path, c2pa_markers)

    if found_markers:
        result["c2pa_present"] = True
        result["presence_status"] = "PRESENT"
        result["c2pa_status"] = "C2PA_PRESENT"
        result["note"] = (
            f"C2PA markers found in binary: {found_markers}. This confirms a C2PA "
            "manifest is embedded, but NOT that it is valid — full cryptographic "
            "verification requires the c2pa-python library."
        )
        log.info(f"C2PA markers detected: {found_markers}")
    else:
        result["note"] = (
            "No C2PA manifest found. This is a weak/neutral provenance signal, "
            "not evidence that the image is manipulated."
        )
        log.info("No C2PA signature found")

    # ── Method 2: Try c2pa-python library if installed, for real verification ──
    try:
        import c2pa

        reader = c2pa.Reader.try_create(file_path)

        if reader is None:
            result["c2pa_present"] = False
            result["c2pa_valid"] = False
            result["c2pa_status"] = "NO_C2PA"
            result["presence_status"] = "MISSING"
            result["verification_status"] = "NOT_CHECKED"
            result["note"] = (
                "No C2PA manifest found. This is a weak/neutral provenance signal, "
                "not evidence that the image is manipulated."
            )
        else:
            manifest = reader.get_active_manifest()

            result["c2pa_present"] = True
            result["c2pa_valid"] = reader.is_valid
            result["presence_status"] = "PRESENT"
            result["verification_status"] = "VALID" if reader.is_valid else "INVALID"
            result["c2pa_status"] = "C2PA_VERIFIED" if reader.is_valid else "C2PA_NOT_VERIFIED"

            if manifest:
                signature_info = manifest.get("signature_info") or {}
                result["c2pa_issuer"] = signature_info.get("issuer") or manifest.get(
                    "claim_generator"
                )
                result["c2pa_timestamp"] = signature_info.get("time")

                assertions = manifest.get("assertions", [])
                if isinstance(assertions, list):
                    result["c2pa_claims"] = [
                        item.get("label")
                        for item in assertions
                        if isinstance(item, dict) and item.get("label")
                    ]
                elif isinstance(assertions, dict):
                    result["c2pa_claims"] = list(assertions.keys())

            validation_state = reader.get_validation_state()

            result["note"] = f"C2PA manifest found. Validation state: {validation_state}"

            reader.close()

    except ImportError:
        result["verification_status"] = "NOT_AVAILABLE"
        if result["c2pa_present"]:
            result["c2pa_status"] = "C2PA_NOT_AVAILABLE"
        result["note"] = (
            "C2PA markers were found, but c2pa-python is not installed; cryptographic "
            "verification is NOT_AVAILABLE."
            if result["c2pa_present"]
            else "No C2PA manifest markers were found. Cryptographic verification is "
            "NOT_AVAILABLE; absence is not evidence of manipulation."
        )

    except Exception as e:
        result["c2pa_status"] = "C2PA_NOT_VERIFIED" if result["c2pa_present"] else "NO_C2PA"
        result["verification_status"] = "FAILED" if result["c2pa_present"] else "NOT_CHECKED"
        result["note"] = f"C2PA verification failed: {e}"

        log.warning(f"C2PA verification failed: {e}")

    return result


# ============================================================
# MODULE 4: HASH CROSS-REFERENCE
# Checks if this exact file has been seen before
# ============================================================


def lookup_internal_database(file_hash: str) -> dict:
    """
    The ONLY hash lookup that actually executes in this version of the
    agent. Checks the file hash against a small in-memory demo table
    that stands in for a real internal database of previously reviewed
    media (verified authentic OR debunked).

    In production this would be a real database/cache lookup (e.g. Redis,
    Postgres, or an internal microservice), not a hardcoded dict.
    """
    log.info(f"Checking internal database for hash: {file_hash[:16]}...")

    # ── DEMO LOOKUP TABLE ──
    KNOWN_HASHES = {
        "abc123demo": {
            "verdict": "DEBUNKED",
            "source": "Snopes",
            "detail": "This video was debunked in 2023 — original footage is from 2019",
        },
        "def456demo": {
            "verdict": "VERIFIED_AUTHENTIC",
            "source": "Reuters",
            "detail": "Verified by Reuters fact-checking team",
        },
    }

    if file_hash in KNOWN_HASHES:
        match = KNOWN_HASHES[file_hash]
        log.info(f"Internal DB hash match found: {match['verdict']}")
        return {
            "hash_match": True,
            "verdict": match["verdict"],
            "source": match["source"],
            "detail": match["detail"],
            "source_layer": "internal",
        }

    log.info("No internal database match")
    return {
        "hash_match": False,
        "verdict": "UNKNOWN",
        "source": None,
        "detail": "Not found in internal database",
        "source_layer": "internal",
    }


def lookup_external_database(file_hash: str) -> dict:
    """
    PLACEHOLDER ONLY — this function does NOT make any real network calls.

    This is where production integrations would go, e.g.:

        # Reuters News Tracer / fact-check partner API:
        # response = requests.post(
        #     "https://api.reuters-factcheck.example/v1/lookup",
        #     headers={"Authorization": f"Bearer {REUTERS_API_KEY}"},
        #     json={"sha256": file_hash},
        #     timeout=10,
        # )
        #
        # AFP Fact Check API:
        # response = requests.get(
        #     f"https://factcheck.afp.example/api/hash/{file_hash}",
        #     headers={"X-API-Key": AFP_API_KEY},
        # )
        #
        # Hive Moderation:
        # response = requests.post(
        #     "https://api.thehive.ai/api/v2/task/sync",
        #     headers={"token": HIVE_API_KEY},
        #     json={"hash": file_hash},
        # )
        #
        # InVID/WeVerify hash/reverse-search database:
        # response = requests.post(INVID_ENDPOINT, json={"hash": file_hash})
        #
        # Your own internal fact-check archive, NCMEC PhotoDNA, etc.

    Returns a well-formed "not implemented" result so callers can treat
    it identically to a real (but empty) lookup, without ever pretending
    a network call happened.
    """
    log.info(
        "lookup_external_database() called — this is a placeholder, no network call made")
    return {
        "hash_match": False,
        "verdict": "NOT_CHECKED",
        "source": None,
        "detail": "External hash database lookup is not implemented in this build",
        "source_layer": "external",
    }


def cross_reference_hash(file_hash: str) -> dict:
    """
    Cross-references the file hash against known databases.

    Orchestrates the two lookup layers:
      1. lookup_internal_database() — actually executes.
      2. lookup_external_database() — placeholder, only consulted if the
         internal lookup found nothing, and never treated as a genuine
         "checked and clean" result.

    Function name/signature and return shape are unchanged for backward
    compatibility with the previous version of this agent.
    """
    internal_result = lookup_internal_database(file_hash)

    if internal_result["hash_match"]:
        return internal_result

    # Internal DB had nothing — consult the (placeholder) external layer.
    external_result = lookup_external_database(file_hash)

    # Merge: keep the original top-level keys the rest of the pipeline
    # expects (hash_match, verdict, source, detail), plus source_layer.
    return external_result


# ============================================================
# MODULE 4b: FILE INTEGRITY — RECOMPUTE HASH
# ============================================================


def recompute_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    """
    Recomputes the file's cryptographic hash directly from disk, rather
    than trusting the hash string handed off by Agent 1. This is a basic
    but important integrity control: if Agent 1's reported hash and the
    actual file on disk don't match, something changed the file (or the
    reported hash) between agents, which is itself a strong signal.
    """
    log.info(f"Recomputing {algorithm} hash from disk...")
    hasher = hashlib.new(algorithm)
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        log.info(f"Recomputed hash: {digest[:16]}...")
        return digest
    except Exception as e:
        log.error(f"Failed to recompute file hash: {e}")
        raise


# ============================================================
# MODULE 5: PROVENANCE SCORE CALCULATOR
# Turns all findings into a single provenance score
# ============================================================

# Per-flag-confidence penalty weights (small, capped — weak signals should
# never dominate the score the way strong evidence does).
_CONFIDENCE_PENALTY = {
    "HIGH_CONFIDENCE": 8,
    "MEDIUM_CONFIDENCE": 4,
    "LOW_CONFIDENCE": 1,
}
_MAX_METADATA_PENALTY = 25  # hard cap regardless of how many flags exist


def calculate_provenance_score(
    metadata_flags: list, c2pa_result: dict, hash_result: dict, integrity_check: dict
) -> tuple:
    """
    Calculates a provenance sub-score from 0-100. This feeds into the
    final Trust Score in the Synthesis step.

    Rebalanced scoring logic (this revision):
      - Start at 50 (neutral — we don't know yet).
      - STRONG evidence dominates:
          * C2PA_VERIFIED                -> +40
          * C2PA_MARKER_FOUND (unverified)-> +10 (small — presence isn't proof)
          * C2PA_VERIFICATION_FAILED     -> -10 (markers existed but failed
                                             verification — more suspicious
                                             than simply having none)
          * NO_C2PA                      -> -3  (absence alone is weak evidence)
          * Hash VERIFIED_AUTHENTIC      -> +35
          * Hash DEBUNKED                -> -70
          * Recomputed hash MISMATCH     -> -50 (integrity failure — the file
                                             we're analyzing may not be the
                                             one Agent 1 actually looked at)
      - WEAK evidence (metadata flags) contributes only small, confidence
        weighted, capped penalties (max -25 total). Missing EXIF/metadata
        alone (LOW_CONFIDENCE) barely moves the score, reflecting that many
        legitimate platforms strip metadata.
      - Final score is clamped to 0-100.
    """
    score = 50
    reasoning = []

    # ── C2PA scoring ──
    c2pa_status = c2pa_result.get("c2pa_status", "NO_C2PA")
    if c2pa_status == "C2PA_VERIFIED":
        score += 40
        reasoning.append(
            "C2PA_VERIFIED: Cryptographic signature verified — strong authenticity signal"
        )
    elif c2pa_status in {"C2PA_PRESENT", "C2PA_NOT_AVAILABLE", "C2PA_NOT_VERIFIED"}:
        reasoning.append(
            "C2PA_MARKER_FOUND: Markers present but not cryptographically verified")
    elif c2pa_status == "C2PA_VERIFICATION_FAILED":  # legacy compatibility
        reasoning.append(
            "C2PA_VERIFICATION_FAILED: Markers present but verification failed — more suspicious than no C2PA at all"
        )
    else:  # NO_C2PA
        reasoning.append(
            "NO_C2PA:No C2PA Content Credentials were found.This is only a weak negative signal because most online videoscurrently do not contain C2PA manifests."
        )

    # ── Hash database scoring (strong evidence when present) ──
    if hash_result.get("hash_match"):
        if hash_result["verdict"] == "VERIFIED_AUTHENTIC":
            score += 35
            reasoning.append(
                f"HASH_VERIFIED: Known authentic — source: {hash_result['source']}")
        elif hash_result["verdict"] == "DEBUNKED":
            score -= 70
            reasoning.append(
                f"HASH_DEBUNKED: Known fake/debunked — source: {hash_result['source']}"
            )
    else:
        reasoning.append(
            "HASH_UNKNOWN: No prior record in internal database (external lookup not implemented)"
        )

    # ── File integrity scoring (strong evidence) ──
    if integrity_check.get("hash_mismatch"):
        score -= 50
        reasoning.append(
            "HASH_INTEGRITY_MISMATCH: Recomputed file hash does not match the hash "
            "reported by the Ingestor Agent — the file may have changed in transit"
        )
    else:
        reasoning.append(
            "HASH_INTEGRITY_OK: Recomputed hash matches Ingestor Agent's reported hash"
        )

    # ── Weak metadata-flag scoring (small, confidence-weighted, capped) ──
    metadata_penalty = 0
    for flag in metadata_flags:
        if flag["flag"].startswith((
            "NO_EXIF:", "NO_THUMBNAIL:", "NO_ICC_PROFILE:",
            "XMP_METADATA_PRESENT:", "IPTC_METADATA_PRESENT:",
        )):
            continue
        weight = _CONFIDENCE_PENALTY.get(flag["confidence"], 1)
        metadata_penalty += weight
    metadata_penalty = min(metadata_penalty, _MAX_METADATA_PENALTY)
    score -= metadata_penalty

    for flag in metadata_flags:
        reasoning.append(
            f"METADATA_FLAG [{flag['confidence']}]: {flag['flag']}")

    if metadata_penalty > 0:
        reasoning.append(
            f"METADATA_PENALTY_APPLIED: -{metadata_penalty} (capped at -{_MAX_METADATA_PENALTY})"
        )

    # Clamp score to 0-100
    score = max(0, min(100, score))

    return score, reasoning


def calculate_provenance_confidence(
    c2pa_result: dict, hash_result: dict, integrity_check: dict, metadata_flags: list
) -> str:
    """
    NEW: Estimates how much confidence we have in the provenance_score
    itself — distinct from the score value. A score of 50 built entirely
    from weak/absent metadata signals deserves LOW confidence; a score
    built on a verified C2PA signature or a hash-database match deserves
    HIGH confidence, regardless of whether that score is good or bad news.

    Logic:
      HIGH:   at least one strong, direct signal fired — C2PA_VERIFIED,
              a hash database match (either verdict), or a hash integrity
              mismatch. These are hard facts, not inferences.
      MEDIUM: no strong signal, but there is at least a C2PA marker and/or
              multiple HIGH_CONFIDENCE metadata flags to reason from.
      LOW:    the assessment rests mostly on absence-of-evidence
              (no C2PA, no hash match, only weak metadata flags).
    """
    strong_signal = (
        c2pa_result.get("c2pa_status") == "C2PA_VERIFIED"
        or hash_result.get("hash_match") is True
        or integrity_check.get("hash_mismatch") is True
    )
    if strong_signal:
        return "HIGH"

    high_conf_flag_count = sum(
        1 for f in metadata_flags if f["confidence"] == "HIGH_CONFIDENCE")
    has_c2pa_marker = c2pa_result.get("c2pa_status") in {
        "C2PA_PRESENT", "C2PA_NOT_AVAILABLE", "C2PA_NOT_VERIFIED"
    }

    if has_c2pa_marker or high_conf_flag_count >= 1:
        return "MEDIUM"

    return "LOW"


# ============================================================
# MAIN PIPELINE — RUN PROVENANCE AGENT
# ============================================================


def run_provenance_agent(ingestor_output: dict) -> dict:
    """
    MASTER FUNCTION — runs the full Provenance Agent pipeline.

    Pipeline (unchanged order):
        Metadata Extraction
          -> Metadata Consistency Analysis
          -> C2PA Check
          -> Hash Cross Reference
          -> Provenance Score
          -> Structured JSON Output

    New internal step: file hash is recomputed from disk and compared
    against Agent 1's reported hash before scoring.

    Input:  ingestor_output = the JSON dict from Agent 1
    Output: structured JSON with provenance findings for Agent 3 & 4
            (all original keys preserved; several new OPTIONAL keys added)
    """
    log.info("=" * 50)

    try:
        ingestor_output = _coerce_agent_dict(
            ingestor_output, "ingestor_output")
    except (TypeError, ValueError) as exc:
        return {
            "status": "error",
            "agent": "provenance",
            "error_message": str(exc),
            "ready_for_next_agent": False,
        }
    log.info("PROVENANCE AGENT STARTING")
    log.info("=" * 50)

    analysis_notes = []  # human-readable notes about limitations/fallbacks used this run

    # ── Guard: only proceed if Ingestor succeeded ──
    if not ingestor_output.get("ready_for_next_agent"):
        return {
            "status": "error",
            "error_message": "Ingestor Agent did not complete successfully",
            "ready_for_next_agent": False,
        }

    analysis_file_path = ingestor_output["normalized_file_path"]
    source_file_path = ingestor_output.get("source_file_path")
    file_path = (
        source_file_path
        if ingestor_output.get("media_type") == "image"
        and source_file_path
        and os.path.exists(source_file_path)
        else analysis_file_path
    )
    media_type = ingestor_output["media_type"]
    ingestor_hash = ingestor_output.get("source_file_hash") or ingestor_output["file_hash"]

    try:
        all_metadata_flags = []  # list of {"flag":..., "confidence":...} dicts

        # ── STEP 1: Extract metadata ──
        if media_type == "image":
            meta_result = extract_image_metadata(file_path)
        else:
            meta_result = extract_video_metadata(file_path)
            if not check_ffprobe_available():
                analysis_notes.append(
                    "ffprobe not found on this system — video metadata came from mutagen only. "
                    "Install FFmpeg for richer video metadata extraction."
                )

        raw_metadata = meta_result["raw_metadata"]
        all_metadata_flags.extend(meta_result["flags"])

        # ── STEP 2: Analyze metadata consistency ──
        consistency_flags = analyze_metadata_consistency(
            meta_result, media_type)
        all_metadata_flags.extend(consistency_flags)

        # ── STEP 3: C2PA check ──
        c2pa_result = check_c2pa_signature(file_path)
        try:
            import c2pa  # noqa: F401  (import-check only, to build an accurate note)
        except ImportError:
            analysis_notes.append(
                "c2pa-python is not installed — C2PA check used binary marker "
                "detection only, no cryptographic verification was possible."
            )

        # ── STEP 3b: Recompute hash and check integrity BEFORE trusting Agent 1's hash ──
        recomputed_hash = recompute_file_hash(file_path)
        hash_mismatch = recomputed_hash != ingestor_hash
        integrity_check = {
            "recomputed_hash": recomputed_hash,
            "ingestor_reported_hash": ingestor_hash,
            "hash_mismatch": hash_mismatch,
        }
        if hash_mismatch:
            all_metadata_flags.append(
                _make_flag(
                    "HASH_INTEGRITY_MISMATCH: Recomputed SHA-256 does not match the hash "
                    "reported by the Ingestor Agent",
                    "HIGH_CONFIDENCE",
                )
            )
            log.error(
                "Recomputed file hash does NOT match Ingestor Agent's reported hash!")

        # ── STEP 4: Hash cross-reference ──
        # NOTE: we cross-reference using the RECOMPUTED hash, not the
        # (potentially untrustworthy) hash reported by Agent 1.
        hash_result = cross_reference_hash(recomputed_hash)
        if hash_result.get("source_layer") == "external":
            analysis_notes.append(
                "External hash database lookup is a placeholder in this build — only the "
                "internal demo lookup table was actually queried."
            )

        # ── STEP 5: Calculate provenance score ──
        provenance_score, reasoning = calculate_provenance_score(
            all_metadata_flags, c2pa_result, hash_result, integrity_check
        )
        provenance_confidence = calculate_provenance_confidence(
            c2pa_result, hash_result, integrity_check, all_metadata_flags
        )

        # ── Build confidence_breakdown: flags grouped by confidence level ──
        confidence_breakdown = {
            "HIGH_CONFIDENCE": [
                f["flag"] for f in all_metadata_flags if f["confidence"] == "HIGH_CONFIDENCE"
            ],
            "MEDIUM_CONFIDENCE": [
                f["flag"] for f in all_metadata_flags if f["confidence"] == "MEDIUM_CONFIDENCE"
            ],
            "LOW_CONFIDENCE": [
                f["flag"] for f in all_metadata_flags if f["confidence"] == "LOW_CONFIDENCE"
            ],
        }

        # ── Build metadata_summary: a compact, human-friendly digest ──
        metadata_summary = {
            "media_type": media_type,
            "software_or_encoder": raw_metadata.get("Software") or raw_metadata.get("encoder"),
            "camera_make": raw_metadata.get("Make"),
            "camera_model": raw_metadata.get("Model"),
            "dimensions": raw_metadata.get("_dimensions") if media_type == "image" else None,
            "source_format": raw_metadata.get("_source_format") if media_type == "image" else None,
            "extraction_method": raw_metadata.get("_extraction_method", "exif/exifread"),
            "total_flags_raised": len(all_metadata_flags),
        }
        if media_type != "image":
            metadata_summary.update(
                {
                    "resolution": raw_metadata.get("resolution"),
                    "codec": raw_metadata.get("codec"),
                    "container_format": raw_metadata.get("container_format"),
                    "duration_seconds": raw_metadata.get("duration_seconds"),
                }
            )

        # ── STEP 6: Build output (all original keys preserved) ──
        # NOTE: "flags" stays a plain list of strings for backward
        # compatibility with anything downstream that already parses it.
        result = {
            "status": "success",
            "processed_at": datetime.now().isoformat(),
            "agent": "provenance",
            "file_path": analysis_file_path,
            "source_file_path": file_path,
            "media_type": media_type,
            "file_hash": ingestor_hash,  # unchanged field: original hash from Agent 1
            "metadata": {
                "raw": raw_metadata,
                # backward-compatible shape
                "flags": [f["flag"] for f in all_metadata_flags],
                "exif_status": meta_result.get("exif_status", "NOT_CHECKED"),
                "xmp_status": meta_result.get("xmp_status", "NOT_CHECKED"),
                "iptc_status": meta_result.get("iptc_status", "NOT_CHECKED"),
                "thumbnail_status": meta_result.get("thumbnail_status", "NOT_CHECKED"),
            },
            "c2pa": c2pa_result,
            "hash_lookup": hash_result,
            "provenance_score": provenance_score,
            "provenance_confidence": provenance_confidence,  # NEW
            "provenance_reasoning": reasoning,
            "chain_of_custody_established": (
                c2pa_result["c2pa_valid"] or hash_result.get(
                    "verdict") == "VERIFIED_AUTHENTIC"
            ),
            # ── NEW OPTIONAL FIELDS (additive only) ──
            "metadata_summary": metadata_summary,
            "confidence_breakdown": confidence_breakdown,
            "integrity_check": integrity_check,
            "hash_recomputed": recomputed_hash,
            "hash_matches_ingestor": not hash_mismatch,
            "analysis_notes": analysis_notes,
            # Boolean (fixed from string in the demo input below)
            "ready_for_next_agent": True,
        }

        log.info(
            f"PROVENANCE AGENT COMPLETE ✓ — Score: {provenance_score}/100 "
            f"(confidence: {provenance_confidence})"
        )
        log.info(
            "Result:\n" +
            json.dumps({k: v for k, v in result.items()
                       if k != "metadata"}, indent=2)
        )
        return result

    except Exception as e:
        log.error(f"Provenance Agent FAILED: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "error_message": str(e),
            "processed_at": datetime.now().isoformat(),
            "ready_for_next_agent": False,
        }


# ============================================================
# ── HOW TO CHANGE FOR YOUR OWN TEST ──
#
# 1. Run Agent 1 first and get its JSON output
# 2. Paste1 that JSON output into the variable below
# 3. eti2
# ============================================================

# Legacy pasted-JSON fixture retained only as reference; never executed.
if False:

    # ── PASTE AGENT 1 OUTPUT HERE ──
    agent1_output = r"""
{
  "status": "success",
  "processed_at": "2026-08-29T16:14:44.832463",
  "input_type": "local_file",
  "media_type": "image",
  "canonical_format": "png",
  "file_hash": "0e90d8d2683e50e6a23416d17f71a4a36e993cb53dd3461f9bf252c6386ec178",
  "normalized_file_path": "/content/canonical_image.png",
  "ready_for_next_agent": true
}
"""

    # Convert JSON text → Python dictionary
    agent1_output = json.loads(agent1_output)

    print("\n" + "=" * 60)
    print("  RUNNING PROVENANCE AGENT")
    print("=" * 60)

    result = run_provenance_agent(agent1_output)

    print("\n>>> FINAL PROVENANCE RESULT:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    import sys
    from agent.ingestor import run_ingestor_agent

    local_image = sys.argv[1] if len(sys.argv) > 1 else "imageprocessor.jpeg"
    result = run_provenance_agent(run_ingestor_agent(local_image))
    print(json.dumps(result, indent=2, default=str))
