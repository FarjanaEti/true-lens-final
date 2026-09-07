#!apt update && apt install -y ffmpeg
#!pip install yt-dlp Pillow requests ffmpeg-python
# agent1
# THE TRUE LENS — INGESTOR AGENT (Full Version, fixed)

import os
import json
import logging
import hashlib
import mimetypes
import tempfile
import subprocess
import requests
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse

import yt_dlp
from PIL import Image, ImageOps, UnidentifiedImageError

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [INGESTOR] %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ============================================================
# MODULE 1: INPUT CLASSIFIER
# ============================================================


def classify_input(user_input: str) -> str:
    log.info(f"Classifying input: {user_input}")

    if os.path.exists(user_input):
        return "local_file"

    try:
        parsed = urlparse(user_input)
        is_url = parsed.scheme in ("http", "https")
    except Exception:
        return "unsupported"

    if not is_url:
        return "unsupported"

    url_lower = user_input.lower()

    if any(x in url_lower for x in ["youtube.com", "youtu.be"]):
        return "youtube_url"

    if any(
        x in url_lower
        for x in [
            "facebook.com",
            "fb.watch",
            "instagram.com",
            "tiktok.com",
            "twitter.com",
            "x.com",
            "whatsapp.com",
        ]
    ):
        return "social_media"

    if url_lower.endswith(".m3u8"):
        return "hls_stream"

    media_extensions = [
        ".mp4",
        ".mov",
        ".avi",
        ".webm",
        ".mkv",
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".gif",
        ".mp3",
        ".wav",
    ]
    if any(url_lower.endswith(ext) for ext in media_extensions):
        return "direct_media"

    # Default — try with yt-dlp anyway (supports 1000+ sites)
    return "social_media"


# ============================================================
# MODULE 2: SECURITY SANITIZATION
# ============================================================

DANGEROUS_EXTENSIONS = [".exe", ".bat", ".sh", ".php", ".js", ".py", ".dll", ".cmd", ".vbs", ".ps1"]

MAX_FILE_SIZE_MB = 500
MAX_IMAGE_PIXELS = 100_000_000
SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "GIF"}


def sanitize_input(user_input: str, input_type: str) -> bool:
    log.info("Running security checks...")

    if input_type == "local_file":
        path = Path(user_input)

        if path.suffix.lower() in DANGEROUS_EXTENSIONS:
            raise ValueError(f"BLOCKED: Dangerous file type '{path.suffix}'")

        size_mb = os.path.getsize(user_input) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise ValueError(
                f"BLOCKED: File too large ({size_mb:.1f}MB). Max is {MAX_FILE_SIZE_MB}MB"
            )

        log.info(f"Local file passed security checks ({size_mb:.1f}MB)")

    else:
        parsed = urlparse(user_input)

        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"BLOCKED: Unsafe URL scheme '{parsed.scheme}'")

        blocked_hosts = ["localhost", "127.0.0.1", "0.0.0.0", "169.254."]
        if any(b in parsed.netloc for b in blocked_hosts):
            raise ValueError("BLOCKED: Local network URLs are not allowed")

        log.info("URL passed security checks")

    return True


# ============================================================
# MODULE 3: MEDIA DOWNLOADER / EXTRACTOR
# ============================================================


def _safe_truncate_filename(path: str, max_len: int = 200) -> str:
    """
    Extra safety belt: if a resulting filename (not counting directory)
    is still too long for any reason, truncate it while preserving the
    extension. Applied defensively after download in case a site gives
    yt-dlp an ID/format string that is unusually long.
    """
    directory, filename = os.path.split(path)
    stem, ext = os.path.splitext(filename)
    if len(filename.encode("utf-8")) <= max_len:
        return path

    # Truncate the stem (not the extension) down to a safe byte length
    safe_stem = stem.encode("utf-8")[: max_len - len(ext.encode("utf-8"))].decode("utf-8", "ignore")
    new_filename = f"{safe_stem}{ext}"
    new_path = os.path.join(directory, new_filename)

    if new_path != path and os.path.exists(path):
        os.rename(path, new_path)
        log.info(f"Filename truncated for safety: {filename} -> {new_filename}")

    return new_path


def download_media(
    user_input: str, input_type: str, temp_dir: str, cookies_file: str = None
) -> str:
    """
    cookies_file: optional path to a cookies.txt file for private/login-gated
    Facebook/Instagram content. Leave as None for public content.
    """

    if input_type == "local_file":
        log.info("Local file — skipping download step")
        return user_input

    if input_type in ("youtube_url", "social_media", "hls_stream"):
        log.info(f"Downloading via yt-dlp from: {user_input}")

        # FIX: use the video/post ID instead of the (often huge) title
        # as the filename. Facebook/Instagram/TikTok "titles" can be
        # entire captions hundreds of characters long, which crashes
        # the filesystem with "File name too long".
        output_template = os.path.join(temp_dir, "%(id)s.%(ext)s")
        options = {
            "outtmpl": output_template,
            "format": "mp4[height<=480]/best[height<=480]/best",
            "quiet": True,
            "no_warnings": True,
            # Extra safety net: yt-dlp will itself sanitize/restrict
            # characters and length in filenames it generates.
            "restrictfilenames": True,
            "windowsfilenames": True,
            "trim_file_name": 150,
        }

        # ── For private/login-gated FB or Instagram posts ──
        if cookies_file and os.path.exists(cookies_file):
            options["cookiefile"] = cookies_file
            log.info("Using cookies file for authenticated access")

        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(user_input, download=True)
            downloaded_path = ydl.prepare_filename(info)

        # Belt-and-braces: truncate if somehow still too long
        downloaded_path = _safe_truncate_filename(downloaded_path)

        log.info(f"Downloaded to: {downloaded_path}")
        return downloaded_path

    if input_type == "direct_media":
        log.info(f"Downloading direct media from: {user_input}")

        response = requests.get(user_input, stream=True, timeout=30)
        response.raise_for_status()

        ext = Path(urlparse(user_input).path).suffix or ".tmp"
        output_path = os.path.join(temp_dir, f"downloaded_media{ext}")

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        log.info(f"Downloaded to: {output_path}")
        return output_path

    raise ValueError(f"Cannot download input type: {input_type}")


# ============================================================
# MODULE 4: MEDIA TYPE DETECTOR
# ============================================================


def detect_media_type(file_path: str) -> str:
    log.info(f"Detecting media type of: {file_path}")

    mime_type, _ = mimetypes.guess_type(file_path)

    if mime_type:
        if mime_type.startswith("video"):
            return "video"
        elif mime_type.startswith("image"):
            return "image"
        elif mime_type.startswith("audio"):
            return "audio"

    ext = Path(file_path).suffix.lower()
    video_exts = [".mp4", ".mov", ".avi", ".webm", ".mkv", ".flv"]
    image_exts = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"]
    audio_exts = [".mp3", ".wav", ".aac", ".ogg", ".flac"]

    if ext in video_exts:
        return "video"
    elif ext in image_exts:
        return "image"
    elif ext in audio_exts:
        return "audio"

    return "unsupported"


def validate_local_image(file_path: str) -> dict:
    """Validate decoded image content rather than trusting its extension."""
    try:
        with Image.open(file_path) as image:
            detected_format = (image.format or "").upper()
            width, height = image.size
            frame_count = int(getattr(image, "n_frames", 1))
            if detected_format not in SUPPORTED_IMAGE_FORMATS:
                raise ValueError(f"Unsupported image format: {detected_format or 'unknown'}")
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise ValueError(
                    f"Unsafe image dimensions: {width}x{height}; maximum is {MAX_IMAGE_PIXELS:,} pixels"
                )
            image.verify()

        # verify() invalidates the decoder, so reopen and force a full decode.
        with Image.open(file_path) as image:
            image.seek(0)
            image.load()

        return {
            "detected_format": detected_format,
            "width": width,
            "height": height,
            "frame_count": frame_count,
        }
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Invalid or corrupt image file: {exc}") from exc


# ============================================================
# MODULE 5: CANONICALIZATION ENGINE
# ============================================================


def canonicalize_video(input_path: str, output_path: str) -> str:
    log.info("Canonicalizing video to MP4 (H.264 + AAC)...")

    result = subprocess.run(
        [
            "ffmpeg",
            "-i",
            input_path,
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            "-y",
            output_path,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")

    log.info(f"Video saved to: {output_path}")
    return output_path


def canonicalize_image(input_path: str, output_path: str) -> str:
    log.info("Canonicalizing image to PNG...")

    with Image.open(input_path) as source:
        source.seek(0)
        img = ImageOps.exif_transpose(source)
        has_alpha = img.mode in ("RGBA", "LA") or "transparency" in img.info
        img = img.convert("RGBA" if has_alpha else "RGB")
        img.save(output_path, "PNG")

    log.info(f"Image saved to: {output_path}")
    return output_path


def canonicalize_audio(input_path: str, output_path: str) -> str:
    log.info("Canonicalizing audio to WAV...")

    result = subprocess.run(
        ["ffmpeg", "-i", input_path, "-ar", "44100", "-ac", "2", "-y", output_path],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")

    log.info(f"Audio saved to: {output_path}")
    return output_path


def canonicalize(file_path: str, media_type: str, output_dir: str) -> tuple:
    if media_type == "video":
        out = os.path.join(output_dir, "canonical_video.mp4")
        return canonicalize_video(file_path, out), "mp4"

    elif media_type == "image":
        out = os.path.join(output_dir, "canonical_image.png")
        return canonicalize_image(file_path, out), "png"

    elif media_type == "audio":
        out = os.path.join(output_dir, "canonical_audio.wav")
        return canonicalize_audio(file_path, out), "wav"

    else:
        raise ValueError(f"Unsupported media type: {media_type}")


# ============================================================
# MODULE 6: FINGERPRINT GENERATOR
# ============================================================


def generate_fingerprint(file_path: str) -> str:
    log.info("Generating SHA-256 fingerprint...")

    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    file_hash = sha256.hexdigest()
    log.info(f"Fingerprint: {file_hash}")
    return file_hash


# ============================================================
# MAIN PIPELINE — RUN INGESTOR AGENT
# ============================================================


def run_ingestor_agent(user_input: str, cookies_file: str = None) -> dict:
    """
    user_input: file path OR any URL (YouTube, Facebook, Instagram,
                TikTok, direct image/video link, HLS stream)
    cookies_file: optional, only needed for private/login-gated posts
    """
    log.info("=" * 50)
    log.info("INGESTOR AGENT STARTING")
    log.info("=" * 50)

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            input_type = classify_input(user_input)
            log.info(f"Input classified as: {input_type}")

            if input_type == "unsupported":
                raise ValueError("Input type not supported")

            sanitize_input(user_input, input_type)

            downloaded_path = download_media(user_input, input_type, temp_dir, cookies_file)

            media_type = detect_media_type(downloaded_path)
            log.info(f"Media type detected: {media_type}")

            if media_type == "unsupported":
                raise ValueError("Unsupported media type — cannot process")

            source_path = os.path.abspath(downloaded_path)
            source_hash = generate_fingerprint(source_path)
            image_info = None
            if media_type == "image":
                image_info = validate_local_image(source_path)

            output_dir = os.getcwd()
            if media_type == "image":
                canonical_path = canonicalize_image(
                    source_path,
                    os.path.join(output_dir, f"canonical_image_{source_hash[:16]}.png"),
                )
                canonical_format = "png"
            else:
                canonical_path, canonical_format = canonicalize(
                    downloaded_path, media_type, output_dir
                )

            normalized_hash = generate_fingerprint(canonical_path)

            result = {
                "status": "success",
                "processed_at": datetime.now().isoformat(),
                "input_type": input_type,
                "media_type": media_type,
                "canonical_format": canonical_format,
                "file_hash": source_hash if media_type == "image" else normalized_hash,
                "normalized_file_path": canonical_path,
                "source_file_path": source_path if input_type == "local_file" else None,
                "source_file_hash": source_hash,
                "normalized_file_hash": normalized_hash,
                "image_info": image_info,
                "ready_for_next_agent": True,
            }

            log.info("INGESTOR AGENT COMPLETE ✓")
            log.info("Result: " + json.dumps(result, indent=2))
            return result

        except Exception as e:
            log.error(f"Ingestor Agent FAILED: {str(e)}")
            return {
                "status": "error",
                "error_message": str(e),
                "processed_at": datetime.now().isoformat(),
                "ready_for_next_agent": False,
            }


# ============================================================
# TEST IT — change these to whatever you want to check eti1
# ============================================================

if __name__ == "__main__":

    test_inputs = [
        # Local image
        "/content/imageprocessor.jpeg",
        # Local video
        # "/content/brakobama.mp4",
        # YouTube link
        # "https://www.youtube.com/watch?v=ZwpgYkbHX7Q",
        # Facebook post link (public only, unless you pass cookies_file)
        # "https://www.facebook.com/share/r/1E6gb4BvhK/",
        # Direct image URL
        # "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
    ]

    for test_input in test_inputs:
        print(f"\n>>> Testing: {test_input}")
        result = run_ingestor_agent(test_input)
        print(f">>> Result status: {result['status']}\n")
        print(json.dumps(result, indent=2))
