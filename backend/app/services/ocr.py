import io
import subprocess
import shutil
from os import environ
from os import name as os_name
from pathlib import Path

import fitz
import pytesseract
import requests
from PIL import Image, UnidentifiedImageError
from app.core.config import get_settings

settings = get_settings()
DEFAULT_TESSERACT_CMD = "/usr/bin/tesseract"
VERIFY_TESSERACT_ON_STARTUP = environ.get("VERIFY_TESSERACT_ON_STARTUP", "1").strip() == "1"
OCR_SPACE_API_KEY = environ.get("OCR_SPACE_API_KEY", "").strip()
OCR_SPACE_ENDPOINT = environ.get("OCR_SPACE_ENDPOINT", "https://api.ocr.space/parse/image").strip() or "https://api.ocr.space/parse/image"

try:
    OCR_SPACE_TIMEOUT_SECONDS = int(environ.get("OCR_SPACE_TIMEOUT_SECONDS", "45").strip())
except ValueError:
    OCR_SPACE_TIMEOUT_SECONDS = 45

_TESSERACT_PROBED = False
_TESSERACT_READY = False


class OCRUnavailableError(RuntimeError):
    pass


def _common_tesseract_candidates() -> list[str]:
    candidates = [
        "tesseract",
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/opt/render/project/.apt/usr/bin/tesseract",
    ]

    # Common install paths for Windows users.
    if os_name == "nt":
        local_appdata = environ.get("LOCALAPPDATA", "").strip()
        candidates.extend(
            [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                str(Path(local_appdata) / "Programs" / "Tesseract-OCR" / "tesseract.exe") if local_appdata else "",
            ]
        )

    return [candidate for candidate in candidates if candidate]


def _resolve_tesseract_cmd() -> str | None:
    configured = (settings.tesseract_cmd or "").strip()
    if configured:
        if Path(configured).exists():
            return configured
        resolved = shutil.which(configured)
        if resolved:
            return resolved

    for candidate in _common_tesseract_candidates():
        if Path(candidate).exists():
            return candidate

        resolved = shutil.which(candidate)
        if resolved:
            return resolved

    return None


TESSERACT_CMD = _resolve_tesseract_cmd()
if not TESSERACT_CMD and os_name != "nt":
    TESSERACT_CMD = DEFAULT_TESSERACT_CMD

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def _probe_tesseract_version() -> bool:
    global _TESSERACT_PROBED, _TESSERACT_READY

    if _TESSERACT_PROBED:
        return _TESSERACT_READY

    cmd = TESSERACT_CMD or "tesseract"
    try:
        result = subprocess.run([cmd, "--version"], capture_output=True, text=True)
        _TESSERACT_READY = result.returncode == 0
        output = (result.stdout or result.stderr or "").strip()
        if output:
            print(output.splitlines()[0])
    except Exception as exc:
        _TESSERACT_READY = False
        print(f"Tesseract version probe failed: {exc}")

    _TESSERACT_PROBED = True
    return _TESSERACT_READY


if VERIFY_TESSERACT_ON_STARTUP:
    _probe_tesseract_version()


def _ensure_tesseract_cmd() -> str | None:
    global TESSERACT_CMD, _TESSERACT_PROBED, _TESSERACT_READY

    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        if _probe_tesseract_version():
            return TESSERACT_CMD

    TESSERACT_CMD = _resolve_tesseract_cmd()
    if not TESSERACT_CMD and os_name != "nt":
        TESSERACT_CMD = DEFAULT_TESSERACT_CMD

    _TESSERACT_PROBED = False
    _TESSERACT_READY = False

    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        if _probe_tesseract_version():
            return TESSERACT_CMD

    # Last-resort check in case PATH changed at runtime.
    path_cmd = shutil.which("tesseract")
    if path_cmd:
        TESSERACT_CMD = path_cmd
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        _TESSERACT_PROBED = False
        _TESSERACT_READY = False
        if _probe_tesseract_version():
            return TESSERACT_CMD

    TESSERACT_CMD = None
    _TESSERACT_PROBED = True
    _TESSERACT_READY = False
    return None


def _require_tesseract() -> None:
    if not _ensure_tesseract_cmd():
        raise OCRUnavailableError(
            "OCR engine is not available. Install Tesseract or upload a text-based PDF."
        )


def _extract_text_with_ocr_space(image_bytes: bytes, file_name: str) -> str:
    if not OCR_SPACE_API_KEY:
        raise OCRUnavailableError(
            "OCR engine is unavailable on this server. Set OCR_SPACE_API_KEY for cloud OCR fallback."
        )

    payload = {
        "language": "eng",
        "isOverlayRequired": "false",
        "scale": "true",
        "OCREngine": "2",
    }
    headers = {"apikey": OCR_SPACE_API_KEY}
    files = {"file": (file_name, image_bytes, "image/png")}

    try:
        response = requests.post(
            OCR_SPACE_ENDPOINT,
            data=payload,
            files=files,
            headers=headers,
            timeout=OCR_SPACE_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise OCRUnavailableError(f"Cloud OCR request failed: {exc}") from exc
    except ValueError as exc:
        raise OCRUnavailableError("Cloud OCR returned invalid JSON response.") from exc

    if data.get("IsErroredOnProcessing"):
        errors = data.get("ErrorMessage")
        if isinstance(errors, list):
            details = " | ".join(str(error) for error in errors if error)
        elif isinstance(errors, str):
            details = errors
        else:
            details = "unknown OCR processing error"
        raise OCRUnavailableError(f"Cloud OCR failed: {details}")

    parsed_results = data.get("ParsedResults") or []
    if not isinstance(parsed_results, list):
        raise OCRUnavailableError("Cloud OCR response format is invalid.")

    text_chunks: list[str] = []
    for result in parsed_results:
        if isinstance(result, dict):
            parsed_text = result.get("ParsedText", "")
            if isinstance(parsed_text, str) and parsed_text.strip():
                text_chunks.append(" ".join(parsed_text.split()))

    if not text_chunks:
        raise OCRUnavailableError("Cloud OCR returned no text.")

    return "\n".join(text_chunks)


def _extract_text_from_png_bytes(png_bytes: bytes, file_name: str) -> str:
    if _ensure_tesseract_cmd():
        try:
            with Image.open(io.BytesIO(png_bytes)) as raw_image:
                image = raw_image.convert("RGB")
            text = pytesseract.image_to_string(image)
            if text.strip():
                return " ".join(text.split())
        except pytesseract.TesseractNotFoundError:
            pass

    return _extract_text_with_ocr_space(png_bytes, file_name)


def extract_text_from_file(file_name: str, file_bytes: bytes) -> str:
    lower_name = file_name.lower()

    if lower_name.endswith(".pdf"):
        return _extract_text_from_pdf(file_bytes)

    return _extract_text_from_image(file_bytes)


def _extract_text_from_image(file_bytes: bytes) -> str:
    try:
        with Image.open(io.BytesIO(file_bytes)) as raw_image:
            image = raw_image.convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            normalized_png = buffer.getvalue()
    except UnidentifiedImageError as exc:
        raise OCRUnavailableError("Unsupported or corrupted image file.") from exc

    return _extract_text_from_png_bytes(normalized_png, "upload.png")


def _extract_text_from_pdf(file_bytes: bytes) -> str:
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        # Prefer embedded/selectable text when available.
        embedded_text_pages: list[str] = []
        for page in doc:
            page_text = page.get_text("text")
            if isinstance(page_text, str) and page_text.strip():
                embedded_text_pages.append(" ".join(page_text.split()))

        if embedded_text_pages:
            return "\n".join(embedded_text_pages)

        pages: list[str] = []

        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(dpi=220)
            page_text = _extract_text_from_png_bytes(pix.tobytes("png"), f"page-{page_index}.png")
            if page_text.strip():
                pages.append(page_text)

        return "\n".join(pages)
