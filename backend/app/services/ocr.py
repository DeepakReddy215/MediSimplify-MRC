import io
import shutil
from os import environ
from os import name as os_name
from pathlib import Path

import fitz
import pytesseract
from PIL import Image, UnidentifiedImageError
from app.core.config import get_settings

settings = get_settings()


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
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def _ensure_tesseract_cmd() -> str | None:
    global TESSERACT_CMD
    if TESSERACT_CMD:
        return TESSERACT_CMD

    TESSERACT_CMD = _resolve_tesseract_cmd()
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    return TESSERACT_CMD


def _require_tesseract() -> None:
    if not _ensure_tesseract_cmd():
        raise OCRUnavailableError(
            "OCR engine is not available. Install Tesseract or upload a text-based PDF."
        )


def extract_text_from_file(file_name: str, file_bytes: bytes) -> str:
    lower_name = file_name.lower()

    if lower_name.endswith(".pdf"):
        return _extract_text_from_pdf(file_bytes)

    return _extract_text_from_image(file_bytes)


def _extract_text_from_image(file_bytes: bytes) -> str:
    _require_tesseract()
    try:
        with Image.open(io.BytesIO(file_bytes)) as raw_image:
            image = raw_image.convert("RGB")
        text = pytesseract.image_to_string(image)
    except UnidentifiedImageError as exc:
        raise OCRUnavailableError("Unsupported or corrupted image file.") from exc
    except pytesseract.TesseractNotFoundError as exc:
        raise OCRUnavailableError(
            "OCR engine is not available. Install Tesseract and retry."
        ) from exc
    return " ".join(text.split())


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

        _require_tesseract()

        pages: list[str] = []

        try:
            for page in doc:
                pix = page.get_pixmap(dpi=220)
                with Image.open(io.BytesIO(pix.tobytes("png"))) as raw_image:
                    image = raw_image.convert("RGB")
                page_text = pytesseract.image_to_string(image)
                pages.append(page_text)
        except pytesseract.TesseractNotFoundError as exc:
            raise OCRUnavailableError(
                "OCR engine is not available. Install Tesseract and retry."
            ) from exc

        return "\n".join(" ".join(page.split()) for page in pages if page.strip())
