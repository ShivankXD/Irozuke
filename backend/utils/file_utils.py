"""
File utilities for Irozuke AI backend.
"""

from pathlib import Path
from fastapi import UploadFile, HTTPException

ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
ALLOWED_PDF_TYPES   = {"application/pdf"}
ALLOWED_TYPES       = ALLOWED_IMAGE_TYPES | ALLOWED_PDF_TYPES

MAX_IMAGE_BYTES = 30  * 1024 * 1024   # 30 MB  for single images
MAX_PDF_BYTES   = 300 * 1024 * 1024   # 300 MB for manga PDF chapters


def validate_file(file: UploadFile) -> str:
    """
    Validate uploaded file. Returns 'image' or 'pdf'.
    Raises HTTP 400 for unsupported types.
    """
    ct = (file.content_type or "").lower()
    if ct in ALLOWED_IMAGE_TYPES:
        return "image"
    if ct in ALLOWED_PDF_TYPES:
        return "pdf"
    raise HTTPException(
        status_code=400,
        detail=(
            f"Unsupported file type: '{ct}'. "
            f"Accepted: PNG, JPEG, WebP images or PDF manga chapters."
        ),
    )


async def save_upload(file: UploadFile, directory: Path, job_id: str, file_kind: str) -> Path:
    """
    Save uploaded file in chunks. Returns saved Path.
    """
    directory.mkdir(parents=True, exist_ok=True)
    max_bytes = MAX_PDF_BYTES if file_kind == "pdf" else MAX_IMAGE_BYTES

    suffix    = Path(file.filename or "upload").suffix or (".pdf" if file_kind == "pdf" else ".png")
    dest_path = directory / f"{job_id}_input{suffix}"

    size = 0
    with dest_path.open("wb") as out:
        while chunk := await file.read(65536):
            size += len(chunk)
            if size > max_bytes:
                dest_path.unlink(missing_ok=True)
                limit_mb = max_bytes // (1024 * 1024)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Max size for {file_kind} is {limit_mb} MB."
                )
            out.write(chunk)

    return dest_path
