"""
/api/colorize  — Upload a manga image OR PDF chapter, get colorized result back.
"""

import uuid
import time
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse

from utils.file_utils import validate_file, save_upload
from utils.colorizer import run_colorizer

router = APIRouter()

UPLOADS_DIR = Path("uploads")
OUTPUTS_DIR = Path("outputs")


@router.post("/colorize")
async def colorize_image(
    file: UploadFile = File(...),
    model: str = Form(default="auto"),
):
    """
    Upload a manga image (PNG/JPEG/WebP) or PDF chapter and get a colorized result.
    """

    # 1. Validate — returns 'image' or 'pdf'
    file_kind = validate_file(file)

    # 2. Save upload
    job_id      = str(uuid.uuid4())[:8]
    upload_path = await save_upload(file, UPLOADS_DIR, job_id, file_kind)

    # 3. Colorize
    start = time.time()
    try:
        output_path = run_colorizer(upload_path, OUTPUTS_DIR, job_id, model=model, file_kind=file_kind)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Colorization failed: {e}")
    elapsed = round(time.time() - start, 2)

    output_url = f"/outputs/{output_path.name}"

    return JSONResponse({
        "job_id":      job_id,
        "model_used":  model,
        "file_kind":   file_kind,
        "output_url":  output_url,
        "elapsed_sec": elapsed,
        "status":      "success",
    })
