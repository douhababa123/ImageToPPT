from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from app.config import settings
from app.logging_config import setup_logging
from app.services.pipeline import convert_images_to_pptx


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
WORK_DIR = settings.output_dir
ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Image To Editable PPT")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# in-memory job progress store
_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def _set_progress(job_id: str, **kwargs):
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.post("/api/convert")
async def convert(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="请至少上传一张图片。")
    if len(files) > settings.max_upload_files:
        raise HTTPException(status_code=400, detail=f"一次最多上传 {settings.max_upload_files} 张图片。")

    job_id = uuid.uuid4().hex
    job_dir = WORK_DIR / job_id
    input_dir = job_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Job %s started with %s uploaded file(s)", job_id, len(files))

    image_paths: list[Path] = []
    for index, upload in enumerate(files, start=1):
        suffix = Path(upload.filename or "image.png").suffix.lower() or ".png"
        if suffix not in ALLOWED_IMAGE_SUFFIXES:
            raise HTTPException(status_code=400, detail=f"{upload.filename or '文件'} 的格式不受支持。")

        data = await upload.read(settings.max_upload_mb * 1024 * 1024 + 1)
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(status_code=400, detail=f"{upload.filename or '文件'} 超过 {settings.max_upload_mb} MB。")
        _verify_image(upload.filename or f"image{suffix}", data)

        image_path = input_dir / f"{index:03d}{suffix}"
        image_path.write_bytes(data)
        image_paths.append(image_path)

    with _lock:
        _jobs[job_id] = {
            "status": "processing",
            "stage": "OCR",
            "current": 0,
            "total": len(image_paths),
            "error": None,
        }

    threading.Thread(target=_run_conversion, args=(job_id, image_paths, job_dir), daemon=True).start()

    return JSONResponse({"job_id": job_id})


def _run_conversion(job_id: str, image_paths: list[Path], job_dir: Path):
    try:
        from app.services.inpaint import remove_text_from_image
        from app.services.ocr import detect_text
        from app.services.ppt import build_pptx

        processed_slides = []
        cleaned_dir = job_dir / "cleaned"
        cleaned_dir.mkdir(parents=True, exist_ok=True)

        for index, image_path in enumerate(image_paths, start=1):
            _set_progress(job_id, stage="OCR", current=index - 1, total=len(image_paths))
            text_boxes = detect_text(image_path)

            _set_progress(job_id, stage="inpaint", current=index - 1, total=len(image_paths))
            cleaned_image = cleaned_dir / f"slide-{index:03d}.png"
            remove_text_from_image(image_path, text_boxes, cleaned_image)
            processed_slides.append((image_path, cleaned_image, text_boxes))
            _set_progress(job_id, stage="OCR", current=index, total=len(image_paths))

        _set_progress(job_id, stage="PPTX", current=len(image_paths), total=len(image_paths))
        pptx_path = build_pptx(processed_slides, job_dir / "editable-images.pptx")

        _set_progress(job_id, status="done", stage="complete", current=len(image_paths), total=len(image_paths))
    except Exception as error:
        logger.exception("Job %s failed", job_id)
        _set_progress(job_id, status="error", error=str(error))


@app.get("/api/status/{job_id}")
def job_status(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在或已过期。")
    return JSONResponse(job)


@app.get("/api/download/{job_id}")
def download_result(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if not job or job.get("status") != "done":
        raise HTTPException(status_code=404, detail="任务未完成或不存在。")

    pptx_path = WORK_DIR / job_id / "editable-images.pptx"
    if not pptx_path.exists():
        raise HTTPException(status_code=404, detail="PPT 文件不存在，可能已被清理。")

    return FileResponse(
        pptx_path,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename="editable-images.pptx",
    )


def _verify_image(filename: str, data: bytes) -> None:
    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=400, detail=f"{filename} 不是有效图片文件。") from error