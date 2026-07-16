from __future__ import annotations

import logging
import uuid
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse
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


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.post("/api/convert")
async def convert(files: list[UploadFile] = File(...)) -> FileResponse:
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
        logger.info("Job %s saved upload %s as %s", job_id, upload.filename, image_path)

    try:
        pptx_path = await run_in_threadpool(convert_images_to_pptx, image_paths=image_paths, output_dir=job_dir)
    except Exception as error:
        logger.exception("Job %s failed", job_id)
        raise HTTPException(status_code=500, detail=f"生成失败，请查看日志。任务 ID：{job_id}") from error

    logger.info("Job %s completed: %s", job_id, pptx_path)
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
