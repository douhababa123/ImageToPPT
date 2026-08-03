from __future__ import annotations

import json
import logging
import threading
import uuid
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from app.config import settings
from app.logging_config import setup_logging
from app.services.ocr import OcrTextBox, detect_text
from app.services.text_rules import should_keep_as_background


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


async def _save_uploads(files: list[UploadFile]) -> tuple[str, list[Path], Path]:
    if not files:
        raise HTTPException(status_code=400, detail="请至少上传一张图片。")
    if len(files) > settings.max_upload_files:
        raise HTTPException(status_code=400, detail=f"一次最多上传 {settings.max_upload_files} 张图片。")

    job_id = uuid.uuid4().hex
    job_dir = WORK_DIR / job_id
    input_dir = job_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

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

    return job_id, image_paths, job_dir


@app.post("/api/ocr")
async def ocr(files: list[UploadFile] = File(...)):
    """Upload images, run OCR, and return detected text boxes so the user can
    choose which ones to turn into editable text."""
    job_id, image_paths, job_dir = await _save_uploads(files)
    logger.info("OCR job %s with %s file(s)", job_id, len(image_paths))

    images_payload = []
    all_boxes: list[list[dict]] = []
    for index, image_path in enumerate(image_paths):
        with Image.open(image_path) as im:
            width, height = im.size
        boxes = detect_text(image_path)
        boxes_payload = []
        for box_index, box in enumerate(boxes):
            boxes_payload.append(
                {
                    "i": box_index,
                    "text": box.text,
                    "confidence": round(box.confidence, 3),
                    "box": [[float(x), float(y)] for x, y in box.box],
                    "keep": should_keep_as_background(box, width, height),
                }
            )
        all_boxes.append(boxes_payload)
        images_payload.append({"index": index, "width": width, "height": height, "boxes": boxes_payload})

    (job_dir / "ocr.json").write_text(json.dumps({"images": all_boxes}, ensure_ascii=False), encoding="utf-8")
    return JSONResponse({"job_id": job_id, "images": images_payload})


@app.get("/api/preview/{job_id}/{index}")
def preview_image(job_id: str, index: int):
    input_dir = WORK_DIR / job_id / "inputs"
    if not input_dir.exists():
        raise HTTPException(status_code=404, detail="任务不存在或已过期。")
    matches = sorted(input_dir.glob(f"{index + 1:03d}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="图片不存在。")
    return FileResponse(matches[0])


@app.post("/api/convert")
async def convert(request: Request):
    """Build the PPT from a previous /api/ocr job, editing only the selected boxes."""
    payload = await request.json()
    job_id = payload.get("job_id")
    edit = payload.get("edit")  # list per image of selected box indices

    job_dir = WORK_DIR / job_id if job_id else None
    if not job_dir or not job_dir.exists():
        raise HTTPException(status_code=404, detail="任务不存在或已过期，请重新上传。")

    ocr_file = job_dir / "ocr.json"
    if not ocr_file.exists():
        raise HTTPException(status_code=404, detail="未找到 OCR 结果，请重新上传。")

    image_paths = sorted((job_dir / "inputs").iterdir())
    if not image_paths:
        raise HTTPException(status_code=404, detail="未找到上传的图片。")

    stored = json.loads(ocr_file.read_text(encoding="utf-8"))["images"]
    per_image_boxes = [
        [OcrTextBox(text=b["text"], confidence=b["confidence"], box=[tuple(p) for p in b["box"]]) for b in boxes]
        for boxes in stored
    ]

    # normalise the selection into a set of indices per image
    selection: list[set[int]] = []
    for image_index, boxes in enumerate(per_image_boxes):
        if edit and image_index < len(edit) and edit[image_index] is not None:
            selection.append({int(i) for i in edit[image_index] if 0 <= int(i) < len(boxes)})
        else:
            selection.append(set(range(len(boxes))))  # default: edit all

    with _lock:
        _jobs[job_id] = {
            "status": "processing",
            "stage": "inpaint",
            "current": 0,
            "total": len(image_paths),
            "error": None,
        }

    threading.Thread(
        target=_run_conversion, args=(job_id, image_paths, job_dir, per_image_boxes, selection), daemon=True
    ).start()

    return JSONResponse({"job_id": job_id})


def _run_conversion(
    job_id: str,
    image_paths: list[Path],
    job_dir: Path,
    per_image_boxes: list[list[OcrTextBox]],
    selection: list[set[int]],
):
    try:
        from app.services.inpaint import remove_text_from_image
        from app.services.ppt import build_pptx

        processed_slides = []
        cleaned_dir = job_dir / "cleaned"
        cleaned_dir.mkdir(parents=True, exist_ok=True)

        for index, image_path in enumerate(image_paths):
            _set_progress(job_id, stage="inpaint", current=index, total=len(image_paths))
            boxes = per_image_boxes[index] if index < len(per_image_boxes) else []
            chosen = selection[index] if index < len(selection) else set(range(len(boxes)))
            editable = [box for i, box in enumerate(boxes) if i in chosen]

            cleaned_image = cleaned_dir / f"slide-{index + 1:03d}.png"
            remove_text_from_image(image_path, editable, cleaned_image)
            processed_slides.append((image_path, cleaned_image, editable))

        _set_progress(job_id, stage="PPTX", current=len(image_paths), total=len(image_paths))
        pptx_path = build_pptx(processed_slides, job_dir / "editable-images.pptx")
        logger.info("Job %s completed: %s", job_id, pptx_path)

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
