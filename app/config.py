from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


class Settings:
    ocr_provider: str = os.getenv("OCR_PROVIDER", "local")
    ocr_fallback_provider: str = os.getenv("OCR_FALLBACK_PROVIDER", "local")
    ocr_lang: str = os.getenv("OCR_LANG", "ch")
    inpaint_provider: str = os.getenv("INPAINT_PROVIDER", "hybrid")
    hybrid_edge_threshold: float = float(os.getenv("HYBRID_EDGE_THRESHOLD", "8.0"))
    slide_mode: str = os.getenv("SLIDE_MODE", "source")
    slide_width_inches: float = float(os.getenv("SLIDE_WIDTH_INCHES", "13.333333"))
    slide_height_inches: float = float(os.getenv("SLIDE_HEIGHT_INCHES", "7.5"))
    image_fit: str = os.getenv("IMAGE_FIT", "cover")
    latin_font: str = os.getenv("LATIN_FONT", "微软雅黑")
    cjk_font: str = os.getenv("CJK_FONT", "微软雅黑")
    mask_padding_ratio: float = float(os.getenv("MASK_PADDING_RATIO", "0.12"))
    lama_crop_padding_ratio: float = float(os.getenv("LAMA_CROP_PADDING_RATIO", "0.08"))
    lama_max_side: int = int(os.getenv("LAMA_MAX_SIDE", "1600"))
    lama_device: str = os.getenv("LAMA_DEVICE", "auto").lower()
    output_dir: Path = Path(os.getenv("OUTPUT_DIR", "outputs"))
    log_dir: Path = Path(os.getenv("LOG_DIR", "logs"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    paddle_api_key: str = os.getenv("PADDLE_API_KEY", "")
    paddle_api_url: str = os.getenv("PADDLE_API_URL", "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs")
    paddle_api_model: str = os.getenv("PADDLE_API_MODEL", "PaddleOCR-VL-1.6")
    paddle_api_timeout: int = int(os.getenv("PADDLE_API_TIMEOUT", "180"))
    paddle_poll_interval: float = float(os.getenv("PADDLE_POLL_INTERVAL", "3"))
    paddle_api_use_env_proxy: bool = os.getenv("PADDLE_API_USE_ENV_PROXY", "false").lower() == "true"
    max_upload_files: int = int(os.getenv("MAX_UPLOAD_FILES", "20"))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "25"))
    job_ttl_hours: float = float(os.getenv("JOB_TTL_HOURS", "24"))


settings = Settings()
