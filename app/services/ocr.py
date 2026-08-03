from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from app.config import settings


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OcrTextBox:
    text: str
    confidence: float
    box: list[tuple[float, float]]

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        xs = [point[0] for point in self.box]
        ys = [point[1] for point in self.box]
        return min(xs), min(ys), max(xs), max(ys)


_local_ocr: Any | None = None


def detect_text(image_path: Path) -> list[OcrTextBox]:
    if settings.ocr_provider == "local":
        return _detect_with_local_paddleocr(image_path)

    if settings.ocr_provider == "paddle_api":
        try:
            return _detect_with_paddle_api(image_path)
        except Exception as error:
            if settings.ocr_fallback_provider == "local":
                logger.warning("Paddle API OCR failed, falling back to local PaddleOCR: %s", error)
                return _detect_with_local_paddleocr(image_path)
            raise

    raise ValueError(f"Unsupported OCR_PROVIDER: {settings.ocr_provider}")


def _detect_with_paddle_api(image_path: Path) -> list[OcrTextBox]:
    if not settings.paddle_api_key:
        raise RuntimeError("OCR_PROVIDER=paddle_api 时必须配置 PADDLE_API_KEY。")

    with image_path.open("rb") as image_file:
        file_data = base64.b64encode(image_file.read()).decode("ascii")

    payload = {
            "file": file_data,
            "fileType": 1,
            "useDocOrientationClassify": False,
            "useDocUnwarping": False,
            "useChartRecognition": False,
        }
    headers = {
            "Authorization": f"token {settings.paddle_api_key}",
            "Content-Type": "application/json",
        }

    logger.info("Calling Paddle OCR API: url=%s use_env_proxy=%s image=%s", settings.paddle_api_url, settings.paddle_api_use_env_proxy, image_path)
    session = requests.Session()
    session.trust_env = settings.paddle_api_use_env_proxy
    try:
        response = session.post(
            settings.paddle_api_url,
            json=payload,
            headers=headers,
            timeout=settings.paddle_api_timeout,
        )
    except requests.exceptions.ProxyError as error:
        raise RuntimeError(
            "飞桨 OCR 请求失败：当前网络代理要求认证（HTTP 407）。"
            "请配置可用的 HTTPS_PROXY/HTTP_PROXY，或在 .env 中设置 PADDLE_API_USE_ENV_PROXY=false 尝试直连。"
        ) from error
    except requests.exceptions.ConnectionError as error:
        message = str(error)
        if "getaddrinfo failed" in message:
            raise RuntimeError(
                "飞桨 OCR 请求失败：当前网络无法解析 aistudio-app.com 域名。"
                "这通常表示公司网络不允许直连，需要配置带认证的代理；程序将尝试使用本地 PaddleOCR 兜底。"
            ) from error
        raise RuntimeError(f"飞桨 OCR 请求失败：{error}") from error
    except requests.exceptions.RequestException as error:
        raise RuntimeError(f"飞桨 OCR 请求失败：{error}") from error

    logger.info("Paddle OCR API returned status=%s", response.status_code)
    response.raise_for_status()

    payload = response.json()
    text_boxes = _parse_paddle_api_result(payload)
    if not text_boxes:
        raise RuntimeError("飞桨 OCR 返回成功，但没有解析到带坐标的文字框；请确认该接口返回 OCR 坐标字段。")
    return text_boxes


def _detect_with_local_paddleocr(image_path: Path) -> list[OcrTextBox]:
    global _local_ocr

    if _local_ocr is None:
        # On Windows, PaddlePaddle can load DLLs that break a later torch import.
        # PaddleOCR imports albumentations, which imports torch, so load torch first.
        import torch  # noqa: F401
        from paddleocr import PaddleOCR

        _local_ocr = PaddleOCR(
            use_angle_cls=True,
            lang=settings.ocr_lang,
            cpu_threads=1,
            use_mp=False,
            total_process_num=1,
        )

    raw_result = _local_ocr.ocr(str(image_path), cls=True)
    return _parse_paddleocr_result(raw_result)


def _parse_paddleocr_result(raw_result: Any) -> list[OcrTextBox]:
    text_boxes: list[OcrTextBox] = []

    if not raw_result:
        return text_boxes

    # PaddleOCR 2.x usually returns: [[[[x,y]...], (text, score)], ...]
    pages = raw_result if _looks_like_pages(raw_result) else [raw_result]
    for page in pages:
        if not page:
            continue
        if isinstance(page, dict):
            text_boxes.extend(_parse_dict_page(page))
            continue

        for item in page:
            parsed = _parse_paddle_line(item)
            if parsed is None:
                continue
            text_boxes.append(parsed)

    return text_boxes


def _looks_like_pages(raw_result: Any) -> bool:
    if not isinstance(raw_result, list) or not raw_result:
        return False
    first = raw_result[0]
    return isinstance(first, (list, dict)) and not _looks_like_paddle_line(first)


def _parse_dict_page(page: dict[str, Any]) -> list[OcrTextBox]:
    texts = page.get("rec_texts") or []
    scores = page.get("rec_scores") or []
    boxes = page.get("rec_polys") or page.get("dt_polys") or page.get("rec_boxes") or page.get("dt_boxes") or []
    result: list[OcrTextBox] = []

    for text, score, box in zip(texts, scores, boxes):
        points = _normalize_box(box)
        if points is None:
            continue
        try:
            confidence = float(score)
        except (TypeError, ValueError):
            continue
        if str(text).strip():
            result.append(OcrTextBox(text=str(text), confidence=confidence, box=points))
    return result


def _parse_paddle_api_result(payload: Any) -> list[OcrTextBox]:
    root = payload.get("result", payload) if isinstance(payload, dict) else payload
    text_boxes: list[OcrTextBox] = []
    _collect_ocr_boxes(root, text_boxes)
    return _deduplicate_text_boxes(text_boxes)


def _collect_ocr_boxes(node: Any, text_boxes: list[OcrTextBox]) -> None:
    if isinstance(node, dict):
        text_boxes.extend(_parse_dict_page(node))

        single = _parse_single_ocr_dict(node)
        if single is not None:
            text_boxes.append(single)

        for value in node.values():
            _collect_ocr_boxes(value, text_boxes)
        return

    if isinstance(node, list):
        if _looks_like_paddle_line(node):
            parsed = _parse_paddleocr_result([node])
            text_boxes.extend(parsed)
            return

        for item in node:
            _collect_ocr_boxes(item, text_boxes)


def _parse_single_ocr_dict(item: dict[str, Any]) -> OcrTextBox | None:
    text = _first_present(item, "text", "transcription", "content", "word")
    box = _first_present(item, "box", "bbox", "poly", "polygon", "points", "quad", "coordinate", "position")
    if text is None or box is None:
        return None

    text_value = str(text).strip()
    if not text_value or len(text_value) > 500:
        return None

    points = _normalize_box(box)
    if points is None:
        return None

    score = _first_present(item, "score", "confidence", "prob", "rec_score")
    try:
        confidence = float(score) if score is not None else 1.0
    except (TypeError, ValueError):
        confidence = 1.0

    return OcrTextBox(text=text_value, confidence=confidence, box=points)


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _normalize_box(box: Any) -> list[tuple[float, float]] | None:
    if isinstance(box, dict):
        if all(key in box for key in ("x1", "y1", "x2", "y2")):
            return _rectangle_to_points(box["x1"], box["y1"], box["x2"], box["y2"])
        if all(key in box for key in ("left", "top", "width", "height")):
            return _rectangle_to_points(box["left"], box["top"], box["left"] + box["width"], box["top"] + box["height"])
        return None

    if not isinstance(box, (list, tuple)):
        return None

    if len(box) == 4 and all(_is_number(value) for value in box):
        return _rectangle_to_points(box[0], box[1], box[2], box[3])

    if len(box) == 8 and all(_is_number(value) for value in box):
        return [(float(box[index]), float(box[index + 1])) for index in range(0, 8, 2)]

    if len(box) >= 4 and all(isinstance(point, (list, tuple)) and len(point) >= 2 for point in box[:4]):
        try:
            return [(float(point[0]), float(point[1])) for point in box[:4]]
        except (TypeError, ValueError):
            return None

    return None


def _rectangle_to_points(x1: Any, y1: Any, x2: Any, y2: Any) -> list[tuple[float, float]] | None:
    try:
        left, top, right, bottom = float(x1), float(y1), float(x2), float(y2)
    except (TypeError, ValueError):
        return None
    return [(left, top), (right, top), (right, bottom), (left, bottom)]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _looks_like_paddle_line(item: list[Any]) -> bool:
    return _parse_paddle_line(item) is not None


def _parse_paddle_line(item: Any) -> OcrTextBox | None:
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        return None
    if not isinstance(item[1], (list, tuple)) or len(item[1]) < 2:
        return None
    if not isinstance(item[1][0], str):
        return None
    if not _is_number(item[1][1]):
        return None

    points = _normalize_box(item[0])
    if points is None:
        return None

    text = item[1][0].strip()
    if not text:
        return None

    return OcrTextBox(text=text, confidence=float(item[1][1]), box=points)


def _deduplicate_text_boxes(text_boxes: list[OcrTextBox]) -> list[OcrTextBox]:
    seen: set[tuple[str, tuple[int, int, int, int]]] = set()
    unique: list[OcrTextBox] = []
    for text_box in text_boxes:
        x1, y1, x2, y2 = text_box.bounds
        key = (text_box.text, (round(x1), round(y1), round(x2), round(y2)))
        if key in seen:
            continue
        seen.add(key)
        unique.append(text_box)
    return unique
