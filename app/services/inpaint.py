from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.config import settings
from app.services.ocr import OcrTextBox
from app.services.text_style import estimate_text_appearance
from app.services.text_rules import should_keep_as_background


_lama_model = None


def remove_text_from_image(image_path: Path, text_boxes: list[OcrTextBox], output_path: Path) -> Path:
    if settings.inpaint_provider == "none":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image_path, output_path)
        return output_path

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片：{image_path}")

    if settings.inpaint_provider == "hybrid":
        cleaned = _erase_hybrid(image, text_boxes)
    elif settings.inpaint_provider in {"opencv", "smart"}:
        cleaned = _erase_text_strokes(image, text_boxes)
    elif settings.inpaint_provider == "lama":
        mask = build_text_mask(image, text_boxes)
        cleaned = _inpaint_with_lama(image, mask) if np.any(mask) else image
    else:
        raise ValueError(f"Unsupported INPAINT_PROVIDER: {settings.inpaint_provider}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cleaned)
    return output_path


def _erase_hybrid(image: np.ndarray, text_boxes: list[OcrTextBox]) -> np.ndarray:
    """Simple backgrounds use OpenCV stroke fill; complex (textured) regions use LaMa."""
    height, width = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cleaned = image.copy()
    complex_mask = np.zeros((height, width), dtype=np.uint8)

    for text_box in text_boxes:
        if should_keep_as_background(text_box, width, height):
            continue
        appearance = estimate_text_appearance(image_rgb, text_box)
        if appearance is None:
            continue
        if appearance.light_text_on_colored_background:
            continue
        if _background_is_simple(image_rgb, appearance):
            roi = cleaned[appearance.top : appearance.bottom, appearance.left : appearance.right]
            filled = _fill_with_row_background(roi, appearance.stroke_mask, appearance.background_rgb)
            cleaned[appearance.top : appearance.bottom, appearance.left : appearance.right] = filled
        else:
            kernel = np.ones((3, 3), dtype=np.uint8)
            dilated = cv2.dilate(appearance.stroke_mask.astype(np.uint8), kernel, iterations=1)
            complex_mask[appearance.top : appearance.bottom, appearance.left : appearance.right] = cv2.bitwise_or(
                complex_mask[appearance.top : appearance.bottom, appearance.left : appearance.right],
                dilated,
            )

    if np.any(complex_mask):
        complex_mask = refine_text_mask(complex_mask)
        cleaned = _inpaint_with_lama(cleaned, complex_mask)

    return cleaned


def _background_is_simple(image_rgb: np.ndarray, appearance) -> bool:
    """A background is 'simple' if it is flat or a smooth gradient (low edge energy)."""
    roi = image_rgb[appearance.top : appearance.bottom, appearance.left : appearance.right]
    if roi.size == 0:
        return True
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY).astype(np.float32)
    kernel = np.ones((5, 5), dtype=np.uint8)
    text_zone = cv2.dilate(appearance.stroke_mask.astype(np.uint8), kernel, iterations=2)
    bg_zone = text_zone == 0
    if int(bg_zone.sum()) < 24:
        return True
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    edge_energy = float(laplacian[bg_zone].mean())
    return edge_energy < settings.hybrid_edge_threshold


def build_text_mask(image_or_shape: np.ndarray | tuple[int, int], text_boxes: list[OcrTextBox]) -> np.ndarray:
    if isinstance(image_or_shape, np.ndarray):
        return _build_precise_text_mask(image_or_shape, text_boxes)

    height, width = image_or_shape
    mask = np.zeros((height, width), dtype=np.uint8)

    for text_box in text_boxes:
        x1, y1, x2, y2 = text_box.bounds
        padding = max(2, int(max(x2 - x1, y2 - y1) * settings.mask_padding_ratio))
        left = max(0, int(x1 - padding))
        top = max(0, int(y1 - padding))
        right = min(width, int(x2 + padding))
        bottom = min(height, int(y2 + padding))
        cv2.rectangle(mask, (left, top), (right, bottom), 255, thickness=-1)

    return refine_text_mask(mask)


def refine_text_mask(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    refined = cv2.dilate(mask, kernel, iterations=1)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel, iterations=1)
    return refined


def _build_precise_text_mask(image: np.ndarray, text_boxes: list[OcrTextBox]) -> np.ndarray:
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for text_box in text_boxes:
        if should_keep_as_background(text_box, width, height):
            continue

        appearance = estimate_text_appearance(image_rgb, text_box)
        if appearance is None:
            continue
        mask[appearance.top:appearance.bottom, appearance.left:appearance.right] = cv2.bitwise_or(
            mask[appearance.top:appearance.bottom, appearance.left:appearance.right],
            appearance.stroke_mask,
        )

    return mask


def _erase_text_strokes(image: np.ndarray, text_boxes: list[OcrTextBox]) -> np.ndarray:
    height, width = image.shape[:2]
    cleaned = image.copy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for text_box in text_boxes:
        if should_keep_as_background(text_box, width, height):
            continue

        appearance = estimate_text_appearance(image_rgb, text_box)
        if appearance is None:
            continue

        if appearance.light_text_on_colored_background:
            # Colored title bars are already clean vector-like UI elements in the
            # source image. Wiping their text tends to expand into icons,
            # rounded corners, and bar edges, so preserve them as raster.
            continue

        roi = cleaned[appearance.top:appearance.bottom, appearance.left:appearance.right]
        roi = _fill_with_row_background(roi, appearance.stroke_mask, appearance.background_rgb)
        cleaned[appearance.top:appearance.bottom, appearance.left:appearance.right] = roi

    return cleaned


def _text_area_mask(stroke_mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(stroke_mask > 0)
    if xs.size == 0 or ys.size == 0:
        return stroke_mask

    x1 = max(0, int(xs.min()) - 2)
    x2 = min(stroke_mask.shape[1], int(xs.max()) + 3)
    y1 = max(0, int(ys.min()) - 1)
    y2 = min(stroke_mask.shape[0], int(ys.max()) + 2)
    area_mask = np.zeros_like(stroke_mask)
    area_mask[y1:y2, x1:x2] = 255
    return area_mask


def _fill_with_row_background(
    image: np.ndarray,
    mask: np.ndarray,
    background_rgb: tuple[int, int, int],
) -> np.ndarray:
    cleaned = image.copy()
    background_bgr = np.array(background_rgb[::-1], dtype=np.float32)
    pixels = image.astype(np.float32)
    distance_to_background = np.linalg.norm(pixels - background_bgr, axis=2)
    reusable = (mask == 0) & (distance_to_background <= 62)

    fallback = np.array(background_rgb[::-1], dtype=np.uint8)
    for y in range(image.shape[0]):
        xs = np.where(mask[y] > 0)[0]
        if xs.size == 0:
            continue

        candidates = image[y][reusable[y]]
        if candidates.size:
            fill = np.median(candidates.reshape(-1, 3), axis=0).astype(np.uint8)
        else:
            fill = fallback
        cleaned[y, xs] = fill

    return cleaned


def _inpaint_with_lama(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    lama = _get_lama_model()
    crop_box = _calculate_lama_crop_box(mask, image.shape[1], image.shape[0])

    if crop_box is None:
        return image.copy()

    x1, y1, x2, y2 = crop_box
    crop_image = image[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]
    resized_image, resized_mask, resize_scale = _resize_for_lama(crop_image, crop_mask)

    image_rgb = Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    mask_image = Image.fromarray(resized_mask).convert("L")
    result = lama(image_rgb, mask_image)
    result_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

    if resize_scale != 1:
        result_bgr = cv2.resize(result_bgr, (crop_image.shape[1], crop_image.shape[0]), interpolation=cv2.INTER_CUBIC)

    cleaned = image.copy()
    alpha = _soft_alpha(crop_mask)
    original_crop = cleaned[y1:y2, x1:x2]
    blended_crop = (result_bgr.astype(np.float32) * alpha + original_crop.astype(np.float32) * (1 - alpha)).astype(np.uint8)
    cleaned[y1:y2, x1:x2] = blended_crop
    return cleaned


def _get_lama_model():
    global _lama_model

    if _lama_model is not None:
        return _lama_model

    try:
        import torch
        from simple_lama_inpainting.models.model import LAMA_MODEL_URL
        from simple_lama_inpainting.utils import download_model, prepare_img_and_mask
    except ImportError as error:
        raise RuntimeError(
            "INPAINT_PROVIDER=lama 需要额外安装 simple-lama-inpainting；"
            "如果部署环境不方便安装，请改用 INPAINT_PROVIDER=opencv。"
        ) from error

    device = _select_lama_device(torch)
    model_path = download_model(LAMA_MODEL_URL)
    _lama_model = _TorchScriptLama(torch, prepare_img_and_mask, model_path, device)
    return _lama_model


def _select_lama_device(torch) -> object:
    if settings.lama_device == "cpu":
        return torch.device("cpu")
    if settings.lama_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("LAMA_DEVICE=cuda 但当前 PyTorch 没有可用 CUDA，请安装 CUDA 版 PyTorch。")
        return torch.device("cuda")
    if settings.lama_device != "auto":
        raise ValueError(f"Unsupported LAMA_DEVICE: {settings.lama_device}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _TorchScriptLama:
    def __init__(self, torch, prepare_img_and_mask, model_path: str, device: object) -> None:
        self._torch = torch
        self._prepare_img_and_mask = prepare_img_and_mask
        self.device = device
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        self.model.to(device)

    def __call__(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        image_tensor, mask_tensor = self._prepare_img_and_mask(image, mask, self.device)
        with self._torch.inference_mode():
            inpainted = self.model(image_tensor, mask_tensor)

        result = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(result)


def _calculate_lama_crop_box(mask: np.ndarray, image_width: int, image_height: int) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    padding = int(max(image_width, image_height) * settings.lama_crop_padding_ratio)

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image_width, x2 + padding)
    y2 = min(image_height, y2 + padding)

    crop_area = (x2 - x1) * (y2 - y1)
    image_area = image_width * image_height
    if crop_area > image_area * 0.82:
        return 0, 0, image_width, image_height

    return x1, y1, x2, y2


def _resize_for_lama(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    max_side = max(image.shape[:2])
    if max_side <= settings.lama_max_side:
        return image, mask, 1

    scale = settings.lama_max_side / max_side
    target_width = max(1, int(image.shape[1] * scale))
    target_height = max(1, int(image.shape[0] * scale))
    resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    return resized_image, resized_mask, scale


def _soft_alpha(mask: np.ndarray) -> np.ndarray:
    alpha = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=1.6, sigmaY=1.6)
    alpha = (alpha.astype(np.float32) / 255.0)[:, :, None]
    return np.clip(alpha, 0, 1)
