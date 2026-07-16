from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.config import settings
from app.services.ocr import OcrTextBox


@dataclass(slots=True)
class TextAppearance:
    left: int
    top: int
    right: int
    bottom: int
    stroke_mask: np.ndarray
    text_rgb: tuple[int, int, int]
    background_rgb: tuple[int, int, int]
    light_text_on_colored_background: bool


def estimate_text_appearance(image_rgb: np.ndarray, text_box: OcrTextBox) -> TextAppearance | None:
    image_height, image_width = image_rgb.shape[:2]
    x1, y1, x2, y2 = text_box.bounds
    padding = max(1, int(max(x2 - x1, y2 - y1) * settings.mask_padding_ratio))
    left = max(0, int(x1 - padding))
    top = max(0, int(y1 - padding))
    right = min(image_width, int(x2 + padding))
    bottom = min(image_height, int(y2 + padding))
    if right <= left or bottom <= top:
        return None

    poly_mask = _polygon_mask(text_box, image_width, image_height)[top:bottom, left:right]
    if not np.any(poly_mask):
        return None

    roi = image_rgb[top:bottom, left:right]
    background_rgb = _estimate_background_rgb(roi, poly_mask)
    foreground_rgb, foreground_distance = _estimate_foreground_rgb(roi, poly_mask, background_rgb)
    if foreground_rgb is None:
        return None

    light_text_on_colored_background = _is_light_text_on_colored_background(foreground_rgb, background_rgb)
    stroke_mask = _foreground_mask(
        roi,
        poly_mask,
        foreground_rgb,
        background_rgb,
        foreground_distance,
        light_text_on_colored_background,
    )
    if not np.any(stroke_mask):
        return None

    return TextAppearance(
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        stroke_mask=stroke_mask,
        text_rgb=foreground_rgb,
        background_rgb=background_rgb,
        light_text_on_colored_background=light_text_on_colored_background,
    )


def _polygon_mask(text_box: OcrTextBox, image_width: int, image_height: int) -> np.ndarray:
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    points = np.array(
        [[max(0, min(image_width - 1, int(x))), max(0, min(image_height - 1, int(y)))] for x, y in text_box.box],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [points], 255)
    return mask


def _estimate_background_rgb(roi: np.ndarray, poly_mask: np.ndarray) -> tuple[int, int, int]:
    gray = _gray(roi)
    saturation = _saturation(roi)
    dominant_inside = _dominant_inside_rgb(roi, poly_mask)
    if dominant_inside is not None:
        return dominant_inside

    outside_pixels = roi[poly_mask == 0]
    calm_inside_pixels = roi[(poly_mask > 0) & (gray > 238) & (saturation < 35)]

    if outside_pixels.size and calm_inside_pixels.size:
        candidates = np.vstack([outside_pixels.reshape(-1, 3), calm_inside_pixels.reshape(-1, 3)])
    elif outside_pixels.size:
        candidates = outside_pixels.reshape(-1, 3)
    elif calm_inside_pixels.size:
        candidates = calm_inside_pixels.reshape(-1, 3)
    else:
        candidates = roi.reshape(-1, 3)

    red, green, blue = np.median(candidates, axis=0).astype(int).tolist()
    return red, green, blue


def _dominant_inside_rgb(roi: np.ndarray, poly_mask: np.ndarray) -> tuple[int, int, int] | None:
    inside_pixels = roi[poly_mask > 0]
    if inside_pixels.size == 0:
        return None

    quantized = (inside_pixels // 16).astype(np.uint8)
    buckets, counts = np.unique(quantized, axis=0, return_counts=True)
    if counts.size == 0:
        return None

    order = np.argsort(counts)[::-1]
    total = inside_pixels.shape[0]
    for index in order[:3]:
        share = counts[index] / total
        if share < 0.18:
            continue

        bucket = buckets[index]
        cluster_mask = np.all(quantized == bucket, axis=1)
        cluster_pixels = inside_pixels[cluster_mask]
        red, green, blue = np.median(cluster_pixels, axis=0).astype(int).tolist()
        return red, green, blue

    return None


def _estimate_foreground_rgb(
    roi: np.ndarray,
    poly_mask: np.ndarray,
    background_rgb: tuple[int, int, int],
) -> tuple[tuple[int, int, int] | None, float]:
    pixels = roi.reshape(-1, 3).astype(np.float32)
    inside = (poly_mask.reshape(-1) > 0)
    if not np.any(inside):
        return None, 0

    background = np.array(background_rgb, dtype=np.float32)
    distance = np.linalg.norm(pixels - background, axis=1)
    gray = _gray(roi).reshape(-1)
    saturation = _saturation(roi).reshape(-1)

    background_gray = background[0] * 0.299 + background[1] * 0.587 + background[2] * 0.114
    background_saturation = float(background.max() - background.min())
    light_text_on_colored_background = background_gray < 200 or background_saturation > 50
    if light_text_on_colored_background:
        candidates = inside & (distance > 28) & (gray > background_gray + 30)
    else:
        candidates = inside & (distance > 24) & ((gray < 245) | (saturation > 30))
    if not np.any(candidates):
        candidates = inside & (distance > 16)
    if not np.any(candidates):
        return None, 0

    candidate_pixels = pixels[candidates]
    percentile = 65 if light_text_on_colored_background else 35
    red, green, blue = np.percentile(candidate_pixels, percentile, axis=0).astype(int).tolist()
    foreground = (red, green, blue)
    foreground_distance = float(np.linalg.norm(np.array(foreground, dtype=np.float32) - background))
    return foreground, foreground_distance


def _foreground_mask(
    roi: np.ndarray,
    poly_mask: np.ndarray,
    foreground_rgb: tuple[int, int, int],
    background_rgb: tuple[int, int, int],
    foreground_distance: float,
    light_text_on_colored_background: bool,
) -> np.ndarray:
    pixels = roi.astype(np.float32)
    foreground = np.array(foreground_rgb, dtype=np.float32)
    background = np.array(background_rgb, dtype=np.float32)
    distance_to_foreground = np.linalg.norm(pixels - foreground, axis=2)
    distance_to_background = np.linalg.norm(pixels - background, axis=2)
    threshold = max(30, min(70, foreground_distance * 0.62))
    edge_threshold = max(8, min(22, foreground_distance * 0.18))
    if light_text_on_colored_background:
        mask = (
            (poly_mask > 0)
            & (distance_to_background >= edge_threshold)
            & (distance_to_foreground <= threshold)
        ).astype(np.uint8) * 255
        protected_mask = _icon_like_components(mask)
    else:
        mask = (
            (poly_mask > 0)
            & (distance_to_background >= edge_threshold)
            & (
                (distance_to_foreground <= threshold)
                | (distance_to_background >= max(18, foreground_distance * 0.28))
            )
        ).astype(np.uint8) * 255
        protected_mask = np.zeros_like(mask)
    if np.any(mask):
        mask[protected_mask > 0] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)
        dilation_size = _dilation_size_for_colored_text(roi) if light_text_on_colored_background else _dilation_size_for_roi(roi)
        mask = cv2.dilate(mask, np.ones((dilation_size, dilation_size), np.uint8), iterations=1)
        mask[protected_mask > 0] = 0
    return mask


def _icon_like_components(mask: np.ndarray) -> np.ndarray:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    protected = np.zeros_like(mask)
    if component_count <= 1:
        return protected

    height, width = mask.shape
    left_icon_limit = max(28, int(width * 0.24))
    for label in range(1, component_count):
        x, y, component_width, component_height, area = stats[label]
        if area < 90:
            continue

        near_left = x < left_icon_limit
        square_like = 0.55 <= component_width / max(component_height, 1) <= 1.8
        icon_sized = component_height >= max(12, height * 0.45) or component_width >= max(12, height * 0.45)
        if near_left and square_like and icon_sized:
            protected[labels == label] = 255

    if np.any(protected):
        protected = cv2.dilate(protected, np.ones((3, 3), np.uint8), iterations=1)
    return protected


def _is_light_text_on_colored_background(
    foreground_rgb: tuple[int, int, int],
    background_rgb: tuple[int, int, int],
) -> bool:
    foreground = np.array(foreground_rgb, dtype=np.float32)
    background = np.array(background_rgb, dtype=np.float32)
    background_gray = background[0] * 0.299 + background[1] * 0.587 + background[2] * 0.114
    foreground_gray = foreground[0] * 0.299 + foreground[1] * 0.587 + foreground[2] * 0.114
    background_saturation = float(background.max() - background.min())
    return (background_gray < 200 or background_saturation > 50) and foreground_gray > background_gray + 30


def _dilation_size_for_roi(roi: np.ndarray) -> int:
    height, width = roi.shape[:2]
    long_side = max(height, width)
    if height >= 32 or long_side >= 380:
        return 9
    if height >= 22 or long_side >= 180:
        return 7
    return 5


def _dilation_size_for_colored_text(roi: np.ndarray) -> int:
    height, width = roi.shape[:2]
    if height >= 28 or width >= 180:
        return 9
    if height >= 20 or width >= 100:
        return 7
    return 5


def _gray(rgb: np.ndarray) -> np.ndarray:
    return rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114


def _saturation(rgb: np.ndarray) -> np.ndarray:
    return rgb.max(axis=2).astype(np.float32) - rgb.min(axis=2).astype(np.float32)
