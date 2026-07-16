from __future__ import annotations

import logging
from pathlib import Path

from app.services.inpaint import remove_text_from_image
from app.services.ocr import detect_text
from app.services.ppt import build_pptx


logger = logging.getLogger(__name__)


def convert_images_to_pptx(image_paths: list[Path], output_dir: Path) -> Path:
    processed_slides = []
    cleaned_dir = output_dir / "cleaned"

    for index, image_path in enumerate(image_paths, start=1):
        logger.info("Processing slide %s/%s: %s", index, len(image_paths), image_path)
        text_boxes = detect_text(image_path)
        logger.info("Slide %s OCR detected %s text boxes", index, len(text_boxes))
        cleaned_image = cleaned_dir / f"slide-{index:03d}.png"
        remove_text_from_image(image_path, text_boxes, cleaned_image)
        logger.info("Slide %s cleaned image saved: %s", index, cleaned_image)
        processed_slides.append((image_path, cleaned_image, text_boxes))

    pptx_path = build_pptx(processed_slides, output_dir / "editable-images.pptx")
    logger.info("PPT generated: %s", pptx_path)
    return pptx_path
