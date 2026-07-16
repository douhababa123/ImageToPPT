from __future__ import annotations

import re

from app.services.ocr import OcrTextBox


LOGO_TERMS = (
    "HIKROBOT",
    "UNIVERSAL",
    "ROBOTS",
)


def should_keep_as_background(text_box: OcrTextBox, image_width: int, image_height: int) -> bool:
    text = text_box.text.strip()
    if not text:
        return True

    x1, y1, x2, y2 = text_box.bounds
    normalized = re.sub(r"[^A-Z0-9]+", "", text.upper())
    letters = [character for character in text if character.isalpha()]
    uppercase_ratio = (
        sum(1 for character in letters if character.upper() == character) / len(letters)
        if letters
        else 0
    )

    if len(text) == 1 and text_box.confidence < 0.9:
        return True

    in_logo_zone = y1 < image_height * 0.28 or x1 > image_width * 0.45
    logo_like = len(text) <= 28 or uppercase_ratio >= 0.72
    if in_logo_zone and logo_like and (any(term in normalized for term in LOGO_TERMS) or normalized == "UR"):
        return True

    # OCR often picks tiny icon labels or logo fragments as editable text.
    width = x2 - x1
    height = y2 - y1
    if len(text) <= 2 and max(width, height) < image_width * 0.04:
        return True

    return False
