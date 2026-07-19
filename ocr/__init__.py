"""Hebrew OCR — PaddleOCR-based Hebrew recognizers + a runnable page pipeline.

Public API:
    from ocr import HebrewOCR
    ocr = HebrewOCR(models_dir="hebrew-ocr-models")   # a downloaded HF snapshot
    result = ocr.read("page.png")                      # {lines, words, meta}
"""
from .pipeline import HebrewOCR, render_page
from .plan_e_rec import PlanECascade
from .heb_postprocess import (
    post_process, fix_geresh_yod_text, fix_geresh_yod_token, reading_order,
)

__version__ = "0.1.0"
__all__ = [
    "HebrewOCR", "render_page", "PlanECascade",
    "post_process", "fix_geresh_yod_text", "fix_geresh_yod_token", "reading_order",
]
