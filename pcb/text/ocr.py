"""Compatibility exports for existing OCR implementations."""

from ..ocr_backends import HybridOCR, SelectiveLocalOCR, TiledEasyOCR, TiledEasyOCRConfig
from ..vision import OCR

__all__ = ["OCR", "HybridOCR", "SelectiveLocalOCR", "TiledEasyOCR", "TiledEasyOCRConfig"]
