"""Configure Tesseract for unstructured hi_res PDF processing on Windows."""

import os
import shutil
from pathlib import Path


def configure_tesseract() -> None:
    """Add Tesseract to PATH if installed in the default Windows location."""
    tesseract = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if tesseract.exists():
        os.environ["PATH"] = str(tesseract.parent) + os.pathsep + os.environ.get("PATH", "")
        try:
            import unstructured_pytesseract

            unstructured_pytesseract.pytesseract.tesseract_cmd = str(tesseract)
        except ImportError:
            pass
    elif shutil.which("tesseract") is None:
        raise RuntimeError(
            "Tesseract not found. Install from https://github.com/UB-Mannheim/tesseract/wiki "
            "or ensure tesseract is on PATH."
        )
