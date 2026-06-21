"""Configure Tesseract for unstructured hi_res PDF processing."""

import os
import platform
import shutil
from pathlib import Path


def configure_tesseract() -> None:
    """Set the tesseract_cmd for pytesseract. Handles Windows install path; on Linux finds it in PATH."""
    if platform.system() == "Windows":
        tesseract = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if tesseract.exists():
            os.environ["PATH"] = str(tesseract.parent) + os.pathsep + os.environ.get("PATH", "")
            _set_tesseract_cmd(str(tesseract))
        elif shutil.which("tesseract"):
            _set_tesseract_cmd(shutil.which("tesseract"))
        else:
            raise RuntimeError(
                "Tesseract not found. Install from https://github.com/UB-Mannheim/tesseract/wiki "
                "or ensure tesseract is on PATH."
            )
    else:
        # Linux/Mac: tesseract installed via apt/brew
        path = shutil.which("tesseract")
        if path is None:
            raise RuntimeError(
                "Tesseract not found. Install with: apt-get install tesseract-ocr"
            )
        # pytesseract defaults to the Windows path even on Linux — override it explicitly
        _set_tesseract_cmd(path)


def _set_tesseract_cmd(cmd: str) -> None:
    try:
        import unstructured_pytesseract
        unstructured_pytesseract.pytesseract.tesseract_cmd = cmd
    except ImportError:
        pass
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = cmd
    except ImportError:
        pass
