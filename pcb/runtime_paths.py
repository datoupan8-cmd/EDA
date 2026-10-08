"""Select bundled dependency paths without binding the project to one PC.

The bundled EasyOCR runtime contains a Windows CPython 3.14 extension.  It is
used only on a compatible interpreter; other environments use the packages
installed from ``requirements.txt``.  This changes no algorithm or parameter
on the verified Windows/Python 3.14 baseline.
"""
from __future__ import annotations

from pathlib import Path
import sys


def _binary_runtime_is_compatible(path: Path) -> bool:
    binaries = list(path.rglob("*.pyd"))
    if not binaries:
        return True
    if sys.platform != "win32":
        return False
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    return all(tag in binary.name for binary in binaries)


def compatible_vendor_paths(project_root: Path) -> tuple[Path, ...]:
    """Return existing bundled runtimes in the original import precedence."""
    vendor = project_root / "vendor"
    candidates = [vendor / "python"]
    easyocr_runtime = vendor / "ocr_runtime"
    if easyocr_runtime.is_dir() and _binary_runtime_is_compatible(easyocr_runtime):
        candidates.append(easyocr_runtime)
    candidates.append(vendor / "rapidocr_runtime")
    return tuple(path for path in candidates if path.is_dir())


def prepend_compatible_vendor_paths(project_root: Path) -> tuple[Path, ...]:
    """Prepend bundled runtimes exactly once and return the selected paths."""
    selected = compatible_vendor_paths(project_root)
    for path in selected:
        value = str(path)
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)
    return selected
