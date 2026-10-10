"""Read-only portability check. Does not install, download, or read datasets."""
from __future__ import annotations
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from pcb.runtime_paths import prepend_compatible_vendor_paths
    prepend_compatible_vendor_paths(ROOT)
    failures = []
    report = {"python": sys.version, "packages": {}, "assets": {}}
    if sys.version_info < (3, 11):
        failures.append("Python >= 3.11 required; Python 3.12 recommended")
    for package in ("torch", "ultralytics", "numpy", "opencv-python", "scipy", "scikit-image", "onnxruntime", "easyocr"):
        try:
            report["packages"][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            failures.append(f"Missing installed package: {package}")
    from pcb.pin.frozen_l39.stages import PIN_WEIGHT_SHA256, LIBRARY_SHA256
    expected = {"reports/pin_learned_locator_l1/best.pt": PIN_WEIGHT_SHA256,
                "assets/cpntLibrary.json": LIBRARY_SHA256,
                "models/component_yolo11n_continue_v2_best.pt": "51b4735d19315ba3ada49143e6f912aa414589f7961e140faab177dc6db7e776"}
    for relative, digest in expected.items():
        p = ROOT / relative
        actual = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
        report["assets"][relative] = actual == digest
        if actual != digest:
            failures.append(f"Missing/wrong asset {relative}; run git lfs pull")
    for relative in ("models/easyocr/english_g2.pth", "models/easyocr/craft_mlt_25k.pth"):
        p = ROOT / relative
        valid = p.is_file() and p.stat().st_size > 512
        report["assets"][relative] = valid
        if not valid:
            failures.append(f"Missing OCR model {relative}; run git lfs pull")
    try:
        import torch
        import cv2
        from pcb.core.registry import build_default_registry
        report["cuda_available"] = torch.cuda.is_available()
        report["opencv"] = cv2.__version__
        report["stages"] = build_default_registry().describe()
    except Exception as exc:
        failures.append(f"Import failure: {exc}")
    report["failures"] = failures
    report["ready"] = not failures
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
