"""Create a reproducible, curated V4.1 delivery ZIP without transient caches."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT.parent / f"{ROOT.name}.zip"
FULL_DIRS = ("pcb", "tests", "tools", "models", "reports", "debug", "splits")
ROOT_SUFFIXES = {".py", ".md", ".txt", ".yml", ".yaml", ".json", ".docx"}
ROOT_NAMES = {"Dockerfile", ".dockerignore", ".gitignore"}
SKIP_PARTS = {"__pycache__", "docx_render"}
SKIP_PREFIXES = ("reports/smoke_",)
SKIP_NAMES = {"delivery_verification.json", "package_result.txt"}


def include(path: Path) -> bool:
    relative = path.relative_to(ROOT).as_posix()
    parts = relative.split("/")
    if path.name in SKIP_NAMES:
        return False
    if any(part in SKIP_PARTS or part.startswith("docx_render") for part in parts) or path.suffix == ".pyc":
        return False
    if any(relative.startswith(prefix) for prefix in SKIP_PREFIXES):
        return False
    if len(parts) == 1:
        return path.name in ROOT_NAMES or path.suffix.lower() in ROOT_SUFFIXES
    if parts[0] == "vendor":
        return len(parts) >= 2 and parts[1] in {"ocr_runtime", "rapidocr_runtime"}
    if parts[0] in FULL_DIRS:
        return True
    if parts[0] == "runs":
        if len(parts) >= 2 and parts[1] in {"v4_1_hybrid_rapid144_best_150", "final_smoke_v4_1"}:
            return True
        return False
    return False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    candidates = [path for path in ROOT.iterdir() if path.is_file()]
    for directory in FULL_DIRS:
        candidates.extend((ROOT / directory).rglob("*"))
    for directory in (ROOT / "vendor" / "ocr_runtime", ROOT / "vendor" / "rapidocr_runtime"):
        candidates.extend(directory.rglob("*"))
    candidates.extend((ROOT / "runs").rglob("*"))
    files = sorted({path for path in candidates if path.is_file() and include(path)})
    manifest_path = ROOT / "reports" / "package_manifest.json"
    payload = {
        "project": ROOT.name,
        "policy": "source+reports+debug+V4.1 BEST 150+single smoke; transient caches and repeated ablation runs excluded",
        "sealed_holdout_used": False,
        "model_sha256": sha256(ROOT / "models" / "component_yolo11n_continue_v2_best.pt"),
        "file_count_before_manifest": len(files),
        "uncompressed_bytes_before_manifest": sum(path.stat().st_size for path in files),
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if manifest_path not in files:
        files.append(manifest_path)
        files.sort()
    with zipfile.ZipFile(DESTINATION, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            relative = path.relative_to(ROOT.parent)
            archive.write(path, relative.as_posix())
    print(json.dumps({
        "zip": str(DESTINATION),
        "files": len(files),
        "uncompressed_bytes": sum(path.stat().st_size for path in files),
        "zip_bytes": DESTINATION.stat().st_size,
        "zip_sha256": sha256(DESTINATION),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
