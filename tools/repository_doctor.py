"""Check whether a clone is complete and suitable for reproducible runs."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"
MACHINE_PATH = re.compile(r"[A-Za-z]:\\Users\\[^\\\r\n]+")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_result(rows: list[dict[str, object]], check: str, ok: bool, detail: str) -> None:
    rows.append({"check": check, "ok": ok, "detail": detail})
    print(f"[{'OK' if ok else 'FAIL'}] {check}: {detail}")


def source_files() -> list[Path]:
    roots = [ROOT / "pcb", ROOT / "tools", ROOT / "configs", ROOT / ".github"]
    files = [ROOT / "main.py", ROOT / "README.md", ROOT / "CONTRIBUTING.md"]
    for root in roots:
        if root.exists():
            files.extend(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".py", ".json", ".yml", ".yaml", ".md"})
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-imports", action="store_true", help="Only check repository files and hashes")
    parser.add_argument("--skip-hashes", action="store_true", help="Skip full model SHA256 calculation")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    rows: list[dict[str, object]] = []

    version_ok = (3, 12) <= sys.version_info[:2] <= (3, 14)
    add_result(rows, "python", version_ok, sys.version.split()[0])

    required = [
        ROOT / "main.py",
        ROOT / "configs" / "current.json",
        ROOT / "models" / "SHA256SUMS.json",
        ROOT / "pcb" / "core" / "pipeline.py",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    add_result(rows, "required_files", not missing, "complete" if not missing else f"missing={missing}")

    hash_manifest = json.loads((ROOT / "models" / "SHA256SUMS.json").read_text(encoding="utf-8"))
    for relative, expected in hash_manifest.items():
        path = ROOT / relative
        if not path.exists():
            add_result(rows, f"artifact:{relative}", False, "missing; run git lfs pull")
            continue
        if path.read_bytes()[: len(LFS_HEADER)] == LFS_HEADER:
            add_result(rows, f"artifact:{relative}", False, "Git LFS pointer only; run git lfs pull")
            continue
        if args.skip_hashes:
            add_result(rows, f"artifact:{relative}", True, f"present ({path.stat().st_size} bytes), hash skipped")
        else:
            actual = sha256(path)
            add_result(rows, f"artifact:{relative}", actual == expected, actual)

    absolute_hits: list[str] = []
    for path in source_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if MACHINE_PATH.search(text):
            absolute_hits.append(str(path.relative_to(ROOT)))
    add_result(rows, "machine_specific_paths", not absolute_hits, "none" if not absolute_hits else str(absolute_hits))

    if not args.skip_imports:
        from pcb.runtime_paths import prepend_compatible_vendor_paths

        selected = prepend_compatible_vendor_paths(ROOT)
        add_result(rows, "vendor_runtime_selection", True, ", ".join(str(path.relative_to(ROOT)) for path in selected) or "system packages")
        packages = {
            "numpy": "numpy",
            "cv2": "opencv-python",
            "scipy": "scipy",
            "skimage": "scikit-image",
            "onnxruntime": "onnxruntime",
            "PIL": "Pillow",
            "torch": "torch",
            "ultralytics": "ultralytics",
            "easyocr": "easyocr",
            "rapidocr_onnxruntime": "rapidocr-onnxruntime",
        }
        for module_name, distribution in packages.items():
            try:
                importlib.import_module(module_name)
                try:
                    version = importlib.metadata.version(distribution)
                except importlib.metadata.PackageNotFoundError:
                    version = "bundled"
                add_result(rows, f"import:{module_name}", True, version)
            except Exception as exc:  # environment diagnostic, preserve full exception type
                add_result(rows, f"import:{module_name}", False, repr(exc))

        try:
            from pcb.core.config import PipelineConfig
            from pcb.core.registry import build_default_registry

            config = PipelineConfig.load(ROOT / "configs" / "current.json")
            registry = build_default_registry()
            selected = config.as_dict()
            unknown = {kind: version for kind, version in selected.items() if kind in registry.describe() and version not in registry.versions(kind)}
            add_result(rows, "stage_registry", not unknown, json.dumps(registry.describe(), ensure_ascii=False))
        except Exception as exc:
            add_result(rows, "stage_registry", False, repr(exc))

    report = {
        "project_root": ".",
        "python": sys.version,
        "ok": all(bool(row["ok"]) for row in rows),
        "checks": rows,
    }
    if args.output:
        path = args.output if args.output.is_absolute() else ROOT / args.output
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRepository ready: {report['ok']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
