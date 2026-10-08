"""Build the refactor ZIP without transient caches or Python bytecode."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import zipfile


EXCLUDED_PARTS = {"__pycache__", "refactor_runtime_cache", "refactor_parity_real"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix == ".pyc":
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.resolve() == output:
            continue
        files.append((path, relative))
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path, relative in sorted(files, key=lambda row: row[1].as_posix()):
            archive.write(path, (Path(root.name) / relative).as_posix())
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(output)
    print(f"files={len(files)}")
    print(f"bytes={output.stat().st_size}")
    print(f"sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
