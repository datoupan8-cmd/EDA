"""Export a tiled YOLO dataset from latest official cases 0001-0150 only."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pcb.coordinates import target_bbox_to_opencv
from pcb.data_policy import ALLOWED_CASE_IDS, discover_allowed_cases


def tile_origins(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    values = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if values[-1] != last:
        values.append(last)
    return values


def keep_negative_tile(name: str, ratio: float) -> bool:
    value = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big") / 2**64
    return ratio >= 1 or (ratio > 0 and value < ratio)


def export_dataset(dataset_root: Path, output_dir: Path, tile_size: int, overlap: int, minimum_visible_fraction: float, negative_tile_ratio: float):
    cases = discover_allowed_cases(dataset_root)
    loaded = []
    type_counts: Counter[str] = Counter()
    for case in cases:
        image = Image.open(case.image_path).convert("RGB")
        target = json.loads(case.target_path.read_text(encoding="utf-8"))
        components = []
        for key, row in target["components"].items():
            bbox = target_bbox_to_opencv(row["bbox"], image.height)
            components.append({"key": key, "type": row["type"], "bbox": bbox})
            type_counts[row["type"]] += 1
        loaded.append((case, image, components))
    class_names = [name for name, _ in sorted(type_counts.items(), key=lambda item: (-item[1], item[0]))]
    class_ids = {name: index for index, name in enumerate(class_names)}
    tile_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    for split in ("train", "dev"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    for case, image, components in loaded:
        for top in tile_origins(image.height, tile_size, overlap):
            for left in tile_origins(image.width, tile_size, overlap):
                tw, th = min(tile_size, image.width - left), min(tile_size, image.height - top)
                labels = []
                for component in components:
                    x1, y1, x2, y2 = component["bbox"]
                    original = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                    ix1, iy1 = max(0.0, x1 - left), max(0.0, y1 - top)
                    ix2, iy2 = min(float(tw), x2 - left), min(float(th), y2 - top)
                    visible = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                    if original <= 0 or visible / original < minimum_visible_fraction:
                        continue
                    labels.append(f"{class_ids[component['type']]} {(ix1+ix2)/(2*tw):.8f} {(iy1+iy2)/(2*th):.8f} {(ix2-ix1)/tw:.8f} {(iy2-iy1)/th:.8f}")
                tile_name = f"case_{case.case_id:04d}_x{left}_y{top}"
                if not labels and not keep_negative_tile(tile_name, negative_tile_ratio):
                    continue
                image.crop((left, top, left + tw, top + th)).save(output_dir / "images" / case.split / f"{tile_name}.png", optimize=True)
                (output_dir / "labels" / case.split / f"{tile_name}.txt").write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
                tile_counts[case.split] += 1
                label_counts[case.split] += len(labels)
        image.close()
    train_ids = [case.case_id for case in cases if case.split == "train"]
    dev_ids = [case.case_id for case in cases if case.split == "dev"]
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy": {"allowed_case_range": [1, 150], "reserved_cases_accessed": False},
        "split": {"rule": "case_id % 5 == 0 -> dev", "train_case_ids": train_ids, "dev_case_ids": dev_ids},
        "tiling": {"tile_size": tile_size, "overlap": overlap, "minimum_visible_fraction": minimum_visible_fraction, "negative_tile_ratio": negative_tile_ratio},
        "counts": {"source_cases": len(cases), "source_components": sum(type_counts.values()), "tiles": dict(tile_counts), "exported_labels": dict(label_counts)},
        "class_names": class_names,
    }
    output_dir.joinpath("manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    yaml = [f"path: {json.dumps(output_dir.resolve().as_posix(), ensure_ascii=False)}", "train: images/train", "val: images/dev", "names:"]
    yaml.extend(f"  {index}: {json.dumps(name, ensure_ascii=False)}" for index, name in enumerate(class_names))
    output_dir.joinpath("dataset.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "components_dataset")
    parser.add_argument("--tile-size", type=int, default=1280)
    parser.add_argument("--overlap", type=int, default=192)
    parser.add_argument("--minimum-visible-fraction", type=float, default=.8)
    parser.add_argument("--negative-tile-ratio", type=float, default=.08)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {args.output_dir}")
        resolved, safe_parent = args.output_dir.resolve(), (ROOT / "artifacts").resolve()
        if resolved == safe_parent or safe_parent not in resolved.parents:
            raise ValueError(f"Unsafe output path: {resolved}")
        shutil.rmtree(resolved)
    manifest = export_dataset(args.dataset_root, args.output_dir, args.tile_size, args.overlap, args.minimum_visible_fraction, args.negative_tile_ratio)
    assert set(manifest["split"]["train_case_ids"] + manifest["split"]["dev_case_ids"]) == set(ALLOWED_CASE_IDS)
    print(json.dumps({"output": str(args.output_dir), **manifest["counts"], "sealed_holdout_used": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
