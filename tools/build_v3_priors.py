"""Build generic V3 priors from latest official cases 0001..0150 only.

The output contains aggregate distributions.  It deliberately excludes case IDs,
absolute coordinates and component keys so it cannot become a coordinate lookup.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import argparse, json, math, re, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.cluster.vq import kmeans2
from pcb.coordinates import target_bbox_to_opencv
from pcb.io import read_image, write_json
from pcb.vision import OCR

DESIGNATOR = re.compile(r"^(?:USB|IC|LED|SW|TP|CN|U|R|C|L|Q|D|J|P|Y|X|F)\d+[A-Za-z]?$", re.I)
SIMPLE = {"r", "c", "l", "d", "led", "crystal", "crystal_2pin", "crystal_3pin", "crystal_4pin", "switch"}


def source(name: str) -> str:
    for value in ("KiCad", "Altium Designer", "Datasheet", "jlc", "other"):
        if value.lower() in name.lower():
            return value
    return "other"


def q(values):
    a = np.asarray(values, float)
    return {"count": int(len(a)), "p10": float(np.percentile(a, 10)), "median": float(np.median(a)), "p90": float(np.percentile(a, 90))}


def clusters(rows):
    a = np.asarray(rows, float)
    if not len(a):
        return []
    k = min(6, max(1, int(round(math.sqrt(len(a) / 5)))))
    if len(a) < k:
        k = len(a)
    # Deterministic initialization from sorted, evenly-spaced observations.
    order = np.lexsort(tuple(a[:, i] for i in range(a.shape[1] - 1, -1, -1)))
    seeds = a[order[np.linspace(0, len(a) - 1, k).astype(int)]]
    centers, labels = kmeans2(a, seeds, minit="matrix", iter=30)
    result = []
    for i, center in enumerate(centers):
        count = int((labels == i).sum())
        if count:
            result.append({"count": count, "center": [round(float(x), 5) for x in center]})
    return sorted(result, key=lambda x: -x["count"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_root", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=ROOT / "pcb" / "v3_priors.json")
    args = ap.parse_args()
    if args.target_root.name != "200_train_cases":
        raise SystemExit("target_root must be exactly 200_train_cases")
    ocr = OCR(ROOT / "runs" / "ocr_cache")
    dims = defaultdict(lambda: defaultdict(list))
    anchor_rows = defaultdict(list)
    pin_rows = defaultdict(list)
    pin_name_counts = defaultdict(Counter)
    type_counts = Counter()
    exact_ocr = Counter()
    total_observable = Counter()
    for number in range(1, 151):
        folder = args.target_root / f"{number:04d}"
        image_path = next(folder.glob("*.png"))
        target_path = next(folder.glob("*_target.json"))
        image = read_image(image_path)
        height, width = image.shape[:2]
        src = source(image_path.name)
        gt = json.loads(target_path.read_text(encoding="utf-8"))
        texts = ocr.recognize(image)
        token_map = defaultdict(list)
        for t in texts:
            token_map[re.sub(r"\s+", "", t.text).upper()].append(t)
        for key, comp in gt["components"].items():
            typ = comp["type"]
            type_counts[typ] += 1
            box = target_bbox_to_opencv(comp["bbox"], height)
            bw, bh = box[2] - box[0], box[3] - box[1]
            dims[(src, typ)]["width"].append(bw)
            dims[(src, typ)]["height"].append(bh)
            dims[("ALL", typ)]["width"].append(bw)
            dims[("ALL", typ)]["height"].append(bh)
            if DESIGNATOR.fullmatch(key):
                total_observable[(src, typ)] += 1
                matches = token_map.get(re.sub(r"\s+", "", key).upper(), [])
                if matches:
                    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                    t = min(matches, key=lambda z: math.dist(z.center, (cx, cy)))
                    th = max(4.0, t.bbox[3] - t.bbox[1])
                    row = [(cx - t.center[0]) / th, (cy - t.center[1]) / th, bw / th, bh / th]
                    anchor_rows[(src, typ)].append(row)
                    anchor_rows[("ALL", typ)].append(row)
                    exact_ocr[(src, typ)] += 1
            pins = gt["pins"].get(key, {})
            if typ in SIMPLE and pins:
                for pin_key, pin in pins.items():
                    px = float(pin["point"]["x"])
                    py = height - float(pin["point"]["y"])
                    if bw > 0 and bh > 0:
                        pin_rows[typ].append([(px - box[0]) / bw, (py - box[1]) / bh])
                    pin_name_counts[typ][str(pin.get("pinname", ""))] += 1
    dim_out = {}
    for (src, typ), fields in dims.items():
        dim_out[f"{src}|{typ}"] = {name: q(values) for name, values in fields.items()}
    anchor_out = {f"{src}|{typ}": clusters(rows) for (src, typ), rows in anchor_rows.items() if rows}
    pin_out = {}
    for typ, rows in pin_rows.items():
        pin_out[typ] = {"clusters": clusters(rows), "pinname_counts": dict(pin_name_counts[typ])}
    out = {
        "contract": "latest official public set, development cases 0001..0150 only",
        "sealed_holdout_used_for_development": False,
        "contains_case_ids_or_absolute_coordinates": False,
        "type_counts": dict(type_counts),
        "dimensions": dim_out,
        "anchor_clusters": anchor_out,
        "exact_ocr_counts": {f"{s}|{t}": int(v) for (s, t), v in exact_ocr.items()},
        "observable_key_counts": {f"{s}|{t}": int(v) for (s, t), v in total_observable.items()},
        "simple_pin_layouts": pin_out,
    }
    write_json(args.output, out)
    print(json.dumps({"types": len(type_counts), "anchor_groups": len(anchor_out), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
