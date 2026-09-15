"""Developer-only inspection helper, hard guarded to public cases 0001..0150."""
from pathlib import Path
import argparse, json, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pcb.io import read_image
from pcb.vision import OCR, symbol_candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_root", type=Path, required=True)
    ap.add_argument("--case", type=int, required=True)
    args = ap.parse_args()
    if not 1 <= args.case <= 150:
        raise SystemExit("sealed holdout guard")
    folder = args.target_root / f"{args.case:04d}"
    image_path = next(folder.glob("*.png"))
    target_path = next(folder.glob("*_target.json"))
    image = read_image(image_path)
    texts = OCR(ROOT / "runs" / "ocr_cache_v3").recognize(image)
    print("image", image.shape, "ocr", len(texts))
    for i, t in enumerate(texts):
        print(i, repr(t.text), tuple(round(x, 1) for x in t.bbox), round(t.score, 3))
    candidates, _, _, colored = symbol_candidates(image, texts)
    print("candidates", len(candidates), "colored", colored)
    for i, c in enumerate(candidates):
        print(i, c["bbox"], c["area"])
    gt = json.loads(target_path.read_text(encoding="utf-8"))
    print("GT components")
    for key, value in gt["components"].items():
        print(key, value)
    print("GT pins")
    for key, value in gt["pins"].items():
        print(key, value)


if __name__ == "__main__":
    main()
