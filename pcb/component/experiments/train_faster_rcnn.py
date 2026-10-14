"""Train a new two-stage detector using only the fixed development training split."""
from __future__ import annotations
from collections import Counter
from contextlib import contextmanager
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
TRAIN_IDS = tuple(f"{n:04d}" for n in range(1, 151) if n % 5)
CHECK_IDS = tuple(f"{n:04d}" for n in range(5, 151, 5))


class TrainingGuard:
    def __init__(self, dataset):
        self.dataset = dataset.resolve()
        self.allowed = None
        self.target_opens = 0
        sys.addaudithook(self.audit)

    def audit(self, event, args):
        if event != "open" or not args or not isinstance(args[0], (str, bytes, Path)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        parts = [part.lower() for part in path.parts]
        if any(part in {"golden", "sealed", "eda_pin_crossing_quicktest"}
               or part.startswith(("10gt", "10_gtcase")) for part in parts):
            raise PermissionError("Reserved data is prohibited during training")
        if "200_train_cases" in parts:
            if not path.is_relative_to(self.dataset):
                raise PermissionError("Unexpected training dataset")
            relative = path.relative_to(self.dataset).parts
            if not relative or relative[0] not in TRAIN_IDS:
                raise PermissionError("Only the fixed 120 training cases may be opened")
            mode = args[1] if len(args) > 1 else None
            if isinstance(mode, str) and any(c in mode for c in "wax+"):
                raise PermissionError("No dataset writes")
        if path.suffix.lower() == ".json" and "_target" in path.name.lower():
            if path != self.allowed:
                raise PermissionError("Training annotation access requires an explicit allowlist entry")
            self.target_opens += 1

    @contextmanager
    def annotation(self, path):
        self.allowed = path.resolve()
        try:
            yield
        finally:
            self.allowed = None


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--samples-per-case", type=int, default=2)
    args = parser.parse_args()
    dataset, output = args.dataset.resolve(), args.output.resolve()
    if dataset.name != "200_train_cases" or not dataset.is_dir():
        raise ValueError("Require canonical development dataset root")
    if output.exists() or output.is_relative_to(dataset):
        raise ValueError("Use a fresh output outside the dataset")
    if args.epochs <= 0 or args.samples_per_case <= 0:
        raise ValueError("Training schedule must be positive")
    # Check all 150 expected canonical directories without opening check-set data.
    for number in range(1, 151):
        folder = dataset / f"{number:04d}"
        if not folder.is_dir() or folder.resolve() != folder:
            raise ValueError(f"Missing or redirected development directory {number:04d}")
    output.mkdir(parents=True)
    guard = TrainingGuard(dataset)
    from pcb.runtime_paths import prepend_compatible_vendor_paths
    prepend_compatible_vendor_paths(ROOT)
    import cv2
    import numpy as np
    import torch
    from pcb.component.alternative_rcnn import build_rcnn, CLASSES
    from pcb.io import read_image
    random.seed(20261013)
    np.random.seed(20261013)
    torch.manual_seed(20261013)
    torch.cuda.manual_seed_all(20261013)
    torch.set_num_threads(4)
    records, histogram = [], Counter()
    for cid in TRAIN_IDS:
        folder = dataset / cid
        images, targets = list(folder.glob("*.png")), list(folder.glob("*_target*.json"))
        if len(images) != 1 or len(targets) != 1:
            raise ValueError(f"{cid}: require unique image and annotation")
        if images[0].resolve().parent != folder or targets[0].resolve().parent != folder:
            raise PermissionError("Input path redirects outside the canonical case")
        image = read_image(images[0])
        height, width = image.shape[:2]
        with guard.annotation(targets[0]):
            target = json.loads(targets[0].read_text(encoding="utf-8"))
        boxes, labels = [], []
        for component in target["components"].values():
            x1, y1, x2, y2 = component["bbox"]
            box = [max(0., float(x1)), max(0., height - float(y2)),
                   min(float(width), float(x2)), min(float(height), height - float(y1))]
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f"{cid}: invalid target bbox")
            label = CLASSES.index(component["type"])
            boxes.append(box)
            labels.append(label)
            histogram[component["type"]] += 1
        if not boxes:
            raise ValueError(f"{cid}: missing training objects")
        records.append({"case_id": cid, "image": str(images[0]), "boxes": boxes, "labels": labels})
    init = ROOT / "models/component_fasterrcnn/coco_init.pth"
    digest = hashlib.sha256(init.read_bytes()).hexdigest()
    if not digest.startswith("258fb6c6"):
        raise ValueError("Public pretrained weight checksum prefix mismatch")
    source_files = [Path(__file__).resolve(), ROOT / "pcb/component/alternative_rcnn.py"]
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    manifest = {"ACTUAL_CASE_IDS": TRAIN_IDS, "development_check_ids": CHECK_IDS,
        "sealed_holdout_used": False, "training_case_count": len(records), "class_counts": dict(histogram),
        "classes": CLASSES, "epochs": args.epochs, "samples_per_case": args.samples_per_case,
        "seed": 20261013, "pretrained_source": "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
        "pretrained_sha256": digest, "source_sha256": source_hashes,
        "training_target_opens": guard.target_opens, "no_check_annotations_used_in_training": True}
    write(output / "manifest.json", manifest)
    print(json.dumps({"event": "start", **manifest}, ensure_ascii=False), flush=True)

    class CropDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.epoch = 0

        def __len__(self):
            return len(records) * args.samples_per_case

        def __getitem__(self, index):
            record = records[index % len(records)]
            rng = random.Random(20261013 + self.epoch * len(self) + index)
            image = read_image(record["image"])
            height, width = image.shape[:2]
            weights = [1 / max(1, histogram[CLASSES[label]]) ** .5 for label in record["labels"]]
            anchor = record["boxes"][rng.choices(range(len(weights)), weights=weights)[0]]
            size = int(max(768, anchor[2] - anchor[0] + 64, anchor[3] - anchor[1] + 64))
            cx = (anchor[0] + anchor[2]) / 2 + rng.uniform(-size / 4, size / 4)
            cy = (anchor[1] + anchor[3]) / 2 + rng.uniform(-size / 4, size / 4)
            left = int(max(0, min(width - size, cx - size / 2)))
            top = int(max(0, min(height - size, cy - size / 2)))
            right, bottom = min(width, left + size), min(height, top + size)
            boxes, labels = [], []
            for box, label in zip(record["boxes"], record["labels"]):
                clipped = [max(left, box[0]), max(top, box[1]), min(right, box[2]), min(bottom, box[3])]
                area = max(0, clipped[2] - clipped[0]) * max(0, clipped[3] - clipped[1])
                if area < .75 * (box[2] - box[0]) * (box[3] - box[1]):
                    continue
                boxes.append([clipped[0] - left, clipped[1] - top, clipped[2] - left, clipped[3] - top])
                labels.append(label)
            patch = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(patch.transpose(2, 0, 1).copy()).float().div_(255)
            return tensor, {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                            "labels": torch.tensor(labels, dtype=torch.int64)}

    model = build_rcnn(pretrained=init).to("cuda")
    model.train()
    crops = CropDataset()
    loader = torch.utils.data.DataLoader(crops, batch_size=1, shuffle=True, num_workers=0,
                                        collate_fn=lambda batch: tuple(zip(*batch)))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.0002, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=.00002)
    scaler = torch.amp.GradScaler("cuda")
    history, started = [], time.perf_counter()
    for epoch in range(args.epochs):
        crops.epoch = epoch
        totals = Counter()
        for step, (images, targets) in enumerate(loader, 1):
            images = [image.to("cuda") for image in images]
            targets = [{key: value.to("cuda") for key, value in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda"):
                losses = model(images, targets)
                loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss; training stops")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            scaler.step(optimizer)
            scaler.update()
            for key, value in losses.items():
                totals[key] += float(value.detach())
            if step % 40 == 0:
                progress = {"event": "training", "epoch": epoch + 1, "epochs": args.epochs,
                            "step": step, "steps": len(loader), "loss": float(loss.detach()),
                            "seconds": time.perf_counter() - started}
                print(json.dumps(progress), flush=True)
                write(output / "progress.json", progress)
        scheduler.step()
        history.append({"epoch": epoch + 1, "losses": {k: v / len(loader) for k, v in totals.items()}})
        write(output / "history.json", history)
        checkpoint = {"model": {k: v.cpu() for k, v in model.state_dict().items()},
                      "classes": list(CLASSES), "training_case_ids": list(TRAIN_IDS), "epoch": epoch + 1}
        torch.save(checkpoint, output / "last.pth")
        print(json.dumps({"event": "epoch_complete", **history[-1]}), flush=True)
    if source_hashes != {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}:
        raise ValueError("Frozen training algorithm changed during the run")
    manifest.update({"seconds": time.perf_counter() - started, "history": history,
                     "checkpoint_sha256": hashlib.sha256((output / "last.pth").read_bytes()).hexdigest(),
                     "training_complete": True})
    write(output / "report.json", manifest)
    print(json.dumps({"event": "complete", "seconds": manifest["seconds"],
                      "checkpoint": str(output / "last.pth")}), flush=True)


if __name__ == "__main__":
    main()
