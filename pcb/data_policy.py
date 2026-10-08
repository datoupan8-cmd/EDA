"""Hard data boundary for Component development and optional detector training."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


FIRST_ALLOWED_CASE_ID = 1
LAST_ALLOWED_CASE_ID = 150
ALLOWED_CASE_IDS = tuple(range(FIRST_ALLOWED_CASE_ID, LAST_ALLOWED_CASE_ID + 1))


@dataclass(frozen=True)
class CaseFiles:
    case_id: int
    case_dir: Path
    image_path: Path
    target_path: Path
    split: str


def assert_allowed_case_id(case_id: int) -> None:
    if not FIRST_ALLOWED_CASE_ID <= int(case_id) <= LAST_ALLOWED_CASE_ID:
        raise PermissionError(f"Case {int(case_id):04d} is sealed; only 0001-0150 are allowed")


def split_for_case(case_id: int) -> str:
    assert_allowed_case_id(case_id)
    return "dev" if int(case_id) % 5 == 0 else "train"


def assert_image_path_allowed(image_path: str | Path) -> None:
    parts = Path(image_path).resolve().parts
    lowered = [part.lower() for part in parts]
    if "10_gtcase" in lowered:
        raise PermissionError("10_GTcase is sealed")
    if "200_train_cases" in lowered:
        index = lowered.index("200_train_cases")
        if index + 1 >= len(parts):
            raise ValueError(f"Cannot locate case id in {image_path}")
        match = re.match(r"(\d+)", parts[index + 1])
        if match is None:
            raise ValueError(f"Cannot parse case id from {parts[index + 1]!r}")
        assert_allowed_case_id(int(match.group(1)))


def discover_allowed_cases(dataset_root: str | Path) -> list[CaseFiles]:
    """Open exactly 0001-0150 without listing the parent or sealed folders."""
    root = Path(dataset_root)
    if root.name != "200_train_cases":
        raise ValueError("dataset_root must be the 200_train_cases directory")
    records = []
    for case_id in ALLOWED_CASE_IDS:
        case_dir = root / f"{case_id:04d}"
        images = sorted(case_dir.glob("*.png"))
        targets = sorted(case_dir.glob("*_target*.json"))
        if len(images) != 1 or len(targets) != 1:
            raise ValueError(f"{case_id:04d}: expected one PNG and one target JSON")
        records.append(CaseFiles(case_id, case_dir, images[0], targets[0], split_for_case(case_id)))
    return records
