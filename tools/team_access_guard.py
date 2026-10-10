"""Unchanged inference/GT access guard extracted from the Pin runner."""
from contextlib import contextmanager
from pathlib import Path
import sys

class Guard:
    """Disallow inference GT and archived intermediates; only evaluator opts in."""
    def __init__(self, install=True):
        self.phase = "idle"
        self.allowed_target = None
        self.target_opens = self.blocked = 0
        if install:
            sys.addaudithook(self.audit)

    def audit(self, event, args):
        if event != "open" or not args or not isinstance(args[0], (str, bytes, Path)):
            return
        value = str(args[0]).replace("\\", "/").lower()
        parts = value.split("/")
        if any(p in {"golden", "sealed", "eda_pin_crossing_quicktest"} or p.startswith(("10gt", "10_gtcase")) for p in parts):
            self.blocked += 1
            raise PermissionError("Reserved/QuickTest data prohibited")
        if "200_train_cases" in parts:
            pos = parts.index("200_train_cases")
            if pos + 1 < len(parts):
                cid = parts[pos + 1]
                if not cid.isdigit() or not 1 <= int(cid) <= 150:
                    self.blocked += 1
                    raise PermissionError("Only development cases 0001..0150")
        if value.endswith(".json") and "_target" in Path(value).name:
            if value != self.allowed_target or self.phase != "evaluation":
                self.blocked += 1
                raise PermissionError("GT may only be read after saved predictions")
            self.target_opens += 1
        mode = args[1] if len(args) > 1 else None
        writing = isinstance(mode, str) and any(c in mode for c in "wax+")
        archived = ("/tmp/pin_sina_frontend/" in value or "/predictions/" in value
                    or "/words/" in value or value.endswith("raw_terminals.json")
                    or ("/reports/" in value and value.endswith("/complete.json")))
        if self.phase == "inference" and archived and not writing:
            self.blocked += 1
            raise PermissionError("Image inference cannot consume archived stage outputs")

    @contextmanager
    def inferring(self):
        self.phase = "inference"
        try:
            yield
        finally:
            self.phase = "idle"

    @contextmanager
    def evaluating(self, path, saved):
        if not all(Path(p).is_file() for p in saved):
            raise AssertionError("Save both PNG-only predictions first")
        self.allowed_target = str(Path(path).resolve()).replace("\\", "/").lower()
        self.phase = "evaluation"
        try:
            yield
        finally:
            self.phase, self.allowed_target = "idle", None
