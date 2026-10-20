"""Pin-owned opt-in CLI that delegates to the unchanged public team runner."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools import run_team_pipeline as runner
from pcb.pin.registry import build_l45_registry


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not any(a == "--config" or a.startswith("--config=") for a in args):
        args = ["--config", "configs/pin/team_l45.json", *args]
    original = runner.build_default_registry
    try:
        runner.build_default_registry = build_l45_registry
        return runner.main(args)
    finally:
        runner.build_default_registry = original


if __name__ == "__main__":
    raise SystemExit(main())
