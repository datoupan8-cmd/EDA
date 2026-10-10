"""Run the maintained, dataset-free module/contract tests from any directory."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pcb.runtime_paths import prepend_compatible_vendor_paths
prepend_compatible_vendor_paths(ROOT)


def main():
    tmp = ROOT / "runs/test_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(tmp)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for pattern in ("test_component_v4.py", "test_tiled_easyocr.py", "test_pin_detection_v4.py",
                    "test_pin_semantics_v4.py", "test_benchmark_pins_v4.py",
                    "test_pin_frontend_cache.py", "test_evaluate_diagnostic.py"):
        suite.addTests(loader.discover(str(ROOT / "tests"), pattern=pattern))
    suite.addTests(loader.discover(str(ROOT / "tests/contracts")))
    suite.addTests(loader.discover(str(ROOT / "tests/parity")))
    suite.addTests(loader.discover(str(ROOT / "tests/pin")))
    for directory in ("component", "wire", "topology"):
        path = ROOT / "tests" / directory
        if path.is_dir():
            suite.addTests(loader.discover(str(path)))
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
