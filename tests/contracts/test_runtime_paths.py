from pathlib import Path
import sys
import unittest

from pcb.runtime_paths import compatible_vendor_paths


ROOT = Path(__file__).resolve().parents[2]


class RuntimePathTests(unittest.TestCase):
    def test_rapidocr_runtime_is_portable_relative_path(self):
        selected = compatible_vendor_paths(ROOT)
        self.assertIn(ROOT / "vendor" / "rapidocr_runtime", selected)

    def test_windows_cp314_easyocr_runtime_selection(self):
        selected = compatible_vendor_paths(ROOT)
        easyocr = ROOT / "vendor" / "ocr_runtime"
        expected = sys.platform == "win32" and sys.version_info[:2] == (3, 14)
        self.assertEqual(easyocr in selected, expected)


if __name__ == "__main__":
    unittest.main()
