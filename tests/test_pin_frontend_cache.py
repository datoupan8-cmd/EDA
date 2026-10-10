import json
from pathlib import Path
import tempfile
import unittest

from tools.pin_frontend_cache import canonical, development_manifest, digest_value, parse_case_ids, verify_scene_checksum


class FrontendWhitelistTests(unittest.TestCase):
    def test_ids_default_to_whitelist_and_reject_holdout_and_duplicates(self):
        self.assertEqual(parse_case_ids(None), [f"{n:04d}" for n in range(1, 151)])
        for invalid in ("151", "0", "1,1", "10_GTcase", "1,200", ""):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_case_ids(invalid)

    def test_manifest_checks_all_required_folders_without_opening_targets(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "200_train_cases"
            for key in parse_case_ids(None):
                folder = root / key
                folder.mkdir(parents=True)
                (folder / f"{key}.png").touch()
                (folder / f"{key}_target_named.json").touch()
            with patch.object(Path, "read_text", side_effect=AssertionError("Must not read GT")), \
                 patch.object(Path, "iterdir", side_effect=AssertionError("Do not enumerate dataset root")):
                manifest = development_manifest(root)
            self.assertEqual(list(manifest), parse_case_ids(None))
            (root / "0001/duplicate.png").touch()
            with self.assertRaises(ValueError):
                development_manifest(root)

    def test_signature_hash_is_stable_after_integer_key_json_normalization(self):
        raw = {"diagnostics": {"class_names": {0: "r", 2: "c", 10: "box"}}}
        normalized = json.loads(canonical(raw))
        cache = {"scene": normalized, "scene_sha256": digest_value(normalized)}
        self.assertEqual(verify_scene_checksum(json.loads(json.dumps(cache))), "json_canonical")
        cache["scene"]["diagnostics"]["class_names"]["0"] = "changed"
        with self.assertRaises(ValueError):
            verify_scene_checksum(cache)


if __name__ == "__main__":
    unittest.main()
