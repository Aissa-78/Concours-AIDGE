from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from check_setup import check_modules, missing_data


class SetupCheckTests(unittest.TestCase):
    def test_import_errors_are_reported_without_installing(self):
        with patch("check_setup.import_module", side_effect=[object(), ImportError("absent")]):
            with patch("builtins.print"):
                self.assertEqual(check_modules(["good", "missing"]), ["missing"])

    def test_private_data_is_separate_from_a_fresh_clone(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            absent = missing_data(root)
            self.assertEqual(len(absent), 3)
            for name in absent:
                path=root/name
                path.parent.mkdir(parents=True,exist_ok=True)
                path.touch()
            self.assertEqual(missing_data(root), [])


if __name__ == "__main__": unittest.main()
