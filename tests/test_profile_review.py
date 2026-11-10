from pathlib import Path
import sys
import tempfile
import unittest
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from review_detection_dataset import describe_image, read_boxes, write_boxes
from review_profile_dataset import prepare_review_copy


class ProfileReviewTests(unittest.TestCase):
    def make_dataset(self, root):
        (root/"portes/validation/labels").mkdir(parents=True)
        (root/"coco").mkdir()
        (root/"SOURCE.json").write_text("{}")
        for name in ("portes", "coco"):
            (root/name/"manifest.csv").write_text("split,image,label,reviewed\n")
        (root/"portes/validation/labels/001.txt").write_text("0 0.5 0.5 0.2 0.6\n")

    def test_new_and_old_manifest_titles(self):
        self.assertIn("015.png", describe_image({"video":"test.mp4","image":"validation/images/015.png"}))
        self.assertIn("frame 30", describe_image({"video":"test.mp4","frame":"30","image":"example.png"}))

    def test_review_copy_is_independent_and_resumes_without_overwriting(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder)/"source", Path(folder)/"review"
            self.make_dataset(source)
            self.assertTrue(prepare_review_copy(source, target))
            relative="portes/validation/labels/001.txt"
            (target/relative).write_text("")
            self.assertNotEqual((target/relative).read_text(), (source/relative).read_text())
            self.assertFalse(prepare_review_copy(source, target))
            self.assertEqual((target/relative).read_text(), "")

    def test_refuses_overwriting_existing_or_nested_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder)/"source", Path(folder)/"unknown"
            self.make_dataset(source)
            target.mkdir()
            for destination in (source, source/"nested", target):
                with self.assertRaises(ValueError): prepare_review_copy(source, destination)

    def test_shared_review_keeps_corrections_after_source_path_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target, moved = (Path(folder)/name for name in ("source", "review", "windows_source"))
            self.make_dataset(source)
            prepare_review_copy(source, target)
            label = "portes/validation/labels/001.txt"
            (target/label).write_text("")
            shutil.copytree(source, moved)
            self.assertFalse(prepare_review_copy(moved, target))
            self.assertEqual((target/label).read_text(), "")
            (moved/label).write_text("0 0.3 0.3 0.1 0.1\n")
            with self.assertRaises(ValueError): prepare_review_copy(moved, target)

    def test_saved_box_round_trip_in_camera_coordinates(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"label.txt"
            write_boxes(path, [[32, 24, 160, 216]], 320, 240)
            actual=read_boxes(path,320,240)[0]
            for a,b in zip(actual,[32,24,160,216]): self.assertAlmostEqual(a,b,places=3)


if __name__=="__main__": unittest.main()
