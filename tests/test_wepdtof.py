"""Tests hors ligne : téléchargement partiel et conversion des annotations."""

import io
import sys
from pathlib import Path
import unittest
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from wepdtof_archive import RangeArchive
from prepare_wepdtof_subset import crop_annotations, fixed_crops, rotated_polygon, scene_group, select_scene


class Response:
    def __init__(self, content, start, end, size, status=206):
        self.status_code = status
        self.headers = {"Content-Range": f"bytes {start}-{end}/{size}", "ETag": '"stable"'}
        self.raw = io.BytesIO(content)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.raw.close()


class Server:
    def __init__(self, data):
        self.data = data
        self.ignore_ranges = False
        self.requests = []

    def get(self, url, headers, **kwargs):
        start, end = map(int, headers["Range"].removeprefix("bytes=").split("-"))
        self.requests.append((start, end))
        return Response(self.data[start:end+1], start, end, len(self.data), 200 if self.ignore_ranges else 206)


class RemoteZipTests(unittest.TestCase):
    def test_real_zip_reader_with_partial_http(self):
        content = io.BytesIO()
        with zipfile.ZipFile(content, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("example.txt", "bonjour")
        server = Server(content.getvalue())
        with RangeArchive("https://example.invalid/file.zip", session=server) as remote:
            with zipfile.ZipFile(remote) as archive:
                self.assertEqual(archive.read("example.txt"), b"bonjour")

    def test_refuses_server_returning_whole_archive(self):
        server = Server(b"abcdef")
        with RangeArchive("https://example.invalid/file.zip", session=server) as remote:
            server.ignore_ranges = True
            with self.assertRaises(ValueError):
                remote.read(2)

    def test_budget_and_seek(self):
        server = Server(b"0123456789")
        with RangeArchive("https://example.invalid/file.zip", max_bytes=4, session=server) as remote:
            remote.seek(-3, io.SEEK_END)
            self.assertEqual(remote.read(2), b"78")
            self.assertEqual(remote.tell(), 9)
            remote.seek(0)
            with self.assertRaises(ValueError):
                remote.read(3)


class AnnotationTests(unittest.TestCase):
    @staticmethod
    def scene_data(count=8):
        return {"images": [{"id": f"frame_{i:03d}", "file_name": f"frame_{i:03d}.jpg", "width": 100, "height": 100}
                            for i in range(count)],
                "annotations": [{"image_id": f"frame_{i:03d}", "category_id": 1,
                                 "bbox": [50, 50, 20, 40, 0]} for i in range(count)]}

    def test_positive_only_scene_does_not_add_negative(self):
        selected = select_scene(self.scene_data(), np.full((100, 100), 255, np.uint8),
                                2, np.random.default_rng(3), min_gap=1, negative_fraction=0)
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(item[3][0] for item in selected))

    def test_existing_frame_and_neighbors_are_excluded(self):
        selected = select_scene(self.scene_data(), np.full((100, 100), 255, np.uint8),
                                1, np.random.default_rng(3), min_gap=2, negative_fraction=0,
                                exclude_image_ids={"frame_003"})
        self.assertGreaterEqual(abs(selected[0][0] - 3), 2)

    def test_multiple_fixed_scales(self):
        crops = list(fixed_crops(100, 100, np.full((100, 100), 255, np.uint8), (.35, .5, .65)))
        self.assertEqual({x2-x1 for x1, _, x2, _ in crops}, {35, 50, 65})

    @staticmethod
    def annotation(box):
        return {"bbox": box, "category_id": 1, "iscrowd": 0}

    def test_axis_aligned_rectangle(self):
        result = crop_annotations([self.annotation([50, 50, 20, 40, 0])], (0, 0, 100, 100))
        np.testing.assert_allclose(result[0], [(0.5, 0.5, 0.2, 0.4)])

    def test_rotated_rectangle_swaps_dimensions(self):
        result = crop_annotations([self.annotation([50, 50, 20, 40, 90])], (0, 0, 100, 100))
        np.testing.assert_allclose(result[0], [(0.5, 0.5, 0.4, 0.2)])
        self.assertEqual(rotated_polygon([50, 50, 20, 40, 45]).shape, (4, 2))

    def test_partial_person_rejects_entire_crop(self):
        result = crop_annotations([self.annotation([99, 50, 20, 40, 0])], (0, 0, 100, 100))
        self.assertIsNone(result)

    def test_tiny_person_does_not_become_background(self):
        result = crop_annotations([self.annotation([50, 50, 3, 5, 0])], (0, 0, 100, 100))
        self.assertIsNone(result)

    def test_no_person_is_a_real_negative(self):
        self.assertEqual(crop_annotations([], (0, 0, 100, 100)), ([], []))

    def test_crops_never_include_unannotated_roi(self):
        roi = np.full((100, 100), 255, np.uint8)
        roi[:, :40] = 0
        crops = list(fixed_crops(100, 100, roi))
        self.assertTrue(crops)
        self.assertTrue(all(x1 >= 40 for x1, _, _, _ in crops))

    def test_related_clips_share_scene_group(self):
        self.assertEqual(scene_group("jewelry_store"), scene_group("jewelry_store_2"))


if __name__ == "__main__":
    unittest.main()
