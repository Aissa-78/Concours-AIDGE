from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import tempfile

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from profile_detector import ProfileDetector
from profile_training_data import augment_profile, WIDTH, HEIGHT, STRIDE
from person_detection_math import targets_from_boxes, evaluate_boxes
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from prepare_profile_mix import local_video,video_name


class ProfileDetectorTests(unittest.TestCase):
    def test_camera_geometry_preserves_checkpoint_shapes_and_decoding(self):
        old = ProfileDetector()
        camera = ProfileDetector(output_stride=8).eval()
        camera.load_state_dict(old.state_dict(), strict=True)
        with torch.no_grad():
            output = camera(torch.zeros(2, 1, 120, 160))
        self.assertEqual(tuple(output.shape), (2, 5, 15, 20))
        boxes = [(.3, .5, .2, .6), (.75, .5, .2, .5)]
        heat, distances, mask = targets_from_boxes(boxes, 160, 120, 8)
        logits = np.full_like(heat, -20)
        logits[mask > 0] = 20
        metrics, _ = evaluate_boxes(np.concatenate([distances, logits])[None],
                                    [boxes], width=160, height=120, stride=8,
                                    local_maximum=True)
        self.assertEqual(metrics['tp'], 2)
        self.assertEqual(metrics['f1'], 1.)

    def test_camera_augmentation_uses_requested_geometry(self):
        image = np.zeros((1, 120, 160), dtype=np.float32)
        value, targets, boxes = augment_profile(image, [(.3, .5, .2, .6)],
            np.random.default_rng(4), width=160, height=120, stride=8)
        self.assertEqual(value.shape, image.shape)
        self.assertEqual(targets[0].shape, (1, 15, 20))
        self.assertEqual(float(targets[2].sum()), 1.)
        self.assertEqual(float(boxes[0, 1]), .5)

    def test_output_geometry_and_parameter_budget(self):
        model=ProfileDetector().eval()
        with torch.no_grad(): output=model(torch.zeros(2,1,HEIGHT,WIDTH))
        self.assertEqual(tuple(output.shape),(2,5,15,20))
        self.assertLess(sum(p.numel() for p in model.parameters()),640_000)

    def test_perfect_targets_decode_at_qvga_geometry(self):
        boxes=[(.3,.5,.2,.6),(.75,.5,.2,.5)]
        heat,distances,mask=targets_from_boxes(boxes,WIDTH,HEIGHT,STRIDE)
        logits=np.full_like(heat,-20)
        logits[mask>0]=20
        output=np.concatenate([distances,logits],axis=0)[None]
        metrics,_=evaluate_boxes(output,[boxes],width=WIDTH,height=HEIGHT,stride=STRIDE,local_maximum=True)
        self.assertEqual(metrics["tp"],2)
        self.assertEqual(metrics["f1"],1.)

    def test_augmentation_never_flips_people_upside_down(self):
        image=np.broadcast_to(np.linspace(0,1,HEIGHT,dtype=np.float32)[None,:,None],(1,HEIGHT,WIDTH)).copy()
        boxes=[(.3,.4,.2,.5)]
        value,targets,actual=augment_profile(image,boxes,np.random.default_rng(9),enabled=False)
        np.testing.assert_array_equal(value,image)
        np.testing.assert_allclose(actual,boxes)
        value,_,actual=augment_profile(image,boxes,np.random.default_rng(1),enabled=True)
        self.assertLess(float(value[:,0].mean()),float(value[:,-1].mean()))
        self.assertEqual(float(actual[0,1]),.4)
        self.assertTrue(all(np.isfinite(v).all() for v in targets))

    def test_generic_geometry_preserves_original_qqvga_defaults(self):
        boxes=[(.5,.5,.4,.5)]
        default=targets_from_boxes(boxes)
        explicit=targets_from_boxes(boxes,width=160,height=120,stride=4)
        for a,b in zip(default,explicit): np.testing.assert_array_equal(a,b)

    def test_shared_video_paths_are_relocated_without_copying_a_mac_venv(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            path=root/"videotest2/Echantillon/example.mp4"
            path.parent.mkdir(parents=True)
            path.touch()
            with patch("prepare_profile_mix.ROOT",root):
                for value in ("/unavailable_mac/FlowSense/example.mp4",r"Z:\unavailable_windows\example.mp4"):
                    self.assertEqual(video_name(value),"example.mp4")
                    self.assertEqual(local_video(value),path)


if __name__=="__main__": unittest.main()
