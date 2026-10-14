"""Failure safeguards for local, licensed MANO asset preparation."""
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools.prepare_mano_assets import closed_wrist_faces, prepare_assets


class ManoAssetPreparationTests(unittest.TestCase):
    def test_unsupported_or_already_closed_topology_is_rejected(self):
        for shape in [(1537, 3), (1552, 3), (1538, 4)]:
            with self.assertRaisesRegex(ValueError, "1538-face"):
                closed_wrist_faces(np.zeros(shape, dtype=np.int64))

    def test_incompatible_topology_is_not_silently_capped(self):
        with self.assertRaisesRegex(ValueError, "does not close"):
            closed_wrist_faces(np.zeros((1538, 3), dtype=np.int64))

    @unittest.skipUnless(os.environ.get("MANO_ASSET_TEST_MODEL_DIR"),
                         "Licensed MANO test model not configured")
    def test_real_mano_preparation_preserves_closed_wrist_and_finger_selection(self):
        import smplx
        import trimesh
        model_dir = Path(os.environ["MANO_ASSET_TEST_MODEL_DIR"])
        model = smplx.MANO(str(model_dir), is_rhand=True, use_pca=False,
                           flat_hand_mean=False)
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = prepare_assets(model_dir, output, project / "configs/finger_vertices.json")
            mesh = trimesh.load(output / "mano_hand.obj", process=False)
            self.assertEqual(report["faces"], 1552)
            self.assertEqual(report["wrist_cap_faces"], 14)
            self.assertTrue(mesh.is_watertight)
            self.assertTrue(mesh.is_winding_consistent)
            np.testing.assert_array_equal(mesh.faces[:1538], model.faces)
            np.testing.assert_allclose(mesh.vertices, model.v_template.numpy(), atol=1e-8)
            with np.load(output / "finger_vertices_mask.npz") as archive:
                self.assertEqual(archive["finger_vertices_mask"].shape, (778,))
                self.assertEqual(int(archive["finger_vertices_mask"].sum()), 516)
            reference = os.environ.get("MANO_ASSET_TEST_REFERENCE_MESH")
            if reference:
                original = trimesh.load(reference, process=False)
                np.testing.assert_array_equal(mesh.faces, original.faces)

    def test_missing_licensed_model_does_not_create_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            with self.assertRaises(FileNotFoundError):
                prepare_assets(base / "missing", base / "output")
            self.assertFalse((base / "output").exists())

    def test_existing_configuration_is_preserved_without_loading_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "MANO_RIGHT.pkl").write_bytes(b"not opened")
            config = base / "mano_config.yaml"
            config.write_text("existing user configuration")
            with patch("tools.prepare_mano_assets.smplx.MANO") as loader:
                with self.assertRaises(FileExistsError):
                    prepare_assets(base, base)
                loader.assert_not_called()
            self.assertEqual(config.read_text(), "existing user configuration")
            self.assertFalse((base / "mano_hand.obj").exists())

    def test_invalid_finger_mask_is_rejected_before_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "MANO_RIGHT.pkl").write_bytes(b"not opened")
            np.savez(base / "bad-mask.npz", finger_vertices_mask=np.ones(21, dtype=bool))
            with patch("tools.prepare_mano_assets.smplx.MANO") as loader:
                with self.assertRaisesRegex(ValueError, "shape"):
                    prepare_assets(base, base / "output", base / "bad-mask.npz")
                loader.assert_not_called()
            self.assertFalse((base / "output").exists())


if __name__ == "__main__":
    unittest.main()
