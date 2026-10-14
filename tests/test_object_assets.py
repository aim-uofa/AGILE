"""Final-asset preservation, verified alignment, and portable dual variants."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image
import trimesh

from optimization.data.object_assets import (prepare_object_asset, read_object_asset, load_mesh,
    fit_asset_to_input, texture_image, file_hash, write_native_object_assets, export_lossless_glb, export_lossless_mesh)
from optimization.data.result_bundle import ResultBundle, pack_result
from tests import test_result_bundle as fixtures


def textured_box():
    mesh = trimesh.creation.box(extents=[.04, .05, .06])
    # Separate every triangle corner to exercise UV seams.
    vertices = mesh.triangles.reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.arange(len(vertices)).reshape(-1, 3), process=False)
    texture = np.arange(32 * 48 * 3, dtype=np.uint16).reshape(32, 48, 3).astype(np.uint8)
    mesh.visual = trimesh.visual.TextureVisuals(uv=np.tile([[0.,0.],[1.,0.],[.5,1.]], (12,1)),
                                               image=Image.fromarray(texture))
    return mesh


class ObjectAssetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'final.obj'
        self.mesh = textured_box(); self.mesh.export(self.source)
        self.mesh = load_mesh(self.source)
        self.canonical = self.mesh.copy()
        self.transform = trimesh.transformations.rotation_matrix(.3, [0, 0, 1])
        self.transform[:3,:3] *= 1.4; self.transform[:3,3] = [.01,.02,.03]
        self.canonical.apply_transform(self.transform)
        self.input = self.root / 'canonical.glb'; self.canonical.export(self.input)
        self.canonical = load_mesh(self.input)

    def register(self):
        return prepare_object_asset(self.source, self.input, self.root / 'asset',
            stage='blender-final', provenance={'selection': 'synthetic final UV fixture'})

    def test_similarity_preserves_source_pixels_and_seams(self):
        manifest = self.register()
        actual, record = read_object_asset(manifest, self.canonical)
        np.testing.assert_allclose(actual.vertices, self.canonical.vertices, atol=1e-8)
        np.testing.assert_array_equal(actual.faces, self.mesh.faces)
        np.testing.assert_array_equal(actual.visual.uv, self.mesh.visual.uv)
        np.testing.assert_array_equal(np.asarray(texture_image(actual)), np.asarray(texture_image(self.mesh)))
        self.assertEqual(file_hash(self.source), record['sha256']['source/final.obj'])
        np.testing.assert_allclose(record['asset_to_input'], self.transform, atol=1e-7)

    def test_wrong_geometry_uv_and_heavy_intermediate_fail(self):
        other = self.canonical.copy(); other.vertices[0] += [.01,0,0]
        with self.assertRaisesRegex(ValueError, 'does not match'):
            fit_asset_to_input(self.mesh, other)
        other = self.canonical.copy(); other.visual.uv[0] += [.1,0]
        with self.assertRaisesRegex(ValueError, 'UV correspondence'):
            fit_asset_to_input(self.mesh, other)
        with self.assertRaisesRegex(ValueError, 'postprocessing stage'):
            prepare_object_asset(self.source, self.input, self.root/'bad', stage='hunyuan-raw', provenance={'a':1})

    def test_source_tampering_and_wrong_input_fail(self):
        manifest = self.register()
        other = self.canonical.copy(); other.vertices *= 2
        with self.assertRaisesRegex(ValueError, 'different canonical input'):
            read_object_asset(manifest, other)
        with (manifest.parent/'source/final.obj').open('a') as stream: stream.write('\n# changed\n')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            read_object_asset(manifest)

    def test_dual_export_and_appearance_guard(self):
        manifest = self.register()
        proxy = self.canonical.copy()
        proxy.visual = trimesh.visual.TextureVisuals(uv=proxy.visual.uv,
            image=Image.fromarray(np.full((2,2,3), 17, dtype=np.uint8)))
        record = write_native_object_assets(self.root/'native', proxy, self.canonical, asset_manifest=manifest)
        display = load_mesh(self.root/'native'/record['variants']['display']['mesh'])
        saved_proxy = load_mesh(self.root/'native'/record['variants']['optimization_proxy']['mesh'])
        self.assertEqual(texture_image(display).size, (48,32))
        self.assertEqual(texture_image(saved_proxy).size, (2,2))
        guarded = write_native_object_assets(self.root/'guarded', proxy, self.canonical,
                                             asset_manifest=manifest, appearance_optimized=True)
        self.assertEqual(guarded['texture_policy'], 'optimized-proxy')
        self.assertIn('source_reference', guarded['variants'])

    def test_jpeg_pixels_are_not_recompressed_in_glb(self):
        source = self.root/'texture.jpg'
        texture_image(self.mesh).save(source, quality=93)
        with Image.open(source) as image:
            self.mesh.visual.material.image = image.copy()
            self.mesh.visual.material.image.format = 'JPEG'
            expected = np.asarray(image).copy()
        target = self.root/'lossless.glb'
        export_lossless_glb(self.mesh, target)
        np.testing.assert_array_equal(np.asarray(texture_image(load_mesh(target)))[:,:,:3], expected[:,:,:3])
        loaded=load_mesh(target); loaded.visual.material.baseColorFactor=None
        legacy=self.root/'legacy.obj'; export_lossless_mesh(loaded,legacy)
        legacy_mesh=load_mesh(legacy)
        np.testing.assert_array_equal(np.asarray(texture_image(legacy_mesh))[:,:,:3],expected[:,:,:3])
        np.testing.assert_array_equal(legacy_mesh.visual.material.diffuse,[255,255,255,255])

    def test_portable_reexport_keeps_pose_scale_and_original_files(self):
        fixture = fixtures.BundleTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        (fixture.root/'config.json').write_text(json.dumps(dict(optimize_object_texture=False,optimize_vertex_colors=False)))
        manifest = self.register()
        old = fixture.root/'old'
        pack_result(fixture.source, old, source_fps=25, source_fps_provenance='fixture',handedness='right')
        # Emulate the old v1 schema which did not retain object scale.
        metadata = json.loads((old/'result.json').read_text())
        metadata['schema_version'] = 1; metadata.pop('optimized_object_scale')
        (old/'result.json').write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'lost the original object scale'):
            pack_result(old, fixture.root/'missing', asset_manifest=manifest, canonical_input=self.input)
        output = fixture.root/'new'
        pack_result(old, output, asset_manifest=manifest, canonical_input=self.input,
                    native_object_poses=fixture.source/'object_poses.npz')
        b = ResultBundle(output); self.assertTrue(b.verify_assets())
        np.testing.assert_allclose(b.object_mesh.vertices, self.canonical.vertices * [2,3,4], atol=2e-8)
        previous = ResultBundle(old)
        np.testing.assert_array_equal(b.object_poses_in_cam, previous.object_poses_in_cam)
        np.testing.assert_array_equal(b.mano_verts, previous.mano_verts)
        np.testing.assert_allclose(b.load_object_variant('optimization_proxy').vertices, previous.object_mesh.vertices)
        self.assertEqual(file_hash(output/'source_asset/source/final.obj'), file_hash(self.source))
        repacked = fixture.root/'repacked'; pack_result(output, repacked)
        moved = fixture.root/'moved'; shutil.move(repacked, moved)
        shutil.rmtree(self.root/'asset'); shutil.rmtree(output)
        restored = ResultBundle(moved); self.assertTrue(restored.verify_assets())
        self.assertEqual(restored.manifest['handedness'], 'right')
        np.testing.assert_allclose(restored.object_mesh.vertices, b.object_mesh.vertices, atol=1e-8)
        np.testing.assert_array_equal(restored.frame_ids, previous.frame_ids)

    def test_unknown_appearance_cannot_be_silently_replaced(self):
        fixture = fixtures.BundleTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        manifest = self.register()
        with self.assertRaisesRegex(ValueError, 'appearance optimization policy is unknown'):
            pack_result(fixture.source,fixture.root/'replacement',asset_manifest=manifest,canonical_input=self.input)

    def test_native_variants_apply_scale_only_once(self):
        fixture = fixtures.BundleTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        manifest = self.register()
        proxy = load_mesh(fixture.source/'object_mesh.obj')
        write_native_object_assets(fixture.source, proxy, self.canonical, asset_manifest=manifest)
        b = ResultBundle(fixture.source)
        np.testing.assert_allclose(b.object_mesh.vertices, self.canonical.vertices * [2,3,4], atol=2e-8)
        output = fixture.root/'bundle'; pack_result(fixture.source, output)
        packed = ResultBundle(output)
        np.testing.assert_allclose(packed.object_mesh.vertices, b.object_mesh.vertices, atol=1e-8)
        np.testing.assert_allclose(packed.load_object_variant('optimization_proxy').vertices,
                                   proxy.vertices * [2,3,4], atol=1e-8)


if __name__ == '__main__':
    unittest.main()
