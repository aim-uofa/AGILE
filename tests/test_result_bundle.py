"""Frame identity, final assets, coordinate space and bounded legacy reading."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import trimesh
from optimization.data.result_bundle import ResultBundle, pack_result, texture_uint8, ZipFrames
from optimization.data.visualization_loader import slice_rotations
from scripts.vis.render_demo import playback_times, composite_panels


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.source=self.root/'aligned';self.source.mkdir()
        n=4
        rgbd=np.zeros((n,3,4,4),np.float32);rgbd[...,:3]=[51,102,204];rgbd[...,3]=.5
        rgbd[:,0,0,0]=np.arange(n)+100
        c=np.tile(np.eye(4),(n,1,1));c[:,0,3]=np.arange(n)*.1
        self.ids=[100,105,110,120]
        np.savez(self.source/'camera_data.npz',rgbd=rgbd,intrinsic=np.array([[100.,0,1],[0,110.,1],[0,0,1]]),
                 prediction_frame_ids=self.ids,cam_c2w=c)
        self.rgbd=rgbd
        verts=np.tile(np.array([[[0.,0,.5],[.1,0,.5],[0,.1,.5]]]),(n,1,1,1))
        np.savez(self.source/'optimized_mano_verts.npz',mano_verts=verts)
        np.savez(self.source/'mano_faces.npz',mano_faces=[[0,1,2]])
        poses=np.tile(np.eye(4),(n,1,1));poses[:,2,3]=.5
        np.savez(self.source/'object_poses.npz',object_poses_in_cam=poses,object_scale=[2.,3.,4.])
        mesh=trimesh.creation.box(extents=[.01,.02,.03]);mesh.visual=trimesh.visual.TextureVisuals(
            uv=np.full((len(mesh.vertices),2),.5),image=Image.fromarray(np.tile(np.array([51,102,204],np.uint8),(2,2,1))))
        mesh.export(self.source/'object_mesh.obj')

    def test_slice_preserves_original_ids_and_time(self):
        b=ResultBundle(self.source,start_frame=1,max_frames=2,source_fps=25,source_fps_provenance='fixture')
        self.assertEqual(b.frame_ids.tolist(),[105,110]);self.assertEqual(b.timestamps.tolist(),[4.2,4.4])
        np.testing.assert_array_equal(b.rgbd[0],self.rgbd[1]);np.testing.assert_array_equal(b.rgbd[0,0],self.rgbd[1,0])
        np.testing.assert_array_equal(b.rgbd[0,...,:3],self.rgbd[1,...,:3])
        np.testing.assert_allclose(b.get_mano_vertices(0,world=True)-b.get_mano_vertices(0),[[.1,0,0]]*3)
        np.testing.assert_allclose(b.get_object_pose(0,world=True)[:3,3],[.1,0,.5])

    def test_rgbd_never_eagerly_loaded_by_npz(self):
        original=np.lib.npyio.NpzFile.__getitem__
        def guarded(obj,key):
            if key=='rgbd':raise AssertionError('Eager video allocation')
            return original(obj,key)
        with patch.object(np.lib.npyio.NpzFile,'__getitem__',guarded):
            b=ResultBundle(self.source,max_frames=1)
            np.testing.assert_array_equal(b.get_rgb(0),self.rgbd[0,...,:3])

    def test_portable_roundtrip_and_single_scale(self):
        out=self.root/'bundle';pack_result(self.source,out,source_fps=25,source_fps_provenance='fixture')
        shutil.rmtree(self.source)
        moved=self.root/'relocated';out.rename(moved)
        b=ResultBundle(moved);self.assertTrue(b.verify_assets())
        np.testing.assert_allclose(b.object_mesh.extents,[.02,.06,.12],rtol=1e-6)
        tex=np.asarray(b.object_mesh.visual.material.baseColorTexture)
        np.testing.assert_array_equal(tex[0,0,:3],[51,102,204])
        np.testing.assert_array_equal(b.get_depth(3),self.rgbd[3,...,3])
        (moved/'rgb/000000.png').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):b.verify_assets()

    def test_reference_images_require_exact_frame_ids(self):
        reference=self.root/'refs';reference.mkdir()
        Image.new('RGB',(4,3)).save(reference/'0000.png')
        with self.assertRaisesRegex(ValueError,'Missing reference'):pack_result(self.source,self.root/'out',reference_rgb_dir=reference)
        for fid in self.ids:Image.new('RGB',(4,3),(10,20,30)).save(reference/f'{fid:04d}.png')
        m=pack_result(self.source,self.root/'out',reference_rgb_dir=reference)
        self.assertEqual(m['rgb_source'],'original-reference-frames')
        self.assertEqual(m['frames'][2]['frame_id'],110)
        with self.assertRaises(FileExistsError):pack_result(self.source,self.root/'out')

    def test_unknown_time_requires_explicit_presentation_setting(self):
        b=ResultBundle(self.source)
        with self.assertRaisesRegex(ValueError,'Unknown source timing'):playback_times(b)
        times,provenance=playback_times(b,5)
        np.testing.assert_allclose(times,[0,.2,.4,.6]);self.assertEqual(provenance,'presentation-fps-override')
        with self.assertRaises(ValueError):ResultBundle(self.source,source_fps=25)

    def test_nonzero_rotation_slice_and_shared_matrix(self):
        r=np.array([[0,0,0],[0,0,np.pi/2],[0,0,np.pi],[0,0,0]])
        np.testing.assert_array_equal(slice_rotations(r,[1],4),r[1:2])
        np.testing.assert_array_equal(slice_rotations(np.eye(3),[1,2],4),np.tile(np.eye(3),(2,1,1)))
        with self.assertRaisesRegex(ValueError,'Ambiguous'):slice_rotations(np.eye(3),[1],3)

    def test_texture_encoding_no_brightness_boost(self):
        np.testing.assert_array_equal(texture_uint8(np.array([.2,.4,.8])),[51,102,204])
        with self.assertRaises(ValueError):texture_uint8(np.array([np.nan]))
        with self.assertRaises(ValueError):texture_uint8(np.array([1.2]))

    def test_overlay_keeps_background_and_rgb_order(self):
        rgb=np.array([[[51,102,204],[255,50,10]]],np.uint8)
        overlay,render=composite_panels(rgb,np.zeros((1,2,3)),np.array([[[0],[1]]]))
        np.testing.assert_array_equal(overlay[0,0],rgb[0,0])
        np.testing.assert_array_equal(render[0,0],[255,255,255])

    def test_dark_legacy_rgb_is_not_auto_exposed(self):
        path=self.source/'camera_data.npz'
        with np.load(path) as z: arrays={k:z[k] for k in z.files}
        arrays['rgbd'][0,...,:3]=1
        np.savez(path,**arrays)
        b=ResultBundle(self.source)
        np.testing.assert_array_equal(b.get_rgb(0),np.ones((3,4,3),np.uint8))

    def test_missing_mesh_or_topology_is_not_marked_valid(self):
        (self.source/'object_mesh.obj').unlink()
        (self.source/'mano_faces.npz').unlink()
        b=ResultBundle(self.source)
        self.assertFalse(b.valid_hand.any())
        self.assertFalse(b.valid_object.any())

    def test_stream_reader_repeated_and_out_of_order_frames(self):
        z=ZipFrames(self.source/'camera_data.npz','rgbd');self.addCleanup(z.close)
        for i in [3,1,1,0,2]:np.testing.assert_array_equal(z[i],self.rgbd[i])

if __name__=='__main__':unittest.main()
