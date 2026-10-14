"""Render a final result with its saved camera, texture and original frame IDs."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from optimization.data.result_bundle import ResultBundle, sha256


def playback_times(bundle, presentation_fps=None):
    """Return relative seconds; never present a guessed FPS as source timing."""
    if presentation_fps is not None:
        if not np.isfinite(presentation_fps) or presentation_fps <= 0:
            raise ValueError('Presentation FPS must be positive')
        return np.arange(bundle.num_frames) / presentation_fps, 'presentation-fps-override'
    if any(t is None for t in bundle.timestamps):
        raise ValueError('Unknown source timing: pack with documented --source-fps, or use --presentation-fps explicitly')
    times = np.asarray(bundle.timestamps, dtype=float)
    return times - times[0], bundle.time_source


class FinalRenderer:
    """No model weights, initialization data, geometry simplification or pose fitting."""
    def __init__(self, bundle, device='cuda'):
        import torch
        from pytorch3d.renderer import AmbientLights, Materials, HardPhongShader
        self.torch, self.bundle, self.device = torch, bundle, device
        self.shader = HardPhongShader(device=device, lights=AmbientLights(device=device),
            materials=Materials(device=device, ambient_color=((1.,1.,1.),),
                                diffuse_color=((0.,0.,0.),), specular_color=((0.,0.,0.),)))
        self.object = None
        self.object_normals = None
        self.surface_distance = None
        if bundle.has_object:
            m = bundle.object_mesh
            visual = m.visual
            material = getattr(visual, 'material', None)
            texture = getattr(material, 'image', None)
            if texture is None:
                texture = getattr(material, 'baseColorTexture', None)
            uvs = getattr(visual, 'uv', None)
            if texture is not None and uvs is not None:
                texture = np.asarray(texture.convert('RGB')).astype(np.float32) / 255.
            else:
                texture, uvs = None, None
            self.object = (m.vertices.copy(), m.faces.copy(), uvs, texture)
            self.object_normals = m.vertex_normals.copy()

    def mesh(self, vertices, faces, uv=None, texture=None, color=(.68,.76,.85)):
        from pytorch3d.structures import Meshes
        from pytorch3d.renderer import TexturesUV
        torch = self.torch
        if texture is None:
            texture = np.tile(np.asarray(color, dtype=np.float32), (2,2,1))
            uv = np.full((len(vertices),2), .5, dtype=np.float32)
        t = lambda a, dtype: torch.as_tensor(np.asarray(a).copy(), dtype=dtype, device=self.device)
        faces = t(faces, torch.int64)
        textures = TexturesUV(maps=[t(texture, torch.float32)], faces_uvs=[faces],
                              verts_uvs=[t(uv, torch.float32)])
        return Meshes(verts=[t(vertices,torch.float32)],faces=[faces],textures=textures)

    def render(self, i, diagnostics=False):
        from pytorch3d.structures import join_meshes_as_scene
        from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
        from optimization.utils.camera_utils import convert_camtoworld_to_pytorch3d_cameras
        torch, b = self.torch, self.bundle
        meshes = []
        diagnostic_colors = []
        object_vertices = hand_vertices = None
        object_faces_count = 0
        light = np.array([.2, -.5, -1.]); light /= np.linalg.norm(light)
        if self.object is not None and b.valid_object[i]:
            v,f,uv,tex = self.object
            pose = b.get_object_pose(i)
            object_vertices = v @ pose[:3,:3].T + pose[:3,3]
            object_faces_count = len(f)
            meshes.append(self.mesh(object_vertices,f,uv,tex))
            if diagnostics:
                intensity = .4 + .6 * np.clip((self.object_normals @ pose[:3,:3].T) @ light, 0, 1)
                diagnostic_colors.append((intensity[:,None] * np.array([.84,.84,.82]))[f])
        if b.has_mano and b.valid_hand[i]:
            # Shade only the diagnostic hand, keeping object albedo unchanged.
            # A one-dimensional UV ramp carries smooth per-vertex lighting while
            # allowing PyTorch3D to join it with the object's UV texture.
            import trimesh
            v = b.get_mano_vertices(i)
            hand_vertices = v
            normals = trimesh.Trimesh(vertices=v, faces=b.mano_faces, process=False).vertex_normals
            intensity = .4 + .6 * np.clip(normals @ light, 0, 1)
            uv = np.column_stack([intensity, np.full(len(v), .5)])
            ramp = np.tile(np.array([[0.,0.,0.],[.68,.76,.85]],np.float32), (2,1,1))
            meshes.append(self.mesh(v, b.mano_faces, uv, ramp))
            if diagnostics:
                diagnostic_colors.append((intensity[:,None] * np.array([.68,.76,.85]))[b.mano_faces])
        if not meshes:
            empty = (np.zeros((b.height,b.width,3),np.float32), np.zeros((b.height,b.width,1),np.float32))
            return (*empty, dict(geometry=empty[0], object_mask=empty[1][...,0], hand_mask=empty[1][...,0], contact=None)) if diagnostics else empty
        k = b.K[i] if b.K.ndim == 3 else b.K
        # Saved geometry is already in camera coordinates: do not apply c2w twice.
        cameras = convert_camtoworld_to_pytorch3d_cameras(torch.eye(4,device=self.device)[None],
            torch.as_tensor(k,dtype=torch.float32,device=self.device)[None],b.height,b.width,self.device)
        raster = MeshRasterizer(cameras=cameras, raster_settings=RasterizationSettings(
            image_size=(b.height,b.width),blur_radius=0.,faces_per_pixel=1,bin_size=0))
        with torch.no_grad():
            mesh = join_meshes_as_scene(meshes)
            fragments = raster(mesh)
            rgba = self.shader(fragments,mesh,cameras=cameras)
            alpha = (fragments.pix_to_face[0,...,:1] >= 0).float().cpu().numpy()
            colors = rgba[0,...,:3].clamp(0,1).cpu().numpy()
            if diagnostics:
                from pytorch3d.ops import interpolate_face_attributes
                attributes = torch.as_tensor(np.concatenate(diagnostic_colors),dtype=torch.float32,device=self.device)
                geometry = interpolate_face_attributes(fragments.pix_to_face,fragments.bary_coords,attributes)[0,...,0,:].cpu().numpy()
                face_ids = fragments.pix_to_face[0,...,0].cpu().numpy()
                contact = None
                if object_vertices is not None and hand_vertices is not None:
                    from optimization.data.surface_distance import SurfaceDistance, summarize_distance
                    if self.surface_distance is None:
                        self.surface_distance = SurfaceDistance(self.object[0], self.object[1])
                    contact = summarize_distance(self.surface_distance.millimetres(hand_vertices,b.get_object_pose(i)))
                return colors, alpha, dict(geometry=geometry,
                    object_mask=(face_ids>=0)&(face_ids<object_faces_count),
                    hand_mask=(face_ids>=object_faces_count)&(face_ids>=0), contact=contact)
        return colors,alpha


def composite_panels(rgb, colors, mask, opacity=.65):
    original = np.asarray(rgb,dtype=np.float32)/255.
    overlay = original*(1-mask*opacity) + colors*mask*opacity
    render = colors*mask + np.ones_like(original)*(1-mask)
    return (np.rint(np.clip(overlay,0,1)*255).astype(np.uint8),
            np.rint(np.clip(render,0,1)*255).astype(np.uint8))


def render_result(data_dir, output, *, device='cuda', presentation_fps=None, max_frames=None):
    import imageio.v2 as imageio
    b = ResultBundle(data_dir,max_frames=max_frames)
    if not b.manifest:
        raise ValueError('First run scripts.vis.pack_result to freeze the final assets and reference frames')
    b.verify_assets()
    times,time_source = playback_times(b,presentation_fps)
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a new visualization directory: '+str(output))
    output.mkdir(parents=True)
    (output/'frames').mkdir()
    renderer = FinalRenderer(b,device=device)
    # CFR is exact for equally spaced predictions. For irregular samples write a
    # timestamped HTML report; avoid silently changing their video timing.
    equal_spacing = len(times)>1 and np.allclose(np.diff(times),np.diff(times)[0],rtol=1e-6,atol=1e-8)
    fps = 1/np.diff(times)[0] if equal_spacing else None
    writer = imageio.get_writer(output/'comparison.mp4',fps=fps,codec='libx264',quality=8,
                                 macro_block_size=2) if fps else None
    frames = []
    try:
        for i in range(b.num_frames):
            rgb = b.get_rgb(i)
            colors,mask = renderer.render(i)
            overlay,render = composite_panels(rgb,colors,mask)
            paths = {}
            for name,arr in [('reference',rgb),('overlay',overlay),('render',render)]:
                path = f'frames/{i:06d}-{name}.jpg'
                Image.fromarray(arr).save(output/path,quality=94,subsampling=0)
                paths[name] = path
            if writer:
                canvas = Image.new('RGB',(b.width*3,b.height+38),(23,33,45))
                canvas.paste(Image.fromarray(np.concatenate([rgb,overlay,render],axis=1)),(0,38))
                draw=ImageDraw.Draw(canvas)
                fid = b.frame_ids[i]
                for j,title in enumerate(['REFERENCE','OVERLAY 65%','FINAL TEXTURED MESH']):
                    draw.text((j*b.width+12,12),f'{title} | frame {fid} | {float(b.timestamps[i]):.3f}s' if b.timestamps[i] is not None else f'{title} | frame {fid}',fill='white')
                writer.append_data(np.asarray(canvas))
            frames.append(dict(index=i,frame_id=b.frame_ids[i],timestamp_seconds=b.timestamps[i],
                               playback_seconds=float(times[i]),valid_hand=bool(b.valid_hand[i]),
                               valid_object=bool(b.valid_object[i]),**paths))
            if i%25==0: print(f'Rendered {i+1}/{b.num_frames}',flush=True)
    finally:
        if writer: writer.close()
    report = dict(schema_version=1,label=b.manifest['label'],frames=frames,
                  time_source=time_source,rgb_source=b.manifest['rgb_source'],
                  source_frame_count=b.manifest.get('reference_frame_count'),
                  bundle_manifest_sha256=sha256(Path(data_dir)/'result.json'),
                  frame_ids_source=b.frame_ids_source,notices=b.notices,
                  video='comparison.mp4' if writer else None,video_fps=float(fps) if fps else None,
                  rendering=dict(coordinates='opencv-camera-metres',camera='full saved K',
                                 geometry='saved final vertices; no decimation',
                                 texture='saved object albedo; ambient lighting; no exposure boost',
                                 hand_display='blue-gray, fixed camera-space diffuse shading',overlay_opacity=.65))
    (output/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    from scripts.vis.make_webpage_demo import write_page
    write_page([report],output/'index.html')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir','--data_dir',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--presentation-fps',type=float)
    p.add_argument('--max-frames',type=int)
    render_result(**vars(p.parse_args()))

if __name__=='__main__': main()
