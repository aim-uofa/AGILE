"""Inspect final hand/object results in world space with bounded scene memory."""
import argparse
import threading
import time
from pathlib import Path
import numpy as np
import trimesh
from optimization.data.visualization_loader import VisualizationLoader


def encode_calibrated_gif(bundle, render_frame, presentation_fps=5):
    """Each encoded frame explicitly renders that index, independent of UI state."""
    import imageio.v2 as imageio
    frames = [render_frame(i) for i in range(bundle.num_frames)]
    if bundle.num_frames > 1 and all(t is not None for t in bundle.timestamps):
        dt = np.diff(np.asarray(bundle.timestamps, dtype=float))
        durations = np.r_[dt, dt[-1]] * 1000
    else:
        durations = np.full(bundle.num_frames, 1000 / presentation_fps)
    durations = np.maximum(10, np.rint(durations / 10) * 10).astype(int)
    return imageio.mimwrite('<bytes>', frames, format='GIF', duration=durations.tolist(), loop=0)


def main(data_dir, port=8090, start=0, end=-1, max_frames=256,
         downsample_factor=4, device='cuda', share=False, object_variant='display'):
    import viser
    import viser.transforms as tf
    import imageio.v2 as imageio
    b=VisualizationLoader(data_dir,start_frame=start,end_frame=end,max_frames=max_frames,object_variant=object_variant,
                          downsample_factor=downsample_factor)
    server=viser.ViserServer(host='127.0.0.1',port=port)
    if share: server.request_share_url()
    lock=threading.RLock()
    nodes=[]
    state={'index':0,'rendering':False,'renderer':None}
    with server.gui.add_folder('Playback'):
        slider=server.gui.add_slider('Array index',min=0,max=max(1,b.num_frames-1),step=1,initial_value=0,disabled=b.num_frames==1)
        info=server.gui.add_text('Original frame / time',initial_value='',disabled=True)
        play=server.gui.add_checkbox('Playing',initial_value=False,disabled=b.num_frames==1)
        rate=server.gui.add_slider('Presentation FPS (unknown time only)',min=1,max=30,step=1,initial_value=5)
        prev=server.gui.add_button('Previous frame');nxt=server.gui.add_button('Next frame')
    with server.gui.add_folder('Display'):
        show_points=server.gui.add_checkbox('Depth points',initial_value=False)
        show_hand=server.gui.add_checkbox('Final hand',initial_value=True)
        show_object=server.gui.add_checkbox('Final textured object',initial_value=True)
        show_texture=server.gui.add_checkbox('Object texture (off = white mesh)',initial_value=True)
        reference=server.gui.add_image(b.get_rgb(0),label='Same original frame')
        server.gui.add_markdown('Geometry and depth points share world coordinates. The camera frustum is illustrative; calibrated exports use the full saved intrinsic matrix.')
    with server.gui.add_folder('Export'):
        png=server.gui.add_button('Calibrated camera PNG')
        free_png=server.gui.add_button('Current free view PNG')
        gif=server.gui.add_button('Calibrated sequence GIF')
        message=server.gui.add_text('Export status',initial_value='Ready',disabled=True)
        server.gui.add_markdown('Exports render final geometry directly, including off-centre intrinsics. GIF timing uses saved timestamps, or the explicit presentation FPS above. GIF durations are rounded to 10 ms.')
    for notice in b.notices: server.gui.add_markdown(notice)

    def update(i):
        i=int(i)
        if not 0<=i<b.num_frames:return
        with lock,server.atomic():
            state['index']=i
            for node in nodes:node.remove()
            nodes.clear()
            c=b.cam_c2w[i]; k=b.K[i] if b.K.ndim==3 else b.K
            if b.has_mano and b.valid_hand[i] and show_hand.value:
                nodes.append(server.scene.add_mesh_simple('/result/hand',vertices=b.get_mano_vertices(i,world=True),
                    faces=b.mano_faces,color=(174,194,217),side='double'))
            if b.has_object and b.valid_object[i] and show_object.value:
                m=b.object_mesh.copy();m.apply_transform(b.get_object_pose(i,world=True))
                if show_texture.value:
                    nodes.append(server.scene.add_mesh_trimesh('/result/object',m))
                else:
                    nodes.append(server.scene.add_mesh_simple('/result/object',vertices=m.vertices,
                        faces=m.faces,color=(220,220,220),side='double'))
            if show_points.value:
                points,colors=b.get_points(i)
                nodes.append(server.scene.add_point_cloud('/result/depth',points=points,colors=colors,point_size=.001))
            nodes.append(server.scene.add_camera_frustum('/result/camera',fov=float(2*np.arctan(b.height/(2*k[1,1]))),
                aspect=b.width/b.height,scale=.05,wxyz=tf.SO3.from_matrix(c[:3,:3]).wxyz,position=c[:3,3]))
            reference.image=b.get_rgb(i)
            stamp='unknown time' if b.timestamps[i] is None else f'{b.timestamps[i]:.3f}s'
            info.value=f'{b.frame_ids[i]} / {stamp} (array {b.indices[i]})'
    @slider.on_update
    def _(_event):update(slider.value)
    for control in [show_points,show_hand,show_object,show_texture]:
        control.on_update(lambda event:update(state['index']))
    @prev.on_click
    def _(_event):
        play.value=False;slider.value=max(0,state['index']-1)
    @nxt.on_click
    def _(_event):
        play.value=False;slider.value=min(b.num_frames-1,state['index']+1)
    @server.on_client_connect
    def _(client):
        c=b.cam_c2w[state['index']]
        client.camera.position=c[:3,3];client.camera.look_at=c[:3,3]+c[:3,2]*.6
        client.camera.up_direction=-c[:3,1]

    def render_frame(i):
        from scripts.vis.render_demo import FinalRenderer,composite_panels
        if state['renderer'] is None:state['renderer']=FinalRenderer(b,device)
        colors,alpha=state['renderer'].render(i)
        return composite_panels(b.get_rgb(i),colors,alpha)[0]
    def export(event,sequence):
        if event.client is None:return
        with lock:
            if state['rendering']:return
            state['rendering']=True
            was_playing=play.value;play.value=False
            saved_index=state['index']
            png.disabled=gif.disabled=True
            try:
                if sequence:
                    def render_progress(i):
                        message.value=f'Rendering {i+1}/{b.num_frames}'
                        return render_frame(i)
                    payload=encode_calibrated_gif(b,render_progress,rate.value)
                    name='calibrated-sequence.gif'
                else:
                    payload=imageio.imwrite('<bytes>',render_frame(saved_index),format='PNG')
                    name=f'frame-{b.frame_ids[saved_index]}-calibrated.png'
                event.client.send_file_download(name,payload)
                message.value='Complete'
            except Exception as error:
                message.value='Failed: '+str(error)
                raise
            finally:
                state['rendering']=False;png.disabled=gif.disabled=False
                play.value=was_playing
    png.on_click(lambda e:export(e,False));gif.on_click(lambda e:export(e,True))
    @free_png.on_click
    def _(event):
        if event.client is None:return
        with lock:
            was_playing=play.value;play.value=False
            try:
                image=event.client.get_render(height=b.height,width=b.width)
                event.client.send_file_download(f'frame-{b.frame_ids[state["index"]]}-free-view.png',
                    imageio.imwrite('<bytes>',image,format='PNG'))
            finally:play.value=was_playing
    update(0)
    print(f'Loaded {b.num_frames} frames, original IDs {b.frame_ids[0]}..{b.frame_ids[-1]}',flush=True)
    while True:
        with lock:
            i=state['index']
            delay=(float(b.timestamps[i+1]-b.timestamps[i]) if i+1<b.num_frames and
                   b.timestamps[i] is not None and b.timestamps[i+1] is not None else 1/rate.value)
        time.sleep(max(.005,delay))
        if play.value and not state['rendering']:
            with lock:slider.value=(state['index']+1)%b.num_frames

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir','--data_dir',required=True,type=Path)
    p.add_argument('--port',type=int,default=8090)
    p.add_argument('--start',type=int,default=0);p.add_argument('--end',type=int,default=-1)
    p.add_argument('--max-frames',type=int,default=256)
    p.add_argument('--downsample-factor',type=int,default=4)
    p.add_argument('--object-variant',choices=('display','optimization_proxy','source_reference'),default='display')
    p.add_argument('--device',default='cuda');p.add_argument('--share',action='store_true')
    main(**vars(p.parse_args()))
