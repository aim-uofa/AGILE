"""Image-based initialization/recovery; no 3D annotations are consumed."""
import numpy as np
import torch
from scipy.optimize import minimize


def fixed_rotation_translation(joints_in_camera_axes, image_joints, K, initial_t):
    """Solve translation with a fixed, already rotated/scaled hand skeleton.

    Linear perspective equations initialize a pixel-distance refinement. Keep
    the prepared translation unless a finite, positive-depth solution improves
    the actual mean joint reprojection distance. Degenerate inputs fall back.
    """
    device, dtype = initial_t.device, initial_t.dtype
    xyz = joints_in_camera_axes.detach().double().cpu().numpy()
    uv = image_joints.detach().double().cpu().numpy()
    camera = K.detach().double().cpu().numpy()
    initial = initial_t.detach().double().cpu().numpy()
    if not all(np.isfinite(v).all() for v in (xyz, uv, camera, initial)):
        return initial_t.detach().clone()
    if len(xyz) < 4 or np.linalg.matrix_rank(xyz - xyz.mean(0), tol=1e-8) < 2:
        return initial_t.detach().clone()
    normalized = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(camera).T
    normalized = normalized[:, :2] / normalized[:, 2:]
    a = np.zeros((len(xyz), 2, 3)); a[:,0,0] = a[:,1,1] = 1
    a[:,:,2] = -normalized
    b = normalized * xyz[:,2:] - xyz[:,:2]
    linear, _, rank, _ = np.linalg.lstsq(a.reshape(-1,3), b.ravel(), rcond=None)
    if rank < 3:
        return initial_t.detach().clone()

    def objective(t):
        points = xyz + t
        if np.min(points[:,2]) <= 1e-5:
            return 1e12
        projected = points @ camera.T
        residual = projected[:,:2] / projected[:,2:] - uv
        return np.sqrt((residual**2).sum(1) + 1e-12).mean()

    candidates = [initial, linear]
    if objective(linear) < 1e12:
        fitted = minimize(objective, linear, method='BFGS', options={'maxiter':100, 'gtol':1e-6})
        candidates.append(fitted.x)
    best = min(candidates, key=objective)
    if not np.isfinite(best).all() or objective(best) >= objective(initial):
        return initial_t.detach().clone()
    return torch.as_tensor(best, device=device, dtype=dtype)


def visible_mask_iou(alpha, object_mask, hand_mask):
    visible = ~hand_mask.bool()
    prediction = (alpha > .5) & visible
    observed = object_mask.bool() & visible
    if not bool(observed.any()):
        return None
    union = (prediction | observed).sum()
    return float(((prediction & observed).sum() / union.clamp_min(1)).item())


@torch.no_grad()
def recover_object_initialization(runner, frame_id, base_R, base_t, Ks, height, width,
                                  object_mask, hand_mask):
    """Compare a lost propagated pose against the existing per-frame NAS pose.

    The candidate shares the current object scale. Its translation is scaled
    along the camera ray so that its original projection is preserved. Select
    it only when visible-mask IoU improves; temporal hand contact is disabled
    for this recovery frame because the previous object track was lost.
    """
    cfg = runner.cfg
    def score(R,t):
        runner.set_obj_base_pose(frame_id,R,t)
        output = runner.render_meshes(camtoworlds=torch.eye(4,device=base_t.device)[None],
            Ks=Ks,height=height,width=width,frame_id=frame_id,render_objects=['object'],
            hard_rasterization=True)
        return visible_mask_iou(output['alpha'][0,...,0],object_mask,hand_mask)
    current_score = score(base_R,base_t)
    if current_score is None or current_score >= cfg.object_recovery_min_iou:
        return base_R,base_t,False
    prepared = runner.trainset[frame_id]['object_pose'].to(base_t.device)
    scale = getattr(runner,'optimized_obj_scale',None)
    factor = 1.
    if scale is not None:
        if not bool(torch.isfinite(scale).all() and (scale > 0).all()):
            raise ValueError('Object recovery requires a positive finite scale')
        factor = torch.exp(torch.log(scale).mean())
    candidate_R, candidate_t = prepared[:3,:3], prepared[:3,3] * factor
    candidate_score = score(candidate_R,candidate_t) if candidate_t[2] > 0 else None
    recovered = candidate_score is not None and candidate_score >= cfg.object_recovery_min_iou and candidate_score > current_score
    selected_R, selected_t = (candidate_R,candidate_t) if recovered else (base_R,base_t)
    runner.set_obj_base_pose(frame_id,selected_R,selected_t)
    event = dict(frame_id=runner.get_actual_frame_id(frame_id), propagated_iou=current_score,
                 prepared_iou=candidate_score, recovered=recovered)
    if not hasattr(runner,'object_recovery_events'):
        runner.object_recovery_events=[]
    runner.object_recovery_events.append(event)
    print('[Object recovery]',event,flush=True)
    return selected_R.detach().clone(),selected_t.detach().clone(),recovered
