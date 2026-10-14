"""MANO camera conversion must rotate around its shape-dependent rest wrist."""
import cv2
import numpy as np
import torch

OPENGL_TO_OPENCV = np.diag([1., -1., -1.])


def convert_mano_camera(pose, translation, betas, model):
    pose_cv = np.asarray(pose).copy()
    device, dtype = model.shapedirs.device, model.shapedirs.dtype
    beta = torch.as_tensor(betas, device=device, dtype=dtype)
    with torch.no_grad():
        shaped = model.v_template + torch.einsum('vci,i->vc', model.shapedirs, beta)
        rest_wrist = (model.J_regressor[0] @ shaped).cpu().numpy()
    pose_cv[:3] = cv2.Rodrigues(OPENGL_TO_OPENCV @ cv2.Rodrigues(pose_cv[:3])[0])[0].ravel()
    translation_cv = OPENGL_TO_OPENCV @ (np.asarray(translation).reshape(3) + rest_wrist) - rest_wrist
    return pose_cv, translation_cv
