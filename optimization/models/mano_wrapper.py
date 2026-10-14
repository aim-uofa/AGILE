# MANOLayer is adapted from SMPL-X. Its original license and copyright notice
# are retained in LICENSES/SMPL-X.txt and LICENSES/SMPL-X-NOTICE.txt.
# The MANO joint adapter is adapted from WiLoR / HaMeR.
# See LICENSES/SOURCES.md for the upstream source references.

import torch
import numpy as np
import pickle
from typing import Optional
import smplx
from smplx.lbs import vertices2joints, lbs
from smplx.utils import MANOOutput, to_tensor, Tensor
from smplx.vertex_ids import vertex_ids
from smplx import MANO as SMPLX_MANO

class MANOLayer(SMPLX_MANO):
    def __init__(self, *args, **kwargs) -> None:
        ''' MANO as a layer model constructor
        '''
        super(MANOLayer, self).__init__(
            create_global_orient=False,
            create_hand_pose=False,
            create_betas=False,
            create_transl=False,
            *args, **kwargs)

    def name(self) -> str:
        return 'MANO'

    def forward(
        self,
        betas: Optional[Tensor] = None,
        global_orient: Optional[Tensor] = None,
        hand_pose: Optional[Tensor] = None,
        transl: Optional[Tensor] = None,
        return_verts: bool = True,
        return_full_pose: bool = False,
        **kwargs
    ) -> MANOOutput:
        ''' Forward pass for the MANO model
        '''
        device, dtype = self.shapedirs.device, self.shapedirs.dtype

        if global_orient is None:
            if hand_pose is None:
                if betas is None:
                    if transl is None:
                        batch_size = 1
                    else:
                        batch_size = transl.shape[0]
                else:
                    batch_size = betas.shape[0]
            else:
                batch_size = hand_pose.shape[0]
        else:
            batch_size = global_orient.shape[0]

        if global_orient is None:
            # batch_size = 1
            global_orient = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, -1, -1, -1).contiguous()
        # else:
        #     batch_size = global_orient.shape[0]
        if hand_pose is None:
            hand_pose = torch.eye(3, device=device, dtype=dtype).view(
                1, 1, 3, 3).expand(batch_size, 15, -1, -1).contiguous()
        if betas is None:
            betas = torch.zeros(
                [batch_size, self.num_betas], dtype=dtype, device=device)
        if transl is None:
            transl = torch.zeros([batch_size, 3], dtype=dtype, device=device)

        full_pose = torch.cat([global_orient, hand_pose], dim=1) #(B, 16, 3, 3)
        vertices, joints = lbs(betas, full_pose, self.v_template,
                               self.shapedirs, self.posedirs,
                               self.J_regressor, self.parents,
                               self.lbs_weights, pose2rot=False)

        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if transl is not None:
            joints = joints + transl.unsqueeze(dim=1)
            vertices = vertices + transl.unsqueeze(dim=1)

        output = MANOOutput(
            vertices=vertices if return_verts else None,
            joints=joints if return_verts else None,
            betas=betas,
            global_orient=global_orient,
            hand_pose=hand_pose,
            full_pose=full_pose if return_full_pose else None)

        return output


class MANO(MANOLayer):
    def __init__(self, *args, joint_regressor_extra: Optional[str] = None, **kwargs):
        """
        Extension of the official MANO implementation to support more joints.
        Args:
            Same as MANOLayer.
            joint_regressor_extra (str): Path to extra joint regressor.
        """
        super(MANO, self).__init__(*args, **kwargs)
        mano_to_openpose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]

        #2, 3, 5, 4, 1
        if joint_regressor_extra is not None:
            self.register_buffer('joint_regressor_extra', torch.tensor(pickle.load(open(joint_regressor_extra, 'rb'), encoding='latin1'), dtype=torch.float32))
        self.register_buffer('extra_joints_idxs', to_tensor(list(vertex_ids['mano'].values()), dtype=torch.long))
        self.register_buffer('joint_map', torch.tensor(mano_to_openpose, dtype=torch.long))

    def forward(self, *args, **kwargs) -> MANOOutput:
        """
        Run forward pass. Same as MANO and also append an extra set of joints if joint_regressor_extra is specified.
        """
        mano_output = super(MANO, self).forward(*args, **kwargs)
        extra_joints = torch.index_select(mano_output.vertices, 1, self.extra_joints_idxs)
        joints = torch.cat([mano_output.joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]
        if hasattr(self, 'joint_regressor_extra'):
            extra_joints = vertices2joints(self.joint_regressor_extra, mano_output.vertices)
            joints = torch.cat([joints, extra_joints], dim=1)
        mano_output.joints = joints
        return mano_output
