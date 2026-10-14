# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import cv2
import numpy as np
import glob
import pickle
import trimesh
import copy
import imageio
import torch
import open3d as o3d
import logging
from transformations import euler_matrix
from PIL import Image

from optimization.utils.geometry import glcam_in_cvcam
from .depth import depth2xyzmap
import json,os,sys


HO3D_ROOT = os.environ.get('HO3D_ROOT', 'datasets/HO3D/raw')
class Ho3dReader:
  def __init__(self,video_dir):
    self.video_dir = video_dir
    print(f'Loading HO3D sequence from {video_dir} ...')
    self.color_files = sorted(glob.glob(f"{self.video_dir}/rgb/*.jpg"))
    meta_file = self.color_files[0].replace('.jpg','.pkl').replace('rgb','meta')
    self.K = pickle.load(open(meta_file,'rb'))['camMat']

    image0 = imageio.imread(self.color_files[0])
    self.H,self.W = image0.shape[:2]

    self.id_strs = []
    for i in range(len(self.color_files)):
      id = os.path.basename(self.color_files[i]).split('.')[0]
      self.id_strs.append(id)

  def __len__(self):
    return len(self.color_files)

  def get_video_name(self):
    # return os.path.dirname(os.path.abspath(self.color_files[0])).split('/')[-2]
    return self.video_dir.split('/')[-1]

  def get_color(self,i):
    color = imageio.imread(self.color_files[i])[...,:3]
    color = cv2.resize(color, (self.W,self.H), interpolation=cv2.INTER_NEAREST)
    return color

  def get_mask(self,i):
    video_name = self.get_video_name()
    index = int(os.path.basename(self.color_files[i]).split('.')[0])
    # mask = cv2.imread(f'{HO3D_ROOT}/masks_XMem/{video_name}/{index:05d}.png',-1)
    masks_dir = f'{HO3D_ROOT}/masks_SAM2/train/{video_name}'
    if not os.path.exists(masks_dir):
      masks_dir = f'{HO3D_ROOT}/masks_SAM2/evaluation/{video_name}'
    path = f'{masks_dir}/object/{index:05d}.png'
    if not os.path.exists(path):
      path = f'{masks_dir}/object/{index:04d}.png'
    mask = cv2.imread(path,-1)
    return mask

  def get_occ_mask(self,i):
    video_name = self.get_video_name()
    index = int(os.path.basename(self.color_files[i]).split('.')[0])
    # mask = cv2.imread(f'{HO3D_ROOT}/masks_XMem/{video_name}_hand/{index:04d}.png',-1)
    masks_dir = f'{HO3D_ROOT}/masks_SAM2/train/{video_name}'
    if not os.path.exists(masks_dir):
      masks_dir = f'{HO3D_ROOT}/masks_SAM2/evaluation/{video_name}'
    path = f'{masks_dir}/hand/{index:05d}.png'
    if not os.path.exists(path):
      path = f'{masks_dir}/hand/{index:04d}.png'
    mask = cv2.imread(path,-1)
    return mask

  def get_gt_mesh_path(self):
    video2name = {
      'AP': '019_pitcher_base',
      'MPM': '010_potted_meat_can',
      'SB': '021_bleach_cleanser',
      'SM': '006_mustard_bottle',
      'ABF': '021_bleach_cleanser',
      'BB': '011_banana',
      'GPMF': '010_potted_meat_can',
      'MC': '003_cracker_box',
      'MDF': '035_power_drill',
      'ShSu': '004_sugar_box',
      'SMu': '025_mug',
      'GSF': '037_scissors'
    }
    video_name = self.get_video_name()
    match_len = 0
    for k in video2name:
      if video_name.startswith(k):
        if len(k) > match_len:
          match_len = len(k)
          ob_name = video2name[k]
    mesh_path = f'{HO3D_ROOT}/models/{ob_name}/textured_simple.obj'
    return mesh_path

  def get_gt_mesh(self):
    mesh_file = self.get_gt_mesh_path()
    mesh = trimesh.load(mesh_file)
    return mesh

  def get_depth(self,i):
    color = imageio.imread(self.color_files[i])
    depth_scale = 0.00012498664727900177
    depth = cv2.imread(self.color_files[i].replace('.jpg','.png').replace('rgb','depth'), -1)
    depth = (depth[...,2]+depth[...,1]*256)*depth_scale
    return depth

  def get_xyz_map(self,i):
    depth = self.get_depth(i)
    xyz_map = depth2xyzmap(depth,self.K)
    return xyz_map

  def get_gt_pose(self,i):
    meta_file = self.color_files[i].replace('.jpg','.pkl').replace('rgb','meta')
    meta = pickle.load(open(meta_file,'rb'))
    ob_in_cam_gt = np.eye(4)
    if meta['objTrans'] is None:
      return None
    else:
      ob_in_cam_gt[:3,3] = meta['objTrans']
      ob_in_cam_gt[:3,:3] = cv2.Rodrigues(meta['objRot'].reshape(3))[0]
      ob_in_cam_gt = glcam_in_cvcam@ob_in_cam_gt
    return ob_in_cam_gt

  # ...existing code...

  def get_mano_data(self, i, mano_model=None):
    meta_file = self.color_files[i].replace('.jpg','.pkl').replace('rgb','meta')
    meta = pickle.load(open(meta_file,'rb'))

    if meta['handBeta'] is None:
        return None

    from optimization.utils.mano_coordinates import convert_mano_camera
    if mano_model is None:
        if not hasattr(self, '_mano_conversion_model'):
            import yaml
            from optimization.models.mano_wrapper import MANO
            root = os.environ.get('HO3D_MANO_ROOT', 'assets/mano')
            with open(os.path.join(root, 'mano_config.yaml')) as f:
                cfg = yaml.safe_load(f)['MANO']
            self._mano_conversion_model = MANO(**{k.lower(): v for k, v in cfg.items()}).to('cpu')
        mano_model = self._mano_conversion_model
    hand_pose_cv, hand_trans_cv = convert_mano_camera(
        meta['handPose'], meta['handTrans'], meta['handBeta'], mano_model)

    return {
        'handBeta': meta['handBeta'],
        'handPose': hand_pose_cv,
        'handTrans': hand_trans_cv,
        'handJoints3d': meta['handJoints3D'],
    }
